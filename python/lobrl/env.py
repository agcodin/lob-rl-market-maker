"""Gymnasium environment wrapping the C++ matching engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lobrl._lobcore import RING_SIZE, EventType, OrderBook, Owner, Side
from lobrl.flow import FlowConfig, MarketFlow


@dataclass
class EnvConfig:
    levels: int = 5                # K in the level-2 snapshot
    max_steps: int = 2_000
    init_mid: int = 10_000         # ticks (cents)
    quote_size: int = 20
    max_inventory: int = 200
    min_offset: float = 0.5        # half-spread bounds, in ticks
    max_offset: float = 10.0
    inventory_penalty: float = 1e-3   # phi
    turnover_penalty: float = 1e-3    # eta
    vol_window: int = 50
    reward_scale: float = 1.0
    terminate_on_inventory: bool = False
    flow: FlowConfig = field(default_factory=FlowConfig)


class MarketMakingEnv(gym.Env):
    """One step = submit two quotes, then advance the market by one tick.

    Observation (float32):
        [K bid offsets, K bid log-volumes, K ask offsets, K ask log-volumes,
         imbalance, inventory/Qmax, spread, volatility, realized PnL, unrealized PnL,
         bid queue-ahead ratio, ask queue-ahead ratio]
    Action (float32, Box(-1, 1, (2,))): [delta_bid, delta_ask] half-spread offsets,
        affinely mapped onto [min_offset, max_offset] ticks.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig | None = None, seed: int | None = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        K = self.cfg.levels
        self.book = OrderBook()
        self.flow = MarketFlow(self.cfg.flow, np.random.default_rng(seed))

        # Snapshot buffers are allocated once and written in place by C++.
        self._bid_px = np.zeros(K, dtype=np.int32)
        self._bid_vol = np.zeros(K, dtype=np.int64)
        self._ask_px = np.zeros(K, dtype=np.int32)
        self._ask_vol = np.zeros(K, dtype=np.int64)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4 * K + 8,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._events = None
        self._reset_state()

    # ---- lifecycle -----------------------------------------------------

    def _reset_state(self) -> None:
        self.inventory = 0
        self.cash = 0.0
        self.realized = 0.0
        self.prev_value = 0.0
        self.prev_mid = float(self.cfg.init_mid)
        self.bid_id = 0
        self.ask_id = 0
        self.step_count = 0
        self._last_event_total = 0
        self._mid_hist = np.full(self.cfg.vol_window, float(self.cfg.init_mid))
        self._hist_n = 0
        self.avg_price = 0.0
        self.trade_count = 0
        self.volume_traded = 0
        self.adverse_log: list[tuple[int, float, int]] = []  # (step, price, signed qty)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.flow.rng = np.random.default_rng(seed)
        self.book.reset()
        self._events = self.book.events_buffer()
        self._reset_state()
        self.flow.seed_book(self.book, self.cfg.init_mid)
        # Burn in so the agent never sees the artificial seeded ladder.
        for _ in range(20):
            self.flow.step(self.book)
        self._last_event_total = self.book.events_total()
        self.prev_mid = float(self.book.mid())
        self.prev_value = self.cash + self.inventory * self.prev_mid
        return self._obs(), self._info()

    # ---- step ----------------------------------------------------------

    def step(self, action):
        cfg = self.cfg
        action = np.asarray(action, dtype=np.float64).reshape(2)
        lo, hi = cfg.min_offset, cfg.max_offset
        offsets = lo + (np.clip(action, -1.0, 1.0) + 1.0) * 0.5 * (hi - lo)

        self._cancel_quotes()
        mid = self.book.mid()
        self._place_quotes(mid, offsets[0], offsets[1])

        self.flow.step(self.book)
        d_inv, fills = self._apply_fills()

        new_mid = float(self.book.mid())
        self._push_mid(new_mid)

        value = self.cash + self.inventory * new_mid
        d_pnl = value - self.prev_value
        reward = cfg.reward_scale * (
            d_pnl
            - cfg.inventory_penalty * float(self.inventory) ** 2
            - cfg.turnover_penalty * float(d_inv) ** 2
        )
        self.prev_value = value
        self.prev_mid = new_mid
        self.step_count += 1

        terminated = bool(
            cfg.terminate_on_inventory and abs(self.inventory) > cfg.max_inventory
        )
        truncated = self.step_count >= cfg.max_steps
        info = self._info()
        info.update({"fills": fills, "d_pnl": d_pnl, "delta": offsets.copy()})
        return self._obs(), float(reward), terminated, truncated, info

    def close(self):
        self._cancel_quotes()

    # ---- order handling -------------------------------------------------

    def _cancel_quotes(self) -> None:
        for attr in ("bid_id", "ask_id"):
            oid = getattr(self, attr)
            if oid > 0:
                self.book.cancel(oid)
                setattr(self, attr, 0)

    def _place_quotes(self, mid: float, d_bid: float, d_ask: float) -> None:
        cfg = self.cfg
        size = cfg.quote_size
        best_bid, best_ask = self.book.best_bid(), self.book.best_ask()

        # Quote only on the side that reduces (or does not worsen) a maxed-out book.
        if self.inventory < cfg.max_inventory:
            px = int(round(mid - d_bid))
            if best_ask >= 0:
                px = min(px, best_ask - 1)  # stay passive: never cross
            if px > 0:
                oid = self.book.limit(Side.BID, px, size, Owner.AGENT)
                self.bid_id = oid if oid > 0 else 0
        if self.inventory > -cfg.max_inventory:
            px = int(round(mid + d_ask))
            if best_bid >= 0:
                px = max(px, best_bid + 1)
            oid = self.book.limit(Side.ASK, px, size, Owner.AGENT)
            self.ask_id = oid if oid > 0 else 0

    def _apply_fills(self):
        """Replay new ring records; returns (inventory delta, fill count)."""
        total = self.book.events_total()
        n_new = min(total - self._last_event_total, RING_SIZE)
        if n_new <= 0:
            self._last_event_total = total
            return 0, 0
        start = (total - n_new) % RING_SIZE
        idx = (start + np.arange(n_new)) % RING_SIZE
        ev = self._events[idx]
        mask = (ev["type"] == int(EventType.FILL)) & (ev["owner"] == int(Owner.AGENT))
        fills = ev[mask]
        d_inv = 0
        for f in fills:
            qty = int(f["qty"])
            price = float(f["price"])
            signed = qty if int(f["side"]) == int(Side.BID) else -qty
            self._book_trade(signed, price)
            self.cash -= signed * price
            d_inv += signed
            self.trade_count += 1
            self.volume_traded += qty
            self.adverse_log.append((self.step_count, price, signed))
        self._last_event_total = total
        return d_inv, int(mask.sum())

    def _book_trade(self, signed: int, price: float) -> None:
        """Average-cost accounting: closing volume realizes PnL, opening re-averages."""
        q = self.inventory
        if q == 0 or (q > 0) == (signed > 0):
            new_q = q + signed
            self.avg_price = (self.avg_price * q + price * signed) / new_q if new_q != 0 else 0.0
            self.inventory = new_q
            return
        closing = min(abs(signed), abs(q))
        self.realized += closing * (price - self.avg_price) * (1.0 if q > 0 else -1.0)
        new_q = q + signed
        if (new_q > 0) != (q > 0) and new_q != 0:
            self.avg_price = price  # flipped through zero
        elif new_q == 0:
            self.avg_price = 0.0
        self.inventory = new_q

    # ---- observation ----------------------------------------------------

    def _push_mid(self, mid: float) -> None:
        # Chronological ring: oldest at index 0, newest last, so diffs are valid.
        self._mid_hist[:-1] = self._mid_hist[1:]
        self._mid_hist[-1] = mid
        self._hist_n += 1

    def _volatility(self) -> float:
        n = min(self._hist_n, self.cfg.vol_window)
        if n < 3:
            return 0.0
        return float(np.std(np.diff(self._mid_hist[-n:])))

    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        self.book.snapshot_into(self._bid_px, self._bid_vol, self._ask_px, self._ask_vol)
        mid = float(self.book.mid())

        bpx = np.where(self._bid_px < 0, 0.0, mid - self._bid_px) / cfg.max_offset
        apx = np.where(self._ask_px < 0, 0.0, self._ask_px - mid) / cfg.max_offset
        bvol = np.log1p(self._bid_vol.astype(np.float64))
        avol = np.log1p(self._ask_vol.astype(np.float64))

        vb, va = float(self._bid_vol.sum()), float(self._ask_vol.sum())
        imbalance = (vb - va) / (vb + va) if (vb + va) > 0 else 0.0
        spread = self.book.spread()
        spread = float(spread) if spread > 0 else 0.0

        unrealized = self.cash + self.inventory * mid
        obs = np.concatenate(
            [
                bpx, bvol, apx, avol,
                [
                    imbalance,
                    self.inventory / cfg.max_inventory,
                    spread / cfg.max_offset,
                    self._volatility(),
                    self.realized / 1000.0,
                    unrealized / 1000.0,
                    self._queue_ratio(self.bid_id),
                    self._queue_ratio(self.ask_id),
                ],
            ]
        )
        return obs.astype(np.float32)

    def _queue_ratio(self, oid: int) -> float:
        if oid <= 0 or not self.book.is_live(oid):
            return -1.0
        ahead = self.book.queue_ahead(oid)
        return float(np.log1p(max(ahead, 0)) / 10.0)

    def _info(self) -> dict:
        mid = float(self.book.mid())
        return {
            "mid": mid,
            "inventory": self.inventory,
            "cash": self.cash,
            "equity": self.cash + self.inventory * mid,
            "trades": self.trade_count,
            "volume": self.volume_traded,
            "spread": self.book.spread(),
        }
