"""Several market makers competing for the same order flow in one order book.

Every maker quotes into the same book against the same synthetic taker flow, so
one maker's fill is a fill another maker did not get. The observation layout is
byte-for-byte the single-agent layout, which is what lets a policy trained alone
and a policy trained under competition meet in the same book (see `league.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from gymnasium import spaces

from lobrl._lobcore import RING_SIZE, EventType, OrderBook, Side
from lobrl.flow import FlowConfig, MarketFlow

OWNER_NOISE = 0
OWNER_BASE = 1  # agent i owns orders tagged OWNER_BASE + i


@dataclass
class MultiEnvConfig:
    n_agents: int = 4
    levels: int = 5
    max_steps: int = 1_000
    init_mid: int = 10_000
    quote_size: int = 20
    # When variable_size is on the action grows to
    # [delta_bid, delta_ask, size_bid, size_ask] and each side's size is mapped
    # onto [min_quote_size, max_quote_size]. A policy that outputs 0 for both
    # size components quotes exactly `quote_size`, so the 2-d action space is a
    # strict subset of the 4-d one and old policies stay comparable.
    # Order-flow ("tape") features. The synthetic market has real structure the
    # agent otherwise cannot see: aggressive orders arrive in Hawkes clusters and
    # their sign leans with book imbalance. Without these the policy sees the
    # book but never the tape, so that structure is unobservable to it.
    flow_features: bool = False
    # Path to a supervised fundamental-gap estimator. Its output is appended to
    # the observation as one extra feature, so the policy only has to learn a
    # simple monotone response instead of extracting the signal itself.
    gap_predictor: str | None = None
    # Signed-volume EWMAs, one per timescale. Adding timescales is supported and
    # measurably improves the fundamental-gap estimate a little -- four scales
    # (0.30, 0.15, 0.06, 0.015) took the estimator from corr 0.476 to 0.493 --
    # but that is worth only ~+0.5 Sharpe and changes the observation width, so
    # the default stays at two to keep the benchmark and its champion stable.
    flow_alphas: tuple = (0.15, 0.03)
    variable_size: bool = False
    min_quote_size: int = 2
    max_quote_size: int = 40
    max_inventory: int = 200
    min_offset: float = 0.5
    max_offset: float = 10.0
    inventory_penalty: float = 5e-3   # phi
    turnover_penalty: float = 1e-3    # eta
    vol_window: int = 50
    # Submission order is reshuffled every step. Without this, whichever agent
    # submits first always wins the queue at a shared price and the ranking
    # measures list position instead of skill.
    randomize_order: bool = True
    flow: FlowConfig = field(default_factory=FlowConfig)


class MultiAgentMarketMakingEnv:
    """Simultaneous-move market making.

    `step(actions)` takes an (n_agents, 2) array of half-spread offsets and
    returns per-agent observations and rewards. All agents quote against the
    same pre-step mid price, so no agent gets to react to another's quote
    within a step; competition resolves through the queue, not through ordering.
    """

    def __init__(self, cfg: MultiEnvConfig | None = None, seed: int | None = None):
        self.cfg = cfg or MultiEnvConfig()
        n, K = self.cfg.n_agents, self.cfg.levels
        if not 1 <= n <= 254:
            raise ValueError("n_agents must be between 1 and 254")

        self.book = OrderBook()
        self.flow = MarketFlow(self.cfg.flow, np.random.default_rng(seed))
        self.rng = np.random.default_rng(seed)

        self._bid_px = np.zeros(K, dtype=np.int32)
        self._bid_vol = np.zeros(K, dtype=np.int64)
        self._ask_px = np.zeros(K, dtype=np.int32)
        self._ask_vol = np.zeros(K, dtype=np.int64)

        self.n_flow = (len(self.cfg.flow_alphas) + 2) if self.cfg.flow_features else 0
        self._gap_net, self._gap_cols, self._gap_sd = None, None, 1.0
        if self.cfg.gap_predictor:
            from lobrl.gap import load as load_gap  # noqa: PLC0415
            self._gap_net, self._gap_cols, self._gap_sd = load_gap(self.cfg.gap_predictor)
        self.n_pred = 1 if self._gap_net is not None else 0
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(4 * K + 8 + self.n_flow + self.n_pred,), dtype=np.float32)
        self.act_dim = 4 if self.cfg.variable_size else 2
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.act_dim,), dtype=np.float32)
        self.n_agents = n

        self._events = None
        self._reset_state()

    # ---- lifecycle -----------------------------------------------------

    def _reset_state(self) -> None:
        n = self.n_agents
        self.inventory = np.zeros(n, dtype=np.int64)
        self.cash = np.zeros(n, dtype=np.float64)
        self.realized = np.zeros(n, dtype=np.float64)
        self.avg_price = np.zeros(n, dtype=np.float64)
        self.prev_value = np.zeros(n, dtype=np.float64)
        self.bid_id = np.zeros(n, dtype=np.int64)
        self.ask_id = np.zeros(n, dtype=np.int64)
        self.quote_bid_px = np.full(n, -1, dtype=np.int64)
        self.quote_ask_px = np.full(n, -1, dtype=np.int64)
        self.trade_count = np.zeros(n, dtype=np.int64)
        self.volume_traded = np.zeros(n, dtype=np.int64)
        self.step_count = 0
        self._last_event_total = 0
        self._mid_hist = np.full(self.cfg.vol_window, float(self.cfg.init_mid))
        self._hist_n = 0
        self.fill_log: list[tuple[int, int, float, int]] = []  # (agent, step, price, signed)
        # Tape state: signed volume at each timescale, plus arrival intensity
        # and mean trade size.
        self.f_vol = np.zeros(len(self.cfg.flow_alphas))
        self.f_rate = 0.0
        self.f_size = 0.0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.flow.rng = np.random.default_rng(seed)
            self.rng = np.random.default_rng(seed)
        self.book.reset()
        self._events = self.book.events_buffer()
        self._reset_state()
        self.flow.seed_book(self.book, self.cfg.init_mid)
        for _ in range(20):
            self.flow.step(self.book)
        self._last_event_total = self.book.events_total()
        mid = float(self.book.mid())
        self.prev_value[:] = self.cash + self.inventory * mid
        return self._obs(), self._info()

    # ---- step ----------------------------------------------------------

    def step(self, actions):
        cfg = self.cfg
        a = np.clip(np.asarray(actions, dtype=np.float64).reshape(self.n_agents, self.act_dim),
                    -1.0, 1.0)
        lo, hi = cfg.min_offset, cfg.max_offset
        offsets = lo + (a[:, :2] + 1.0) * 0.5 * (hi - lo)
        sizes = self._sizes(a)

        self._cancel_all()
        mid = float(self.book.mid())          # one snapshot price for every agent
        order = np.arange(self.n_agents)
        if cfg.randomize_order:
            self.rng.shuffle(order)
        for i in order:
            self._place(int(i), mid, offsets[i, 0], offsets[i, 1], sizes[i, 0], sizes[i, 1])

        self.flow.step(self.book)
        d_inv = self._apply_fills()

        new_mid = float(self.book.mid())
        self._push_mid(new_mid)

        value = self.cash + self.inventory * new_mid
        d_pnl = value - self.prev_value
        rewards = (
            d_pnl
            - cfg.inventory_penalty * self.inventory.astype(np.float64) ** 2
            - cfg.turnover_penalty * d_inv.astype(np.float64) ** 2
        )
        self.prev_value = value
        self.step_count += 1

        truncated = self.step_count >= cfg.max_steps
        info = self._info()
        info["offsets"] = offsets
        info["sizes"] = sizes
        info["d_pnl"] = d_pnl
        return self._obs(), rewards.astype(np.float64), False, truncated, info

    # ---- order handling -------------------------------------------------

    def _cancel_all(self) -> None:
        for i in range(self.n_agents):
            for arr in (self.bid_id, self.ask_id):
                if arr[i] > 0:
                    self.book.cancel(int(arr[i]))
                    arr[i] = 0
        self.quote_bid_px[:] = -1
        self.quote_ask_px[:] = -1

    def _sizes(self, a: np.ndarray) -> np.ndarray:
        """Map the size half of the action onto share counts.

        Action 0 maps to `quote_size` exactly, so a 2-d policy padded with zeros
        reproduces the fixed-size behaviour it was trained with.
        """
        cfg = self.cfg
        if not cfg.variable_size:
            return np.full((self.n_agents, 2), cfg.quote_size, dtype=np.int64)
        u = a[:, 2:4]
        below = cfg.quote_size - cfg.min_quote_size
        above = cfg.max_quote_size - cfg.quote_size
        scaled = np.where(u >= 0, cfg.quote_size + u * above, cfg.quote_size + u * below)
        return np.clip(np.rint(scaled), cfg.min_quote_size, cfg.max_quote_size).astype(np.int64)

    def _place(self, i: int, mid: float, d_bid: float, d_ask: float,
               sz_bid: int = 0, sz_ask: int = 0) -> None:
        cfg = self.cfg
        owner = OWNER_BASE + i
        sz_bid = int(sz_bid) or cfg.quote_size
        sz_ask = int(sz_ask) or cfg.quote_size
        # The clamp reads the live book, so a maker can never cross a rival's
        # resting quote -- every agent order is passive by construction.
        if self.inventory[i] < cfg.max_inventory:
            px = int(round(mid - d_bid))
            ba = self.book.best_ask()
            if ba >= 0:
                px = min(px, ba - 1)
            if px > 0:
                oid = self.book.limit(Side.BID, px, sz_bid, owner)
                if oid > 0:
                    self.bid_id[i] = oid
                    self.quote_bid_px[i] = px
        if self.inventory[i] > -cfg.max_inventory:
            px = int(round(mid + d_ask))
            bb = self.book.best_bid()
            if bb >= 0:
                px = max(px, bb + 1)
            oid = self.book.limit(Side.ASK, px, sz_ask, owner)
            if oid > 0:
                self.ask_id[i] = oid
                self.quote_ask_px[i] = px

    def _apply_fills(self) -> np.ndarray:
        d_inv = np.zeros(self.n_agents, dtype=np.int64)
        total = self.book.events_total()
        n_new = min(total - self._last_event_total, RING_SIZE)
        self._last_event_total = total
        if n_new <= 0:
            return d_inv

        idx = ((total - n_new) + np.arange(n_new)) % RING_SIZE
        ev = self._events[idx]
        all_fills = ev[ev["type"] == int(EventType.FILL)]
        self._update_tape(all_fills)
        fills = all_fills[all_fills["owner"] >= OWNER_BASE]
        for f in fills:
            i = int(f["owner"]) - OWNER_BASE
            if not 0 <= i < self.n_agents:
                continue
            qty = int(f["qty"])
            price = float(f["price"])
            signed = qty if int(f["side"]) == int(Side.BID) else -qty
            self._book_trade(i, signed, price)
            self.cash[i] -= signed * price
            d_inv[i] += signed
            self.trade_count[i] += 1
            self.volume_traded[i] += qty
            self.fill_log.append((i, self.step_count, price, signed))
        return d_inv

    def _update_tape(self, fills) -> None:
        """Fold this step's market-wide trades into the tape EWMAs.

        A fill whose resting side is ASK means a buyer lifted the offer, so it is
        buy-side flow; a resting BID means a seller hit the bid.
        """
        cfg = self.cfg
        if len(fills):
            qty = fills["qty"].astype(np.float64)
            sign = np.where(fills["side"] == int(Side.ASK), 1.0, -1.0)
            signed_vol = float((qty * sign).sum())
            n = float(len(fills))
            mean_size = float(qty.mean())
        else:
            signed_vol, n, mean_size = 0.0, 0.0, 0.0
        alphas = np.asarray(cfg.flow_alphas)
        self.f_vol += alphas * (signed_vol - self.f_vol)
        self.f_rate += alphas[1] * (n - self.f_rate)
        self.f_size += 0.1 * (mean_size - self.f_size)

    def _book_trade(self, i: int, signed: int, price: float) -> None:
        """Average-cost accounting for agent i."""
        q = int(self.inventory[i])
        if q == 0 or (q > 0) == (signed > 0):
            new_q = q + signed
            self.avg_price[i] = ((self.avg_price[i] * q + price * signed) / new_q) if new_q else 0.0
            self.inventory[i] = new_q
            return
        closing = min(abs(signed), abs(q))
        self.realized[i] += closing * (price - self.avg_price[i]) * (1.0 if q > 0 else -1.0)
        new_q = q + signed
        if new_q == 0:
            self.avg_price[i] = 0.0
        elif (new_q > 0) != (q > 0):
            self.avg_price[i] = price
        self.inventory[i] = new_q

    # ---- observation ----------------------------------------------------

    def _push_mid(self, mid: float) -> None:
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
        # The book half of the observation is identical for every agent, so the
        # snapshot is taken once and only the private tail varies.
        self.book.snapshot_into(self._bid_px, self._bid_vol, self._ask_px, self._ask_vol)
        mid = float(self.book.mid())
        bpx = np.where(self._bid_px < 0, 0.0, mid - self._bid_px) / cfg.max_offset
        apx = np.where(self._ask_px < 0, 0.0, self._ask_px - mid) / cfg.max_offset
        bvol = np.log1p(self._bid_vol.astype(np.float64))
        avol = np.log1p(self._ask_vol.astype(np.float64))
        shared = np.concatenate([bpx, bvol, apx, avol])

        vb, va = float(self._bid_vol.sum()), float(self._ask_vol.sum())
        imbalance = (vb - va) / (vb + va) if (vb + va) > 0 else 0.0
        spread = self.book.spread()
        spread = float(spread) if spread > 0 else 0.0
        vol = self._volatility()

        flow = ()
        if self.n_flow:
            # Squash to keep the scale comparable with the other features; the
            # running normalizer handles the rest.
            flow = tuple(np.tanh(self.f_vol / 200.0)) + (
                np.tanh(self.f_rate / 3.0),
                np.tanh(self.f_size / 60.0),
            )

        out = np.empty((self.n_agents, 4 * cfg.levels + 8 + self.n_flow + self.n_pred),
                       dtype=np.float32)
        width = 4 * cfg.levels + 8 + self.n_flow
        for i in range(self.n_agents):
            unrealized = self.cash[i] + self.inventory[i] * mid
            out[i, :width] = np.concatenate([
                shared,
                [
                    imbalance,
                    self.inventory[i] / cfg.max_inventory,
                    spread / cfg.max_offset,
                    vol,
                    self.realized[i] / 1000.0,
                    unrealized / 1000.0,
                    self._queue_ratio(int(self.bid_id[i])),
                    self._queue_ratio(int(self.ask_id[i])),
                ],
                flow,
            ])
        if self.n_pred:
            out[:, -1] = self._predict_gap(out[0])
        return out

    def _predict_gap(self, row: np.ndarray) -> float:
        """One market-wide estimate per step, shared by every agent."""
        import torch

        x = torch.as_tensor(row[self._gap_cols][None, :], dtype=torch.float32)
        with torch.no_grad():
            return float(np.tanh(float(self._gap_net(x)[0]) / (2.0 * self._gap_sd)))

    def _queue_ratio(self, oid: int) -> float:
        if oid <= 0 or not self.book.is_live(oid):
            return -1.0
        return float(np.log1p(max(self.book.queue_ahead(oid), 0)) / 10.0)

    def _info(self) -> dict:
        mid = float(self.book.mid())
        return {
            "mid": mid,
            "inventory": self.inventory.copy(),
            "cash": self.cash.copy(),
            "equity": self.cash + self.inventory * mid,
            "trades": self.trade_count.copy(),
            "volume": self.volume_traded.copy(),
            "spread": self.book.spread(),
            # Training-time-only label: never enters the observation, only the
            # auxiliary supervised head. The policy still trades on what it can
            # actually see.
            "fundamental_gap": float(self.flow.fundamental - mid) if self.cfg.flow.fundamental_vol > 0 else 0.0,
            "quote_bid": self.quote_bid_px.copy(),
            "quote_ask": self.quote_ask_px.copy(),
        }
