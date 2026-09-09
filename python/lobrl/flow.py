"""Synthetic order flow: Poisson liquidity, queue-dependent cancels, Hawkes takers.

The generator only ever talks to the engine through public order operations, so
the agent faces exactly the same matching rules as the background traders.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lobrl._lobcore import Owner, Side


@dataclass
class FlowConfig:
    # Passive liquidity: Poisson point process, one intensity per side.
    lambda_limit: float = 20.0
    # Placement depth behind the touch, in ticks (geometric tail).
    depth_scale: float = 2.5
    max_depth: int = 20
    limit_size_mean: float = 20.0
    # Fraction of passive orders that improve the touch by a tick. This is what
    # closes the spread after a sweep; too high and the book pins to one tick.
    p_improve: float = 0.04

    # Cancellations: lambda_C(d) = alpha / d**beta, d = depth from best quote.
    cancel_alpha: float = 0.35
    cancel_beta: float = 0.7

    # Hawkes market-order arrivals: lambda(t) = mu + sum alpha*exp(-beta*(t-ti)).
    hawkes_mu: float = 2.0
    hawkes_alpha: float = 0.6
    hawkes_beta: float = 1.6
    # Power-law market-order sizes: P(V > x) ~ x**-tail_index.
    size_xmin: float = 10.0
    size_tail_index: float = 1.5
    size_cap: int = 800

    # Sign of aggressive flow leans with book imbalance (order-flow feedback).
    imbalance_bias: float = 0.35

    tick: int = 1
    dt: float = 1.0


@dataclass
class MarketFlow:
    cfg: FlowConfig = field(default_factory=FlowConfig)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    hawkes_state: float = 0.0
    _ref: int = 0

    # ---- setup ---------------------------------------------------------

    def seed_book(self, book, mid_tick: int, levels: int = 12, size: int = 60) -> None:
        """Fill a fresh book with a symmetric ladder so step 0 has two sides."""
        self._ref = int(mid_tick)
        self.hawkes_state = 0.0
        half = max(1, self.cfg.tick)
        for d in range(levels):
            qty = int(size * (1.0 + 0.15 * d))
            book.limit(Side.BID, mid_tick - half - d, qty, Owner.NOISE)
            book.limit(Side.ASK, mid_tick + half + d, qty, Owner.NOISE)

    # ---- one simulation tick -------------------------------------------

    def step(self, book) -> dict:
        cfg = self.cfg
        dt = cfg.dt
        stats = {"limits": 0, "cancels": 0, "markets": 0, "market_volume": 0}
        self._place_limits(book, dt, stats)
        self._cancel(book, dt, stats)
        self._aggress(book, dt, stats)
        return stats

    # Poisson passive placement, anchored on the current touch.
    def _place_limits(self, book, dt, stats) -> None:
        cfg = self.cfg
        n = self.rng.poisson(cfg.lambda_limit * dt)
        if n == 0:
            return
        depths = self.rng.geometric(1.0 / cfg.depth_scale, size=n).clip(1, cfg.max_depth)
        # offset 0 joins the touch, positive rests behind it, -1 improves it.
        offsets = depths - 1
        offsets[self.rng.random(n) < cfg.p_improve] = -1
        sizes = self.rng.poisson(cfg.limit_size_mean, size=n).clip(1, None)
        sides = self.rng.random(n) < 0.5
        for is_bid, d, q in zip(sides, offsets, sizes):
            if is_bid:
                anchor = book.best_bid()
                if anchor < 0:
                    anchor = self._ref - cfg.tick
                px = anchor - int(d) * cfg.tick
                opp = book.best_ask()
                if opp >= 0:
                    px = min(px, opp - cfg.tick)  # passive flow never crosses
                if px > 0:
                    book.limit(Side.BID, int(px), int(q), Owner.NOISE)
                    stats["limits"] += 1
            else:
                anchor = book.best_ask()
                if anchor < 0:
                    anchor = self._ref + cfg.tick
                px = anchor + int(d) * cfg.tick
                opp = book.best_bid()
                if opp >= 0:
                    px = max(px, opp + cfg.tick)
                book.limit(Side.ASK, int(px), int(q), Owner.NOISE)
                stats["limits"] += 1
        if book.best_bid() >= 0 and book.best_ask() >= 0:
            self._ref = int(book.mid())

    # Cancellation hazard accelerates near the touch: lambda_C(d) = a / d**b.
    def _cancel(self, book, dt, stats) -> None:
        cfg = self.cfg
        for side, best, sign in ((Side.BID, book.best_bid(), -1), (Side.ASK, book.best_ask(), 1)):
            if best < 0:
                continue
            for d in range(1, cfg.max_depth + 1):
                px = best + sign * (d - 1) * cfg.tick
                if px <= 0:
                    break
                count = book.count_at(side, px)
                if count == 0:
                    continue
                rate = cfg.cancel_alpha / (d ** cfg.cancel_beta)
                k = self.rng.poisson(rate * count * dt)
                for _ in range(min(int(k), count)):
                    oid = book.tail_id_at(side, px)
                    if oid <= 0:
                        break
                    # Never cancel on the agent's behalf.
                    if book.cancel(oid):
                        stats["cancels"] += 1

    # Hawkes-clustered aggressive flow with power-law sizes.
    def _aggress(self, book, dt, stats) -> None:
        cfg = self.cfg
        self.hawkes_state *= float(np.exp(-cfg.hawkes_beta * dt))
        intensity = cfg.hawkes_mu + self.hawkes_state
        n = self.rng.poisson(intensity * dt)
        if n == 0:
            return
        bid_vol = self._touch_volume(book, Side.BID, book.best_bid(), -1)
        ask_vol = self._touch_volume(book, Side.ASK, book.best_ask(), 1)
        total = bid_vol + ask_vol
        imb = (bid_vol - ask_vol) / total if total > 0 else 0.0
        # Positive imbalance (heavy bid) makes buy-side aggression more likely.
        p_buy = float(np.clip(0.5 + cfg.imbalance_bias * imb, 0.05, 0.95))
        for _ in range(int(n)):
            u = self.rng.random()
            size = int(min(cfg.size_xmin * u ** (-1.0 / cfg.size_tail_index), cfg.size_cap))
            side = Side.BID if self.rng.random() < p_buy else Side.ASK
            filled = book.market(side, max(1, size), Owner.NOISE)
            stats["markets"] += 1
            stats["market_volume"] += int(filled)
            self.hawkes_state += cfg.hawkes_alpha

    @staticmethod
    def _touch_volume(book, side, best, sign, depth: int = 5) -> float:
        if best < 0:
            return 0.0
        return float(sum(book.volume_at(side, best + sign * d) for d in range(depth)))
