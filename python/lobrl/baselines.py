"""Heuristic quoting policies used as benchmarks for the learned agent."""

from __future__ import annotations

import numpy as np

from lobrl.env import EnvConfig, MarketMakingEnv


def _to_action(delta_bid: float, delta_ask: float, cfg: EnvConfig) -> np.ndarray:
    """Invert the env's affine action map so policies can think in ticks."""
    lo, hi = cfg.min_offset, cfg.max_offset
    d = np.array([delta_bid, delta_ask], dtype=np.float64)
    return np.clip(2.0 * (d - lo) / (hi - lo) - 1.0, -1.0, 1.0).astype(np.float32)


class FixedSpreadPolicy:
    """Symmetric constant half-spread, no inventory management."""

    def __init__(self, half_spread: float = 3.0):
        self.half_spread = half_spread

    def reset(self):
        pass

    def __call__(self, obs, env: MarketMakingEnv) -> np.ndarray:
        return _to_action(self.half_spread, self.half_spread, env.cfg)


class AvellanedaStoikovPolicy:
    """Closed-form A-S quotes.

    Reservation price  r = s - q * gamma * sigma^2 * (T - t)
    Optimal spread     psi = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)

    The env's action is a pair of half-spreads around the mid, so the inventory
    skew shows up as an asymmetry between the two offsets.
    """

    def __init__(self, gamma: float = 0.05, k: float = 1.5, horizon: int | None = None):
        self.gamma = gamma
        self.k = k
        self.horizon = horizon

    def reset(self):
        pass

    def __call__(self, obs, env: MarketMakingEnv) -> np.ndarray:
        cfg = env.cfg
        T = self.horizon or cfg.max_steps
        t_left = max(0.0, (T - env.step_count) / T)
        sigma = max(env._volatility(), 1e-3)
        q = float(env.inventory) / cfg.quote_size  # inventory in lots

        skew = q * self.gamma * sigma**2 * t_left
        psi = self.gamma * sigma**2 * t_left + (2.0 / self.gamma) * np.log1p(self.gamma / self.k)
        half = 0.5 * psi
        # r = mid - skew  =>  bid = r - half, ask = r + half
        delta_bid = half + skew
        delta_ask = half - skew
        return _to_action(delta_bid, delta_ask, cfg)


class RandomPolicy:
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def reset(self):
        pass

    def __call__(self, obs, env):
        return self.rng.uniform(-1, 1, size=2).astype(np.float32)
