"""Financial evaluation metrics for a market-making episode."""

from __future__ import annotations

import numpy as np


def sharpe(pnl_increments: np.ndarray, steps_per_year: float = 252 * 6.5 * 3600) -> float:
    """Annualized Sharpe of per-step PnL increments (zero risk-free rate)."""
    x = np.asarray(pnl_increments, dtype=np.float64)
    if x.size < 2:
        return 0.0
    sd = x.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(x.mean() / sd * np.sqrt(steps_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough decline of the equity curve, in currency units."""
    e = np.asarray(equity, dtype=np.float64)
    if e.size == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    return float(np.max(peak - e))


def adverse_selection(fills, mids, horizon: int = 10) -> float:
    """Mean markout against the agent `horizon` steps after each fill.

    Positive = the agent was adversely selected (price moved against the fill).
    `fills` is a list of (step, price, signed_qty); `mids` is the mid series
    indexed by step.
    """
    mids = np.asarray(mids, dtype=np.float64)
    if len(fills) == 0 or mids.size == 0:
        return 0.0
    out = []
    for step, price, signed in fills:
        j = min(step + horizon, mids.size - 1)
        # A buy (signed > 0) is adversely selected when the mid falls afterwards.
        out.append(-np.sign(signed) * (mids[j] - price))
    return float(np.mean(out))


def summarize(equity: np.ndarray, mids, fills, inventory: np.ndarray,
              steps_per_year: float = 252 * 6.5 * 3600) -> dict:
    equity = np.asarray(equity, dtype=np.float64)
    incr = np.diff(equity, prepend=equity[0] if equity.size else 0.0)
    return {
        "final_pnl": float(equity[-1]) if equity.size else 0.0,
        "sharpe": sharpe(incr[1:], steps_per_year),
        "max_drawdown": max_drawdown(equity),
        "adverse_selection_10": adverse_selection(fills, mids, 10),
        "mean_abs_inventory": float(np.mean(np.abs(inventory))) if len(inventory) else 0.0,
        "max_abs_inventory": float(np.max(np.abs(inventory))) if len(inventory) else 0.0,
        "n_fills": len(fills),
    }
