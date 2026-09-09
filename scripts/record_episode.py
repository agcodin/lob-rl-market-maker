#!/usr/bin/env python
"""Record per-step traces of each policy for the dashboard.

Writes a single JSON blob: a full step-by-step trace of the PPO episode (book
snapshots, quotes, fills, inventory, PnL) plus scalar curves and summary metrics
for every baseline on the same seed, and the aggregate held-out comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lobrl.baselines import AvellanedaStoikovPolicy, FixedSpreadPolicy
from lobrl.env import EnvConfig, MarketMakingEnv
from lobrl.evaluate import PPOPolicy, evaluate
from lobrl.metrics import summarize


def record(policy, seed: int, steps: int, levels: int = 5, full: bool = False) -> dict:
    env = MarketMakingEnv(EnvConfig(max_steps=steps, levels=levels), seed=seed)
    obs, _ = env.reset(seed=seed)
    policy.reset()

    trace = {k: [] for k in ("mid", "best_bid", "best_ask", "quote_bid", "quote_ask",
                             "inventory", "equity", "realized", "cash", "reward",
                             "delta_bid", "delta_ask", "spread")}
    book = {"bid_px": [], "bid_vol": [], "ask_px": [], "ask_vol": []}
    fills = []
    n_fills_seen = 0

    while True:
        action = policy(obs, env)
        obs, r, term, trunc, info = env.step(action)

        trace["mid"].append(info["mid"])
        trace["best_bid"].append(env.book.best_bid())
        trace["best_ask"].append(env.book.best_ask())
        trace["quote_bid"].append(info["quote_bid"])
        trace["quote_ask"].append(info["quote_ask"])
        trace["inventory"].append(info["inventory"])
        trace["equity"].append(round(info["equity"], 2))
        trace["realized"].append(round(info["realized"], 2))
        trace["cash"].append(round(info["cash"], 2))
        trace["reward"].append(round(float(r), 3))
        trace["delta_bid"].append(round(float(info["delta"][0]), 2))
        trace["delta_ask"].append(round(float(info["delta"][1]), 2))
        trace["spread"].append(int(info["spread"]))

        if full:
            book["bid_px"].append(env._bid_px.tolist())
            book["bid_vol"].append(env._bid_vol.tolist())
            book["ask_px"].append(env._ask_px.tolist())
            book["ask_vol"].append(env._ask_vol.tolist())
            for step, price, signed in env.adverse_log[n_fills_seen:]:
                fills.append({"step": step, "price": price, "qty": int(signed)})
            n_fills_seen = len(env.adverse_log)

        if term or trunc:
            break

    metrics = summarize(np.array(trace["equity"]), trace["mid"], env.adverse_log,
                        np.array(trace["inventory"]), steps_per_year=steps * 252)
    out = {"trace": trace, "metrics": metrics, "n_steps": len(trace["mid"])}
    if full:
        out["book"] = book
        out["fills"] = fills
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="runs/ppo/policy.pt")
    ap.add_argument("--seed", type=int, default=10_000, help="held-out episode to replay")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=30, help="episodes for the aggregate table")
    ap.add_argument("--out", default="runs/dashboard_data.json")
    args = ap.parse_args()

    policies = {
        "ppo": PPOPolicy(args.ckpt),
        "avellaneda_stoikov": AvellanedaStoikovPolicy(gamma=0.05, k=1.5),
        "fixed_1t": FixedSpreadPolicy(1.0),
        "fixed_3t": FixedSpreadPolicy(3.0),
        "fixed_6t": FixedSpreadPolicy(6.0),
    }

    print(f"recording seed {args.seed} ...")
    episodes = {
        name: record(pol, args.seed, args.steps, full=(name == "ppo"))
        for name, pol in policies.items()
    }

    print(f"aggregating {args.episodes} held-out seeds ...")
    seeds = [args.seed + i for i in range(args.episodes)]
    aggregate = evaluate(policies, seeds, args.steps)

    blob = {
        "seed": args.seed,
        "steps": args.steps,
        "levels": 5,
        "episodes": episodes,
        "aggregate": aggregate,
        "config": {
            "quote_size": EnvConfig().quote_size,
            "max_inventory": EnvConfig().max_inventory,
            "inventory_penalty": 5e-3,
            "min_offset": EnvConfig().min_offset,
            "max_offset": EnvConfig().max_offset,
        },
        "training": json.loads(Path("runs/ppo/history.json").read_text())
        if Path("runs/ppo/history.json").exists() else [],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(blob, separators=(",", ":")))
    kb = Path(args.out).stat().st_size / 1024
    print(f"wrote {args.out} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
