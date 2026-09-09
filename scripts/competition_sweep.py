#!/usr/bin/env python
"""Does competition compress the spread?

Each policy was trained by self-play at its own table size, so every row is that
market's equilibrium rather than one policy dropped into a crowd it never faced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lobrl.league import LoadedPolicy, run_league
from lobrl.multi_env import MultiEnvConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed-offset", type=int, default=20_000)
    ap.add_argument("--out", default="runs/competition.json")
    args = ap.parse_args()

    rows = []
    for n in args.sizes:
        ckpt = f"runs/selfplay_n{n}/policy.pt"
        if not Path(ckpt).exists():
            print(f"[skip] {ckpt} missing")
            continue
        pol = LoadedPolicy(ckpt, label=f"selfplay_n{n}")
        per_agent, spreads, book_spreads, fills, inv = [], [], [], [], []
        for e in range(args.episodes):
            res = run_league([pol] * n, args.seed_offset + e, args.steps,
                             MultiEnvConfig(n_agents=n, max_steps=args.steps))
            per_agent += [a["final_pnl"] for a in res["agents"]]
            spreads += [a["mean_half_spread"] for a in res["agents"]]
            fills += [a["n_fills"] for a in res["agents"]]
            inv += [a["mean_abs_inventory"] for a in res["agents"]]
            book_spreads.append(res["book_spread"])
        rows.append({
            "n_agents": n,
            "pnl_per_maker": float(np.mean(per_agent)),
            "pnl_total": float(np.mean(per_agent)) * n,
            "half_spread_quoted": float(np.mean(spreads)),
            "book_spread": float(np.mean(book_spreads)),
            "fills_per_maker": float(np.mean(fills)),
            "abs_inventory": float(np.mean(inv)),
        })
        r = rows[-1]
        print(f"n={n:2d}  quoted half-spread {r['half_spread_quoted']:5.2f}t  "
              f"book spread {r['book_spread']:5.2f}t  PnL/maker {r['pnl_per_maker']:8.1f}  "
              f"total {r['pnl_total']:8.1f}  fills/maker {r['fills_per_maker']:6.1f}  "
              f"|inv| {r['abs_inventory']:5.1f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
