#!/usr/bin/env python
"""Record the overnight failure and the fix that followed, for the dashboard.

Three things: the overnight trajectory (what went wrong), a paired evaluation of
each intervention against the previous champion (what fixed it), and the current
standings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lobrl.arena import ArenaConfig, FrozenPolicy, _Heuristic, _run_episode
from lobrl.baselines import AvellanedaStoikovPolicy, FixedSpreadPolicy


def paired(cand: FrozenPolicy, ref: FrozenPolicy, seeds, n=4, steps=1000) -> dict:
    """Both policies quote the same book on the same seed, seats alternating.

    Pairing cancels the shared market noise, which is what makes a few Sharpe
    points distinguishable at all.
    """
    cfg = ArenaConfig(n_agents=n, episode_steps=steps)
    cs, rs, inv, pnl = [], [], [], []
    for e, s in enumerate(seeds):
        pol = [(cand if (i + e) % 2 == 0 else ref) for i in range(n)]
        r = _run_episode(pol, cfg, s)
        c = [r[i] for i in range(n) if (i + e) % 2 == 0]
        h = [r[i] for i in range(n) if (i + e) % 2 == 1]
        cs.append(np.mean([x["sharpe"] for x in c]))
        rs.append(np.mean([x["sharpe"] for x in h]))
        inv.append(np.mean([x["mean_abs_inventory"] for x in c]))
        pnl.append(np.mean([x["final_pnl"] for x in c]))
    d = np.array(cs) - np.array(rs)
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"sharpe": float(np.mean(cs)), "ref_sharpe": float(np.mean(rs)),
            "delta": float(d.mean()), "stderr": se, "t": float(d.mean() / se) if se else 0.0,
            "abs_inventory": float(np.mean(inv)), "pnl": float(np.mean(pnl)),
            "episodes": len(seeds)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--league-episodes", type=int, default=32)
    ap.add_argument("--out", default="runs/arena_data.json")
    args = ap.parse_args()

    blob = {}

    # --- 1. the overnight trajectory ------------------------------------
    hist_p = Path("runs/arena/history.jsonl")
    if hist_p.exists():
        h = [json.loads(l) for l in hist_p.read_text().splitlines() if l.strip()]
        w = max(1, len(h) // 12)
        blob["overnight"] = {
            "generations": len(h),
            "transitions": h[-1]["transitions"],
            "hours": h[-1]["elapsed_hours"],
            "promotions": sum(1 for x in h if x["promoted"]),
            "windows": [
                {"gen": i + w // 2,
                 "sharpe": round(float(np.mean([x["candidate"]["sharpe"] for x in h[i:i + w]])), 2),
                 "inv": round(float(np.mean([x["candidate"]["abs_inv"] for x in h[i:i + w]])), 1),
                 "pnl": round(float(np.mean([x["candidate"]["pnl"] for x in h[i:i + w]])), 0)}
                for i in range(0, len(h), w)
            ],
            "corr_inv_pnl": round(float(np.corrcoef(
                [x["candidate"]["abs_inv"] for x in h], [x["candidate"]["pnl"] for x in h])[0, 1]), 3),
            "corr_inv_sharpe": round(float(np.corrcoef(
                [x["candidate"]["abs_inv"] for x in h], [x["candidate"]["sharpe"] for x in h])[0, 1]), 3),
        }
        print("overnight:", {k: v for k, v in blob["overnight"].items() if k != "windows"})

    # --- 2. the interventions, paired against the previous champion -----
    ref = FrozenPolicy.load("runs/selfplay_n4/policy.pt")
    arms = [("A", "inventory penalty 5e-3 to 2e-2", "runs/arm_A/best.pt"),
            ("B", "no heuristics in opponent pool", "runs/arm_B/best.pt"),
            ("C", "learner takes all four seats", "runs/arm_C/best.pt"),
            ("A+B", "both together", "runs/arm_AB/best.pt")]
    blob["arms"] = []
    for key, desc, path in arms:
        if not Path(path).exists():
            continue
        r = paired(FrozenPolicy.load(path), ref, [80_000 + i for i in range(args.episodes)])
        blob["arms"].append({"arm": key, "change": desc, **{k: round(v, 3) for k, v in r.items()}})
        a = blob["arms"][-1]
        print(f"arm {key:<4} delta {a['delta']:+6.2f} +/- {a['stderr']:.2f}  t {a['t']:+.2f}  "
              f"|inv| {a['abs_inventory']:.1f}", flush=True)

    # --- 3. current standings -------------------------------------------
    ent = []
    for path, name in [("runs/champion.pt", "Arena champion"),
                       ("runs/selfplay_n4/policy.pt", "Previous champion"),
                       ("runs/ppo/policy.pt", "Solo-trained")]:
        if Path(path).exists():
            ent.append((FrozenPolicy.load(path), name))
    ent.append((_Heuristic(AvellanedaStoikovPolicy(gamma=0.05, k=1.5), "as"), "Avellaneda-Stoikov"))
    ent.append((_Heuristic(FixedSpreadPolicy(3.0), "f3"), "Fixed 3 ticks"))

    n = len(ent)
    cfg = ArenaConfig(n_agents=n, episode_steps=1000)
    rows = {nm: [] for _, nm in ent}
    for e in range(args.league_episodes):
        order = [(i + e) % n for i in range(n)]
        r = _run_episode([ent[i][0] for i in order], cfg, 95_000 + e)
        for seat, i in enumerate(order):
            rows[ent[i][1]].append(r[seat])
    keys = ["final_pnl", "sharpe", "max_drawdown", "mean_abs_inventory",
            "adverse_selection_10", "n_fills"]
    blob["standings"] = {}
    for nm, rs in rows.items():
        m = {k: round(float(np.mean([x[k] for x in rs])), 3) for k in keys}
        m["win_rate"] = round(float(np.mean([x["final_pnl"] > 0 for x in rs])), 3)
        blob["standings"][nm] = m
    blob["standings_order"] = [nm for _, nm in ent]
    blob["episodes"] = args.league_episodes
    print("standings:", json.dumps(blob["standings"], indent=2)[:200], "...")

    Path(args.out).write_text(json.dumps(blob, separators=(",", ":")))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
