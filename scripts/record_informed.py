#!/usr/bin/env python
"""Record the informed-market results for the dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lobrl.arena import ArenaConfig, FrozenPolicy, _Heuristic, _run_episode
from lobrl.baselines import AvellanedaStoikovPolicy, FixedSpreadPolicy


def paired(cand, ref, cfg, seeds, n=4):
    a, b = [], []
    for e, s in enumerate(seeds):
        pol = [(cand if (i + e) % 2 == 0 else ref) for i in range(n)]
        x = _run_episode(pol, cfg, s)
        a.append(np.mean([x[i]["sharpe"] for i in range(n) if (i + e) % 2 == 0]))
        b.append(np.mean([x[i]["sharpe"] for i in range(n) if (i + e) % 2 == 1]))
    d = np.array(a) - np.array(b)
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"cand": float(np.mean(a)), "ref": float(np.mean(b)),
            "diff": float(d.mean()), "stderr": se, "t": float(d.mean() / se),
            "episodes": len(seeds)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--out", default="runs/informed_data.json")
    args = ap.parse_args()

    cfg = ArenaConfig(n_agents=4, episode_steps=1000, flow_features=True,
                      informed_frac=0.35, fundamental_vol=0.35,
                      gap_predictor="runs/gap_predictor.pt")
    plain = ArenaConfig(n_agents=4, episode_steps=1000, flow_features=True,
                        gap_predictor="runs/gap_predictor.pt")

    champ = FrozenPolicy.load("runs/champion_informed.pt")
    entries = [(champ, "Informed champion")]
    base = Path("/tmp/base33/best.pt")
    if base.exists():
        entries.append((FrozenPolicy.load(str(base)), "Uninformed champion"))
    entries.append((_Heuristic(AvellanedaStoikovPolicy(gamma=0.05, k=1.5), "as"),
                    "Avellaneda-Stoikov"))
    entries.append((_Heuristic(FixedSpreadPolicy(3.0), "f3"), "Fixed 3 ticks"))

    keys = ["final_pnl", "sharpe", "max_drawdown", "mean_abs_inventory",
            "adverse_selection_10", "n_fills"]
    standings = {}
    for label_cfg, tag in ((cfg, "informed"), (plain, "uninformed")):
        rows = {nm: [] for _, nm in entries}
        n = len(entries)
        for e in range(args.episodes):
            order = [(i + e) % n for i in range(n)]
            r = _run_episode([entries[i][0] for i in order], label_cfg, 400_000 + e)
            for seat, i in enumerate(order):
                rows[entries[i][1]].append(r[seat])
        standings[tag] = {}
        for nm, rs in rows.items():
            m = {k: round(float(np.mean([x[k] for x in rs])), 3) for k in keys}
            m["win_rate"] = round(float(np.mean([x["final_pnl"] > 0 for x in rs])), 3)
            standings[tag][nm] = m
        print(tag, json.dumps(standings[tag], indent=1)[:180], "...", flush=True)

    blob = {
        "standings": standings,
        "order": [nm for _, nm in entries],
        "episodes": args.episodes,
        # Measurements recorded verbatim from the session's own runs.
        "chain": [
            {"step": "Uninformed champion, dropped into the informed market",
             "sharpe_note": "Sharpe 44.6 -> 26.7 when informed traders are switched on"},
            {"step": "+ learnable gap lean", "diff": 6.17, "t": 5.33, "episodes": 64},
            {"step": "+ RL-tuned lean (to -0.89/+0.76)", "diff": 2.22, "t": 2.18, "episodes": 160},
        ],
        "probes": [
            {"name": "Oracle: knows the true fundamental", "diff": 24.42, "t": 10.20,
             "verdict": "huge headroom exists"},
            {"name": "Lean on the estimate (what is learnable)", "diff": 8.17, "t": 4.50,
             "verdict": "most of it is reachable"},
            {"name": "Hold a position sized by the estimate", "diff": -4.30, "t": -4.72,
             "verdict": "probe was invalid: assumed the position was free to hold"},
        ],
        "progression": [
            {"name": "Gap lean wired into the action head", "diff": 6.17, "t": 5.33,
             "episodes": 64, "source": "architecture", "kept": True},
            {"name": "RL tunes the lean vector", "diff": 2.22, "t": 2.18,
             "episodes": 160, "source": "reinforcement learning", "kept": True},
            {"name": "Estimator trained on-policy", "diff": 3.88, "t": 2.68,
             "episodes": 96, "source": "training distribution", "kept": True},
            {"name": "Lean recalibrated after that change", "diff": 2.39, "t": 2.89,
             "episodes": 128, "source": "calibration", "kept": True},
            {"name": "Estimator given history", "diff": 3.55, "t": 2.88,
             "episodes": 112, "source": "estimator design", "kept": True},
            {"name": "Lean rescaled 1.3x (promoted, then reverted)", "diff": -1.01, "t": -1.19,
             "episodes": 128, "source": "isolated afterwards and dropped", "kept": False},
        ],
        "refuted": [
            {"idea": "Variable quote size", "why": "hand-crafted upper bound not significant"},
            {"idea": "Tape features, uninformed market",
             "why": "less predictive than the imbalance already observed"},
            {"idea": "Auxiliary head on forward returns", "why": "prediction ceiling R2 0.024"},
            {"idea": "Behaviour cloning the lean", "why": "action MSE was 28% of the signal"},
            {"idea": "Multi-timescale tape", "why": "corr 0.476 -> 0.493, worth ~+0.5"},
            {"idea": "Inventory target on the estimate", "why": "-4.30 Sharpe (t=-4.72)"},
            {"idea": "Seed variance from the champion", "why": "all three seeds regressed"},
        ],
        "estimator": {"linear_ceiling": 0.29, "rl_managed": 0.15, "supervised": 0.476,
                      "with_history": 0.543},
    }
    Path(args.out).write_text(json.dumps(blob, separators=(",", ":")))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
