#!/usr/bin/env python
"""Morning report for the overnight arena. Runs unattended; never raises."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def load_history(d: Path) -> list[dict]:
    f = d / "history.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def final_league(arena_dir: Path, episodes: int, steps: int) -> dict:
    """The overnight policy against every earlier policy, in one shared book."""
    from lobrl.arena import ArenaConfig, FrozenPolicy, _Heuristic, _run_episode
    from lobrl.baselines import AvellanedaStoikovPolicy, FixedSpreadPolicy

    entries = []
    best = arena_dir / "best.pt"
    if best.exists():
        entries.append((FrozenPolicy.load(best), "Overnight arena"))
    for path, name in [("runs/selfplay_n4/policy.pt", "Self-play (5 min)"),
                       ("runs/ppo/policy.pt", "Solo-trained")]:
        if Path(path).exists():
            entries.append((FrozenPolicy.load(path), name))
    entries.append((_Heuristic(AvellanedaStoikovPolicy(gamma=0.05, k=1.5), "as"),
                    "Avellaneda-Stoikov"))
    entries.append((_Heuristic(FixedSpreadPolicy(3.0), "f3"), "Fixed 3 ticks"))

    n = len(entries)
    cfg = ArenaConfig(n_agents=n, episode_steps=steps)
    rows = {name: [] for _, name in entries}
    for e in range(episodes):
        order = [(i + e) % n for i in range(n)]          # rotate seats
        res = _run_episode([entries[i][0] for i in order], cfg, 40_000 + e)
        for seat, i in enumerate(order):
            rows[entries[i][1]].append(res[seat])

    keys = ["final_pnl", "sharpe", "max_drawdown", "mean_abs_inventory",
            "adverse_selection_10", "n_fills"]
    out = {}
    for name, rs in rows.items():
        out[name] = {k: float(np.mean([r[k] for r in rs])) for k in keys}
        out[name]["win_rate"] = float(np.mean([r["final_pnl"] > 0 for r in rs]))
        out[name]["episodes"] = len(rs)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="runs/arena")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()
    d = Path(args.dir)

    hist = load_history(d)
    lines = ["# Overnight arena report", ""]
    lines.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}.")
    lines.append("")

    if not hist:
        lines.append("No generations completed. Check `runs/arena/arena.log`.")
        (d / "MORNING_REPORT.md").write_text("\n".join(lines))
        print("\n".join(lines))
        return

    promos = [h for h in hist if h["promoted"]]
    first, last = hist[0], hist[-1]
    lines += [
        "## Run", "",
        f"- Generations: **{len(hist)}**",
        f"- Promotions (candidate beat the incumbent head to head): **{len(promos)}**",
        f"- Transitions trained: **{last['transitions']:,}**",
        f"- Wall clock: **{last['elapsed_hours']:.2f} h**",
        f"- Opponent pool at end: **{last['pool_size']} snapshots**",
        "",
    ]

    # Progress: candidate strength over the night, in windows.
    pnl = np.array([h["candidate"]["pnl"] for h in hist])
    shp = np.array([h["candidate"]["sharpe"] for h in hist])
    inv = np.array([h["candidate"]["abs_inv"] for h in hist])
    w = max(1, len(hist) // 6)
    lines += ["## Progress through the night", "",
              "| window | generations | cand. PnL | Sharpe | avg position |",
              "|---|---|---|---|---|"]
    for i in range(0, len(hist), w):
        sl = slice(i, min(i + w, len(hist)))
        lines.append(f"| {i+1}-{min(i+w, len(hist))} | {sl.stop - sl.start} | "
                     f"{pnl[sl].mean():.0f} | {shp[sl].mean():.1f} | {inv[sl].mean():.1f} |")
    lines.append("")
    if len(hist) >= 10:
        early, late = pnl[: len(pnl) // 4], pnl[-len(pnl) // 4:]
        lines.append(f"First quarter mean PnL {early.mean():.0f} -> last quarter "
                     f"{late.mean():.0f} ({(late.mean()-early.mean()):+.0f}).")
        lines.append("")

    try:
        lg = final_league(d, args.episodes, args.steps)
        lines += [f"## Final head-to-head ({args.episodes} shared-book episodes, seats rotated)", "",
                  "| strategy | PnL | Sharpe | max DD | avg pos | markout | fills | win rate |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, r in sorted(lg.items(), key=lambda kv: -kv[1]["sharpe"]):
            lines.append(f"| {name} | {r['final_pnl']:.0f} | {r['sharpe']:.1f} | "
                         f"{r['max_drawdown']:.0f} | {r['mean_abs_inventory']:.1f} | "
                         f"{r['adverse_selection_10']:.2f} | {r['n_fills']:.0f} | "
                         f"{r['win_rate']*100:.0f}% |")
        lines.append("")
        (d / "final_league.json").write_text(json.dumps(lg, indent=2))
    except Exception as exc:                                  # never lose the report
        lines += ["## Final head-to-head", "", f"Failed: `{exc}`", ""]

    lines += ["## Files", "",
              "- `runs/arena/best.pt` — the strongest policy of the night",
              "- `runs/arena/pool/` — opponent snapshots",
              "- `runs/arena/history.jsonl` — one row per generation",
              "- `runs/arena/status.json` — live progress while running",
              "- `runs/arena/arena.log` — full training log",
              ""]

    (d / "MORNING_REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
