#!/usr/bin/env python
"""Record the competitive results for the dashboard.

Produces three things in one blob: the competition sweep (spread and profit vs
number of makers), a head-to-head league table, and a tick-by-tick replay of one
episode with several makers quoting into the same book.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lobrl.baselines import AvellanedaStoikovPolicy, FixedSpreadPolicy
from lobrl.league import LoadedPolicy, _AgentView, run_league
from lobrl.metrics import summarize
from lobrl.multi_env import MultiAgentMarketMakingEnv, MultiEnvConfig


def record_replay(policies, labels, seed: int, steps: int, levels: int = 5) -> dict:
    n = len(policies)
    cfg = MultiEnvConfig(n_agents=n, max_steps=steps, levels=levels)
    env = MultiAgentMarketMakingEnv(cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    for p in policies:
        p.reset()
    views = [_AgentView(env, i) for i in range(n)]

    tr = {"mid": [], "spread": []}
    per = [{"quote_bid": [], "quote_ask": [], "inventory": [], "equity": []} for _ in range(n)]
    book = {"bid_px": [], "bid_vol": [], "ask_px": [], "ask_vol": []}
    fills = []
    seen = 0

    while True:
        actions = np.zeros((n, 2), np.float32)
        for i, p in enumerate(policies):
            views[i].sync(i)
            actions[i] = p(obs[i], views[i])
        obs, rew, term, trunc, info = env.step(actions)

        tr["mid"].append(info["mid"])
        tr["spread"].append(int(info["spread"]))
        for i in range(n):
            per[i]["quote_bid"].append(int(info["quote_bid"][i]))
            per[i]["quote_ask"].append(int(info["quote_ask"][i]))
            per[i]["inventory"].append(int(info["inventory"][i]))
            per[i]["equity"].append(round(float(info["equity"][i]), 2))
        book["bid_px"].append(env._bid_px.tolist())
        book["bid_vol"].append(env._bid_vol.tolist())
        book["ask_px"].append(env._ask_px.tolist())
        book["ask_vol"].append(env._ask_vol.tolist())
        for (a, s, px, q) in env.fill_log[seen:]:
            fills.append({"agent": a, "step": s, "price": px, "qty": q})
        seen = len(env.fill_log)
        if term or trunc:
            break

    metrics = []
    for i in range(n):
        mine = [(s, px, q) for (a, s, px, q) in env.fill_log if a == i]
        m = summarize(np.array(per[i]["equity"]), tr["mid"], mine,
                      np.array(per[i]["inventory"]), steps_per_year=steps * 252)
        m["label"] = labels[i]
        metrics.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})

    return {"trace": tr, "agents": per, "book": book, "fills": fills,
            "labels": labels, "metrics": metrics, "n_steps": len(tr["mid"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--replay-seed", type=int, default=20_000)
    ap.add_argument("--out", default="runs/multi_data.json")
    args = ap.parse_args()

    # --- 1. competition sweep: each size at its own trained equilibrium ---
    sweep = []
    for n in args.sizes:
        ck = f"runs/selfplay_n{n}/policy.pt"
        if not Path(ck).exists():
            print(f"[skip] {ck}")
            continue
        pol = LoadedPolicy(ck, label=f"n={n}")
        pnl, half, fills, inv, bs = [], [], [], [], []
        for e in range(args.episodes):
            r = run_league([pol] * n, args.replay_seed + e, args.steps,
                           MultiEnvConfig(n_agents=n, max_steps=args.steps))
            pnl += [a["final_pnl"] for a in r["agents"]]
            half += [a["mean_half_spread"] for a in r["agents"]]
            fills += [a["n_fills"] for a in r["agents"]]
            inv += [a["mean_abs_inventory"] for a in r["agents"]]
            bs.append(r["book_spread"])
        sweep.append({"n_agents": n,
                      "pnl_per_maker": round(float(np.mean(pnl)), 1),
                      "pnl_total": round(float(np.mean(pnl)) * n, 1),
                      "half_spread": round(float(np.mean(half)), 3),
                      "book_spread": round(float(np.mean(bs)), 3),
                      "fills_per_maker": round(float(np.mean(fills)), 1),
                      "abs_inventory": round(float(np.mean(inv)), 2)})
        print("sweep", sweep[-1], flush=True)

    # --- 2. league: four different policies, one book, rotating seats ---
    roster, labels = [], []
    if Path("runs/selfplay_n4/policy.pt").exists():
        roster.append(LoadedPolicy("runs/selfplay_n4/policy.pt", label="Self-play PPO"))
        labels.append("Self-play PPO")
    if Path("runs/ppo/policy.pt").exists():
        roster.append(LoadedPolicy("runs/ppo/policy.pt", label="Solo-trained PPO"))
        labels.append("Solo-trained PPO")
    asp = AvellanedaStoikovPolicy(gamma=0.05, k=1.5); asp.label = "Avellaneda-Stoikov"
    fx = FixedSpreadPolicy(3.0); fx.label = "Fixed 3 ticks"
    roster += [asp, fx]
    labels += ["Avellaneda-Stoikov", "Fixed 3 ticks"]

    n = len(roster)
    keys = ["final_pnl", "sharpe", "max_drawdown", "adverse_selection_10",
            "mean_abs_inventory", "n_fills", "mean_half_spread"]
    runs = []
    for e in range(args.episodes):
        order = [(i + e) % n for i in range(n)]      # rotate seats
        r = run_league([roster[i] for i in order], args.replay_seed + 500 + e, args.steps)
        for seat, i in enumerate(order):
            runs.append({"policy": labels[i], **r["agents"][seat]})
    league = {}
    for lab in labels:
        rows = [x for x in runs if x["policy"] == lab]
        league[lab] = {k: round(float(np.mean([x[k] for x in rows])), 3) for k in keys}
        league[lab]["win_rate"] = round(float(np.mean([x["final_pnl"] > 0 for x in rows])), 3)
    print("league:", json.dumps(league, indent=2), flush=True)

    # --- 3. replay of one four-maker episode ---
    replay = record_replay(roster, labels, args.replay_seed + 500, args.steps)

    hist = {}
    for n_ in args.sizes:
        p = Path(f"runs/selfplay_n{n_}/history.json")
        if p.exists():
            rows = json.loads(p.read_text())
            hist[str(n_)] = [{"steps": r["steps"], "ret": round(r["mean_return"], 1),
                              "inv": round(r["mean_abs_inv"], 2),
                              "off": round(r["mean_offset"], 3)} for r in rows[::2]]

    blob = {"sweep": sweep, "league": league, "league_order": labels,
            "replay": replay, "training": hist, "steps": args.steps,
            "episodes": args.episodes, "replay_seed": args.replay_seed + 500}
    Path(args.out).write_text(json.dumps(blob, separators=(",", ":")))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
