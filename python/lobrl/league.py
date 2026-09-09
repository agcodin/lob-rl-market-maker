"""Head-to-head: different policies quoting in the same order book at once.

The single-agent evaluation gives each policy its own private market. A league
puts them in one book, where a fill one policy wins is a fill the others lose --
the only comparison that actually scores them against each other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lobrl.baselines import AvellanedaStoikovPolicy, FixedSpreadPolicy
from lobrl.metrics import summarize
from lobrl.multi_env import MultiAgentMarketMakingEnv, MultiEnvConfig
from lobrl.ppo import ActorCritic, PPOConfig, RunningNorm


class LoadedPolicy:
    """A trained checkpoint, callable on one agent's observation row."""

    def __init__(self, ckpt: str, label: str | None = None, deterministic: bool = True):
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.net = ActorCritic(blob["obs_dim"], blob["act_dim"], PPOConfig())
        self.net.load_state_dict(blob["model"])
        self.net.eval()
        self.norm = RunningNorm(blob["obs_dim"])
        self.norm.load_state_dict(blob["norm"])
        self.deterministic = deterministic
        self.label = label or Path(ckpt).parent.name
        self.trained_n = blob.get("n_agents", 1)

    def reset(self):
        pass

    def __call__(self, obs, view):
        with torch.no_grad():
            a, _, _ = self.net.act(torch.as_tensor(self.norm(obs)[None, :]),
                                   deterministic=self.deterministic)
        return np.clip(a.numpy()[0], -1.0, 1.0)


class _AgentView:
    """The per-agent slice of the shared env that heuristic policies expect."""

    __slots__ = ("cfg", "step_count", "inventory", "_env")

    def __init__(self, env, i):
        self._env = env
        self.cfg = env.cfg
        self.step_count = 0
        self.inventory = 0

    def sync(self, i):
        self.step_count = self._env.step_count
        self.inventory = int(self._env.inventory[i])

    def _volatility(self):
        return self._env._volatility()


def run_league(policies: list, seed: int, steps: int, cfg: MultiEnvConfig | None = None) -> dict:
    """Run one episode with `policies[i]` controlling agent i."""
    n = len(policies)
    cfg = cfg or MultiEnvConfig(n_agents=n, max_steps=steps)
    cfg.n_agents, cfg.max_steps = n, steps
    env = MultiAgentMarketMakingEnv(cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    for p in policies:
        p.reset()
    views = [_AgentView(env, i) for i in range(n)]

    equity = np.zeros((steps, n))
    inv = np.zeros((steps, n))
    offs = np.zeros((steps, n, 2))
    mids, spreads = [], []
    t = 0
    while True:
        actions = np.zeros((n, 2), np.float32)
        for i, p in enumerate(policies):
            views[i].sync(i)
            actions[i] = p(obs[i], views[i])
        obs, rew, term, trunc, info = env.step(actions)
        equity[t] = info["equity"]
        inv[t] = info["inventory"]
        offs[t] = info["offsets"]
        mids.append(info["mid"])
        spreads.append(info["spread"])
        t += 1
        if term or trunc:
            break

    out = []
    for i, p in enumerate(policies):
        fills = [(s, px, q) for (a, s, px, q) in env.fill_log if a == i]
        m = summarize(equity[:t, i], mids, fills, inv[:t, i], steps_per_year=steps * 252)
        m["label"] = getattr(p, "label", type(p).__name__)
        m["mean_half_spread"] = float(offs[:t, i].mean())
        m["volume"] = int(env.volume_traded[i])
        out.append(m)
    return {
        "agents": out,
        "book_spread": float(np.mean(spreads)),
        "mid_path": mids,
        "equity": equity[:t].tolist(),
        "inventory": inv[:t].tolist(),
    }


def build_roster(args) -> list:
    roster = []
    for spec in args.policy:
        if spec.startswith("ckpt:"):
            path, _, label = spec[5:].partition("@")
            roster.append(LoadedPolicy(path, label or None))
        elif spec.startswith("fixed:"):
            w = float(spec[6:])
            p = FixedSpreadPolicy(w)
            p.label = f"fixed {w:g}t"
            roster.append(p)
        elif spec == "as":
            p = AvellanedaStoikovPolicy(gamma=0.05, k=1.5)
            p.label = "Avellaneda-Stoikov"
            roster.append(p)
        else:
            raise SystemExit(f"unknown policy spec: {spec}")
    return roster


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", action="append", required=True,
                    help="ckpt:PATH[@label] | as | fixed:WIDTH (repeatable, one per seat)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed-offset", type=int, default=10_000)
    ap.add_argument("--out", default="runs/league.json")
    args = ap.parse_args()

    roster = build_roster(args)
    n = len(roster)
    runs = []
    for e in range(args.episodes):
        # Rotate seat assignment so no policy keeps a seat-specific advantage.
        order = [(i + e) % n for i in range(n)]
        res = run_league([roster[i] for i in order], args.seed_offset + e, args.steps)
        for seat, i in enumerate(order):
            runs.append({"policy": getattr(roster[i], "label", str(i)), **res["agents"][seat]})

    labels = [getattr(p, "label", str(i)) for i, p in enumerate(roster)]
    keys = ["final_pnl", "sharpe", "max_drawdown", "adverse_selection_10",
            "mean_abs_inventory", "n_fills", "mean_half_spread"]
    table = {}
    for lab in labels:
        rows = [r for r in runs if r["policy"] == lab]
        table[lab] = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        table[lab]["win_rate"] = float(np.mean([r["final_pnl"] > 0 for r in rows]))
        table[lab]["episodes"] = len(rows)

    print(f"\n{args.episodes} episodes, {n} makers sharing one book\n")
    hdr = ["policy"] + keys + ["win_rate"]
    print("".join(f"{h:>22}" if i else f"{h:<24}" for i, h in enumerate(hdr)))
    for lab, r in sorted(table.items(), key=lambda kv: -kv[1]["final_pnl"]):
        print(f"{lab:<24}" + "".join(f"{r[k]:>22.3f}" for k in keys + ["win_rate"]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(table, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
