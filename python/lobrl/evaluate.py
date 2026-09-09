"""Out-of-sample comparison of the PPO agent against heuristic baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lobrl.baselines import AvellanedaStoikovPolicy, FixedSpreadPolicy
from lobrl.env import EnvConfig, MarketMakingEnv
from lobrl.metrics import summarize
from lobrl.ppo import ActorCritic, PPOConfig, RunningNorm


class PPOPolicy:
    def __init__(self, ckpt: str, deterministic: bool = True):
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.net = ActorCritic(blob["obs_dim"], blob["act_dim"], PPOConfig())
        self.net.load_state_dict(blob["model"])
        self.net.eval()
        self.norm = RunningNorm(blob["obs_dim"])
        self.norm.load_state_dict(blob["norm"])
        self.deterministic = deterministic

    def reset(self):
        pass

    def __call__(self, obs, env):
        with torch.no_grad():
            a, _, _ = self.net.act(
                torch.as_tensor(self.norm(obs)[None, :]), deterministic=self.deterministic
            )
        return np.clip(a.numpy()[0], -1.0, 1.0)


def run_episode(policy, seed: int, episode_steps: int) -> dict:
    env = MarketMakingEnv(EnvConfig(max_steps=episode_steps), seed=seed)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    equity, mids, inv, rewards = [], [], [], []
    while True:
        action = policy(obs, env)
        obs, r, term, trunc, info = env.step(action)
        equity.append(info["equity"])
        mids.append(info["mid"])
        inv.append(info["inventory"])
        rewards.append(r)
        if term or trunc:
            break
    res = summarize(np.array(equity), mids, env.adverse_log, np.array(inv),
                    steps_per_year=episode_steps * 252)
    res["reward_sum"] = float(np.sum(rewards))
    res["turnover"] = env.volume_traded
    return res


def evaluate(policies: dict, seeds, episode_steps: int) -> dict:
    out = {}
    for name, pol in policies.items():
        runs = [run_episode(pol, s, episode_steps) for s in seeds]
        agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        agg["pnl_std"] = float(np.std([r["final_pnl"] for r in runs]))
        agg["win_rate"] = float(np.mean([r["final_pnl"] > 0 for r in runs]))
        out[name] = agg
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="runs/ppo/policy.pt")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--episode-steps", type=int, default=1000)
    p.add_argument("--seed-offset", type=int, default=10_000, help="held-out seed block")
    p.add_argument("--out", default="runs/eval.json")
    args = p.parse_args()

    policies = {
        "fixed_1t": FixedSpreadPolicy(1.0),
        "fixed_3t": FixedSpreadPolicy(3.0),
        "fixed_6t": FixedSpreadPolicy(6.0),
        "avellaneda_stoikov": AvellanedaStoikovPolicy(gamma=0.05, k=1.5),
    }
    if Path(args.ckpt).exists():
        policies["ppo"] = PPOPolicy(args.ckpt)
    else:
        print(f"[warn] {args.ckpt} not found; comparing baselines only")

    seeds = [args.seed_offset + i for i in range(args.episodes)]
    res = evaluate(policies, seeds, args.episode_steps)

    cols = ["final_pnl", "sharpe", "max_drawdown", "adverse_selection_10",
            "mean_abs_inventory", "n_fills", "win_rate"]
    print(f"{'policy':<20}" + "".join(f"{c:>22}" for c in cols))
    for name, r in res.items():
        print(f"{name:<20}" + "".join(f"{r[c]:>22.3f}" for c in cols))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
