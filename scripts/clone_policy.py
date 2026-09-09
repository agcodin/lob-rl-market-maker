#!/usr/bin/env python
"""Behaviour-clone a known-good hand-crafted policy into a network checkpoint.

Some behaviours are easy to write down and hard for policy gradient to find. The
fundamental-gap lean is one: a measured +6.5 Sharpe (t=5.1) over the champion,
yet PPO left it undiscovered even with the predicted gap handed to it as an input
feature at full weight. Cloning installs the behaviour directly; RL then refines
it from a good starting point instead of searching for it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lobrl.arena import ArenaConfig, FrozenPolicy, env_config
from lobrl.multi_env import MultiAgentMarketMakingEnv
from lobrl.ppo import ActorCritic, PPOConfig


def collect(base: FrozenPolicy, cfg: ArenaConfig, k: float, episodes: int, seed0: int):
    """Roll out the leaning policy and record (observation, leaning action)."""
    X, Y = [], []
    for e in range(episodes):
        env = MultiAgentMarketMakingEnv(env_config(cfg, cfg.n_agents), seed=seed0 + e)
        obs, _ = env.reset(seed=seed0 + e)
        while True:
            a = base.act_batch(obs, deterministic=True)
            shift = np.clip(k * obs[:, -1], -1.0, 1.0)
            a[:, 0] -= shift
            a[:, 1] += shift
            a = np.clip(a, -1.0, 1.0)
            X.append(obs.copy())
            Y.append(a.copy())
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    return np.concatenate(X), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="/tmp/base33/best.pt")
    ap.add_argument("--out", default="runs/cloned/best.pt")
    ap.add_argument("--k", type=float, default=0.4)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--gap-predictor", default="runs/gap_predictor.pt")
    args = ap.parse_args()

    cfg = ArenaConfig(n_agents=4, episode_steps=1000, flow_features=True,
                      informed_frac=0.35, fundamental_vol=0.35,
                      gap_predictor=args.gap_predictor)
    base = FrozenPolicy.load(args.base)
    print(f"collecting {args.episodes} episodes of the k={args.k} leaning policy ...", flush=True)
    X, Y = collect(base, cfg, args.k, args.episodes, 300_000)
    print(f"  {len(X):,} samples")

    net = ActorCritic(X.shape[1], Y.shape[1], PPOConfig())
    net.load_state_dict(base.net.state_dict(), strict=False)   # keep the critic
    opt = torch.optim.Adam(net.actor.parameters(), 1e-3)
    Xn = torch.tensor(base.norm(X), dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)

    n = len(Xn)
    split = int(n * 0.9)
    for step in range(args.steps):
        idx = torch.randint(0, split, (4096,))
        opt.zero_grad()
        loss = ((net.actor(Xn[idx]) - Yt[idx]) ** 2).mean()
        loss.backward()
        opt.step()
        if (step + 1) % 800 == 0:
            with torch.no_grad():
                v = ((net.actor(Xn[split:]) - Yt[split:]) ** 2).mean().item()
            print(f"  step {step+1:5d}  train {loss.item():.5f}  held-out {v:.5f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "norm": base.norm.state_dict(),
                "obs_dim": X.shape[1], "act_dim": Y.shape[1], "n_agents": 4,
                "cloned_k": args.k}, args.out)
    with torch.no_grad():
        err = float(((net.actor(Xn[split:]) - Yt[split:]) ** 2).mean())
    print(f"saved {args.out}  held-out action MSE {err:.5f}")


if __name__ == "__main__":
    main()
