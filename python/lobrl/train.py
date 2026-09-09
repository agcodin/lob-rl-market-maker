"""Train the PPO market maker."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lobrl.env import EnvConfig, MarketMakingEnv
from lobrl.ppo import PPOConfig, PPOTrainer, RolloutBuffer, RunningNorm


def train(total_steps: int = 200_000, rollout: int = 2048, episode_steps: int = 1000,
          seed: int = 0, out: str = "runs/ppo", device: str = "cpu",
          log_every: int = 1, inventory_penalty: float | None = None) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_cfg = EnvConfig(max_steps=episode_steps)
    if inventory_penalty is not None:
        env_cfg.inventory_penalty = inventory_penalty
    env = MarketMakingEnv(env_cfg, seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    trainer = PPOTrainer(obs_dim, act_dim, PPOConfig(), device=device)
    norm = RunningNorm(obs_dim)
    buf = RolloutBuffer(rollout, obs_dim, act_dim)

    obs, _ = env.reset(seed=seed)
    ep_ret, ep_len, ep_returns, history = 0.0, 0, [], []
    t0 = time.time()
    steps = 0
    update = 0

    while steps < total_steps:
        while not buf.full():
            norm.update(obs[None, :])
            nobs = norm(obs)
            with torch.no_grad():
                a, logp, v = trainer.net.act(torch.as_tensor(nobs[None, :]))
            action = a.numpy()[0]
            next_obs, rew, term, trunc, info = env.step(np.clip(action, -1, 1))
            buf.add(nobs, action, logp.item(), rew, v.item(), float(term))
            ep_ret += rew
            ep_len += 1
            steps += 1
            obs = next_obs
            if term or trunc:
                ep_returns.append((ep_ret, ep_len, info["equity"], info["inventory"], info["trades"]))
                ep_ret, ep_len = 0.0, 0
                obs, _ = env.reset()

        with torch.no_grad():
            last_val = trainer.net.critic(torch.as_tensor(norm(obs)[None, :])).item()
        stats = trainer.update(buf, last_val)
        update += 1

        recent = ep_returns[-10:]
        row = {
            "update": update,
            "steps": steps,
            "mean_return": float(np.mean([r[0] for r in recent])) if recent else 0.0,
            "mean_equity": float(np.mean([r[2] for r in recent])) if recent else 0.0,
            "mean_abs_inv": float(np.mean([abs(r[3]) for r in recent])) if recent else 0.0,
            "mean_trades": float(np.mean([r[4] for r in recent])) if recent else 0.0,
            "sps": steps / (time.time() - t0),
            **{k: float(v) for k, v in stats.items()},
        }
        history.append(row)
        if update % log_every == 0:
            print(
                f"upd {update:4d} | steps {steps:7d} | ret {row['mean_return']:10.1f} "
                f"| equity {row['mean_equity']:9.1f} | |inv| {row['mean_abs_inv']:6.1f} "
                f"| trades {row['mean_trades']:6.1f} | ent {row['entropy']:.3f} "
                f"| {row['sps']:.0f} step/s",
                flush=True,
            )

    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": trainer.net.state_dict(), "norm": norm.state_dict(),
         "obs_dim": obs_dim, "act_dim": act_dim},
        outdir / "policy.pt",
    )
    (outdir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved -> {outdir/'policy.pt'}")
    return outdir / "policy.pt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--rollout", type=int, default=2048)
    p.add_argument("--episode-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/ppo")
    p.add_argument("--device", default="cpu")
    p.add_argument("--inventory-penalty", type=float, default=None,
                   help="override phi in the training reward")
    args = p.parse_args()
    train(args.total_steps, args.rollout, args.episode_steps, args.seed, args.out,
          args.device, inventory_penalty=args.inventory_penalty)


if __name__ == "__main__":
    main()
