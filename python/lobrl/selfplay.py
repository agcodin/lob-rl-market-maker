"""Shared-policy self-play PPO: every maker in the book runs the same network.

Because all competitors are copies of the policy being trained, the opposition
gets better exactly as fast as the policy does -- the arms race a fixed
statistical opponent cannot provide. The agents still diverge within an episode:
they hold different inventory and sit in different queue positions, so the same
network produces different quotes for each of them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lobrl.multi_env import MultiAgentMarketMakingEnv, MultiEnvConfig
from lobrl.ppo import PPOConfig, PPOTrainer, RunningNorm


class MultiRolloutBuffer:
    """(T, n_agents) rollout streams, flattened to one batch at update time."""

    def __init__(self, size: int, n_agents: int, obs_dim: int, act_dim: int):
        self.T, self.n = size, n_agents
        self.obs = np.zeros((size, n_agents, obs_dim), np.float32)
        self.act = np.zeros((size, n_agents, act_dim), np.float32)
        self.logp = np.zeros((size, n_agents), np.float32)
        self.rew = np.zeros((size, n_agents), np.float32)
        self.val = np.zeros((size, n_agents), np.float32)
        self.done = np.zeros((size, n_agents), np.float32)
        self.t = 0
        self.ptr = 0

    def add(self, obs, act, logp, rew, val, done):
        i = self.t
        self.obs[i], self.act[i], self.logp[i] = obs, act, logp
        self.rew[i], self.val[i], self.done[i] = rew, val, done
        self.t += 1

    def full(self) -> bool:
        return self.t >= self.T

    def reset(self):
        self.t = 0
        self.ptr = 0

    def compute_gae(self, last_val, gamma: float, lam: float):
        """GAE down each agent's own stream, then flatten for the PPO update.

        Also rewrites `obs`/`act`/`logp` as flat (T*n, ...) views and sets `ptr`,
        which is the interface `PPOTrainer.update` consumes.
        """
        T, n = self.t, self.n
        adv = np.zeros((T, n), np.float32)
        gae = np.zeros(n, np.float32)
        last_val = np.asarray(last_val, np.float32).reshape(n)
        for t in reversed(range(T)):
            next_val = last_val if t == T - 1 else self.val[t + 1]
            nonterminal = 1.0 - self.done[t]
            delta = self.rew[t] + gamma * next_val * nonterminal - self.val[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
        ret = adv + self.val[:T]

        self.obs = self.obs[:T].reshape(T * n, -1)
        self.act = self.act[:T].reshape(T * n, -1)
        self.logp = self.logp[:T].reshape(T * n)
        self.ptr = T * n
        return adv.reshape(T * n), ret.reshape(T * n)


def train(n_agents: int = 4, total_steps: int = 400_000, batch: int = 2048,
          episode_steps: int = 1000, seed: int = 0, out: str = "runs/selfplay",
          device: str = "cpu", inventory_penalty: float = 5e-3,
          init_from: str | None = None) -> Path:
    """Train a shared policy across `n_agents` seats in one book.

    `total_steps` counts environment steps; each one yields `n_agents`
    transitions. `batch` is transitions per PPO update, held constant as the
    table size changes -- otherwise a bigger table would silently also mean a
    bigger (and better-conditioned) gradient batch, and a sweep over n_agents
    would measure batch size instead of competition.
    """
    rollout = max(1, batch // n_agents)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = MultiEnvConfig(n_agents=n_agents, max_steps=episode_steps,
                         inventory_penalty=inventory_penalty)
    print(f"n_agents={n_agents}  rollout={rollout} env steps/update  "
          f"batch={rollout * n_agents} transitions/update")
    env = MultiAgentMarketMakingEnv(cfg, seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    trainer = PPOTrainer(obs_dim, act_dim, PPOConfig(), device=device)
    norm = RunningNorm(obs_dim)
    if init_from:
        blob = torch.load(init_from, map_location=device, weights_only=False)
        trainer.net.load_state_dict(blob["model"])
        norm.load_state_dict(blob["norm"])
        print(f"warm-started from {init_from}")

    obs, _ = env.reset(seed=seed)
    history, episodes = [], []
    ep_ret = np.zeros(n_agents)
    t0 = time.time()
    steps = 0
    update = 0

    while steps < total_steps:
        buf = MultiRolloutBuffer(rollout, n_agents, obs_dim, act_dim)
        while not buf.full():
            norm.update(obs)
            nobs = norm(obs)
            with torch.no_grad():
                a, logp, v = trainer.net.act(torch.as_tensor(nobs))
            action = a.numpy()
            next_obs, rew, term, trunc, info = env.step(np.clip(action, -1, 1))
            # A time limit is not a terminal state, but cutting the trace here
            # keeps the next episode's returns from leaking backwards.
            buf.add(nobs, action, logp.numpy(), rew, v.numpy(),
                    np.full(n_agents, 1.0 if trunc else 0.0, np.float32))
            ep_ret += rew
            steps += 1
            obs = next_obs
            if trunc:
                episodes.append({
                    "ret": ep_ret.copy(), "equity": info["equity"].copy(),
                    "inv": np.abs(info["inventory"]).copy(), "trades": info["trades"].copy(),
                    "offsets": info["offsets"].mean(axis=0),
                })
                ep_ret[:] = 0.0
                obs, _ = env.reset()

        with torch.no_grad():
            last_val = trainer.net.critic(torch.as_tensor(norm(obs))).squeeze(-1).numpy()
        stats = trainer.update(buf, last_val)
        update += 1

        recent = episodes[-5:]
        row = {
            "update": update, "steps": steps, "n_agents": n_agents,
            "mean_return": float(np.mean([e["ret"].mean() for e in recent])) if recent else 0.0,
            "mean_equity": float(np.mean([e["equity"].mean() for e in recent])) if recent else 0.0,
            "mean_abs_inv": float(np.mean([e["inv"].mean() for e in recent])) if recent else 0.0,
            "mean_trades": float(np.mean([e["trades"].mean() for e in recent])) if recent else 0.0,
            "mean_offset": float(np.mean([e["offsets"].mean() for e in recent])) if recent else 0.0,
            "eps": steps / (time.time() - t0),
            **{k: float(v) for k, v in stats.items()},
        }
        history.append(row)
        if update % 5 == 0 or update == 1:
            print(f"upd {update:4d} | env steps {steps:7d} | ret {row['mean_return']:9.1f} "
                  f"| equity {row['mean_equity']:8.1f} | |inv| {row['mean_abs_inv']:6.1f} "
                  f"| fills {row['mean_trades']:6.1f} | offset {row['mean_offset']:5.2f}t "
                  f"| ent {row['entropy']:.2f} | {row['eps']:.0f} env-step/s", flush=True)

    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": trainer.net.state_dict(), "norm": norm.state_dict(),
                "obs_dim": obs_dim, "act_dim": act_dim, "n_agents": n_agents},
               outdir / "policy.pt")
    (outdir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved -> {outdir/'policy.pt'}")
    return outdir / "policy.pt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--total-steps", type=int, default=400_000)
    p.add_argument("--batch", type=int, default=2048,
                   help="transitions per PPO update, held constant across table sizes")
    p.add_argument("--episode-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--inventory-penalty", type=float, default=5e-3)
    p.add_argument("--init-from", default=None, help="warm-start from a checkpoint")
    args = p.parse_args()
    out = args.out or f"runs/selfplay_n{args.n_agents}"
    train(args.n_agents, args.total_steps, args.batch, args.episode_steps,
          args.seed, out, args.device, args.inventory_penalty, args.init_from)


if __name__ == "__main__":
    main()
