"""PPO actor-critic for continuous two-dimensional quote offsets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    hidden: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 10
    minibatch: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 5e-3
    max_grad_norm: float = 0.5
    init_log_std: float = -0.5
    target_kl: float | None = 0.03
    # Auxiliary prediction task. The tape genuinely predicts the forward mid
    # move, but the reward gradient toward using it is tiny -- the benefit is
    # indirect (predict price -> skew quotes -> avoid an adverse fill), so it is
    # swamped by return variance and the input weights stay near zero. A
    # supervised head on the actor trunk forces the representation to carry the
    # signal, and the policy layer can then use it for free.
    aux_coef: float = 0.25
    aux_horizon: int = 10


class RunningNorm:
    """Welford normalizer for observations; frozen at evaluation time."""

    def __init__(self, dim: int, eps: float = 1e-4):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.atleast_2d(x).astype(np.float64)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + delta**2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


def _mlp(inp: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out),
    )


class ActorCritic(nn.Module):
    """Diagonal-Gaussian policy with a state-independent log-std, plus a value head."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig):
        super().__init__()
        self.actor = _mlp(obs_dim, cfg.hidden, act_dim)
        self.critic = _mlp(obs_dim, cfg.hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), cfg.init_log_std))
        # Reads the actor trunk and predicts the forward mid move.
        self.aux = nn.Linear(cfg.hidden, 1)
        nn.init.orthogonal_(self.aux.weight, gain=0.1)
        nn.init.zeros_(self.aux.bias)
        # Small final-layer gain keeps the initial policy near the action-space centre.
        for head in (self.actor, self.critic):
            nn.init.orthogonal_(head[-1].weight, gain=0.01)
            nn.init.zeros_(head[-1].bias)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        """Actor trunk output, shared by the policy head and the aux head."""
        h = obs
        for layer in self.actor[:-1]:
            h = layer(h)
        return h

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mu = self.actor[-1](self.features(obs))
        return torch.distributions.Normal(mu, self.log_std.exp())

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        return self.aux(self.features(obs)).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a, d.log_prob(a).sum(-1), self.critic(obs).squeeze(-1)

    def evaluate(self, obs: torch.Tensor, act: torch.Tensor):
        feat = self.features(obs)
        d = torch.distributions.Normal(self.actor[-1](feat), self.log_std.exp())
        return (d.log_prob(act).sum(-1), d.entropy().sum(-1),
                self.critic(obs).squeeze(-1), self.aux(feat).squeeze(-1))


class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, act_dim: int):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, act_dim), np.float32)
        self.logp = np.zeros(size, np.float32)
        self.rew = np.zeros(size, np.float32)
        self.val = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.ptr = 0
        self.size = size

    def add(self, obs, act, logp, rew, val, done):
        i = self.ptr
        self.obs[i], self.act[i], self.logp[i] = obs, act, logp
        self.rew[i], self.val[i], self.done[i] = rew, val, done
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.size

    def reset(self):
        self.ptr = 0

    def compute_gae(self, last_val: float, gamma: float, lam: float):
        n = self.ptr
        adv = np.zeros(n, np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_val = last_val if t == n - 1 else self.val[t + 1]
            nonterminal = 1.0 - self.done[t]
            delta = self.rew[t] + gamma * next_val * nonterminal - self.val[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
        return adv, adv + self.val[:n]


class PPOTrainer:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig | None = None,
                 device: str = "cpu"):
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(device)
        self.net = ActorCritic(obs_dim, act_dim, self.cfg).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)

    def update(self, buf, last_val, aux_target=None, aux_mask=None) -> dict:
        cfg = self.cfg
        adv, ret = buf.compute_gae(last_val, cfg.gamma, cfg.gae_lambda)
        n = buf.ptr
        obs = torch.as_tensor(buf.obs[:n], device=self.device)
        act = torch.as_tensor(buf.act[:n], device=self.device)
        old_logp = torch.as_tensor(buf.logp[:n], device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(ret, device=self.device)
        has_aux = aux_target is not None and cfg.aux_coef > 0
        if has_aux:
            aux_t = torch.as_tensor(np.asarray(aux_target, np.float32), device=self.device)
            aux_m = torch.as_tensor(np.asarray(
                aux_mask if aux_mask is not None else np.ones(n), np.float32), device=self.device)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0, "aux_loss": 0.0}
        batches = 0
        for _ in range(cfg.epochs):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, cfg.minibatch):
                b = idx[start:start + cfg.minibatch]
                logp, ent, val, aux = self.net.evaluate(obs[b], act[b])
                ratio = (logp - old_logp[b]).exp()
                p_loss = -torch.min(
                    ratio * adv_t[b],
                    torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_t[b],
                ).mean()
                v_loss = ((val - ret_t[b]) ** 2).mean()
                loss = p_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent.mean()
                if has_aux:
                    m = aux_m[b]
                    denom = m.sum().clamp(min=1.0)
                    a_loss = (((aux - aux_t[b]) ** 2) * m).sum() / denom
                    loss = loss + cfg.aux_coef * a_loss
                    stats["aux_loss"] += a_loss.item()

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    kl = (old_logp[b] - logp).mean().item()
                stats["policy_loss"] += p_loss.item()
                stats["value_loss"] += v_loss.item()
                stats["entropy"] += ent.mean().item()
                stats["kl"] += kl
                batches += 1
            if cfg.target_kl is not None and abs(stats["kl"] / max(batches, 1)) > cfg.target_kl:
                break  # early stop before the policy moves too far

        for k in stats:
            stats[k] /= max(batches, 1)
        buf.reset()
        return stats
