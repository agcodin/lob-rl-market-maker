"""Overnight league training: a learner improving against a growing opponent pool.

Plain self-play plateaus and can cycle -- a policy learns to beat its current
self, forgets what beat its older self, and goes round in circles. This runs
prioritized fictitious self-play instead:

  * the learner holds some seats; the rest are frozen opponents,
  * opponents are sampled from a pool of past snapshots plus the live policy
    and the occasional heuristic, so old strategies keep having to be beaten,
  * a candidate is only promoted to "best" if it actually beats the incumbent
    head to head over a batch of shared-seed episodes.

That promotion gate is what makes the run monotone rather than merely long.
Everything here is offline: no network, no external services.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from lobrl.baselines import AvellanedaStoikovPolicy, FixedSpreadPolicy
from lobrl.league import _AgentView
from lobrl.metrics import summarize
from lobrl.flow import FlowConfig
from lobrl.multi_env import MultiAgentMarketMakingEnv, MultiEnvConfig
from lobrl.ppo import ActorCritic, PPOConfig, PPOTrainer, RunningNorm
from lobrl.selfplay import MultiRolloutBuffer


@dataclass
class ArenaConfig:
    n_agents: int = 4
    learner_seats: int = 2
    hours: float = 6.0
    chunk_transitions: int = 150_000   # one generation of training
    batch: int = 2048
    episode_steps: int = 1_000
    eval_episodes: int = 16
    p_self: float = 0.45               # opponent seat is the live learner
    p_baseline: float = 0.15           # opponent seat is a heuristic quoter
    pool_recent: int = 6               # bias sampling toward recent snapshots
    pool_max: int = 60
    # Promote on risk-adjusted return, not raw PnL. Raw PnL is nearly blind to
    # inventory (measured +0.16 correlation) while Sharpe strongly penalises it
    # (-0.58), so a PnL gate lets a policy drift into inventory gambling and
    # still certify each drifted step as an "improvement".
    promote_metric: str = "sharpe"     # "sharpe" | "pnl"
    promote_margin: float = 0.0
    # Hard guardrail for an unattended run: never promote a candidate carrying
    # far more inventory than the incumbent, whatever its score says.
    max_inventory_ratio: float = 1.35
    max_abs_inventory: float = 40.0
    patience: int = 8                  # failed generations before reverting to best
    # 5e-3 (the single-agent default) is too weak here: against a pool of mixed
    # opponents the learner drifts toward inventory gambling within minutes.
    # Measured: at 5e-3 the candidate ran |inv| ~16 and never beat the incumbent;
    # at 2e-2 it holds ~12 and beats it by +5.6 Sharpe (t=2.9, 48 paired episodes).
    inventory_penalty: float = 2e-2
    # Richer action space: the learner also chooses how many shares to show.
    variable_size: bool = False
    min_quote_size: int = 2
    max_quote_size: int = 40
    # Tape features + informed takers. Together these turn adverse selection
    # into something a policy can actually learn to see and avoid.
    flow_features: bool = False
    informed_frac: float = 0.0
    fundamental_vol: float = 0.0
    aux_coef: float = 2.0        # weight on the auxiliary supervised loss
    aux_target: str = "gap"
    gap_predictor: str | None = None
    lean_col: int | None = None
    lean_init: float = 0.4      # "gap" (latent fundamental) or "fwd" (return)
    hidden: int = 128
    seed: int = 0


class FrozenPolicy:
    """A snapshot that acts but never learns."""

    def __init__(self, net_state, norm_state, obs_dim, act_dim, label):
        hidden = net_state["actor.0.weight"].shape[0]
        # A checkpoint that carries a lean vector was trained with the skip
        # connection; rebuild it the same way or the weights will not load.
        lean_col = obs_dim - 1 if "lean" in net_state else None
        self.net = ActorCritic(obs_dim, act_dim,
                               PPOConfig(hidden=hidden, lean_col=lean_col))
        # Older checkpoints predate the auxiliary head; its fresh init is fine.
        self.net.load_state_dict(net_state, strict=False)
        self.net.eval()
        self.norm = RunningNorm(obs_dim)
        self.norm.load_state_dict(norm_state)
        self.label = label

    @classmethod
    def from_live(cls, net, norm, obs_dim, act_dim, label):
        return cls(copy.deepcopy(net.state_dict()),
                   copy.deepcopy(norm.state_dict()), obs_dim, act_dim, label)

    @classmethod
    def load(cls, path):
        b = torch.load(path, map_location="cpu", weights_only=False)
        return cls(b["model"], b["norm"], b["obs_dim"], b["act_dim"], Path(path).stem)

    def save(self, path, obs_dim, act_dim):
        torch.save({"model": self.net.state_dict(), "norm": self.norm.state_dict(),
                    "obs_dim": obs_dim, "act_dim": act_dim, "label": self.label}, path)

    def reset(self):
        pass

    @property
    def act_dim(self):
        return self.net.log_std.numel()

    def act_batch(self, obs, deterministic=False):
        with torch.no_grad():
            a, _, _ = self.net.act(torch.as_tensor(self.norm(obs)), deterministic=deterministic)
        return np.clip(a.numpy(), -1.0, 1.0)

    def __call__(self, obs, view):
        return self.act_batch(obs[None, :])[0]


class _Heuristic:
    """Wraps a baseline so it can hold a seat in the league."""

    def __init__(self, policy, label):
        self.policy = policy
        self.label = label

    def reset(self):
        self.policy.reset()

    def act_batch(self, obs, view=None, deterministic=True):
        return np.array([self.policy(obs[0], view)])


def make_heuristics():
    return [
        _Heuristic(AvellanedaStoikovPolicy(gamma=0.05, k=1.5), "avellaneda-stoikov"),
        _Heuristic(FixedSpreadPolicy(3.0), "fixed-3t"),
        _Heuristic(FixedSpreadPolicy(1.5), "fixed-1.5t"),
    ]


# --------------------------------------------------------------------------
# head-to-head evaluation
# --------------------------------------------------------------------------

def head_to_head(a, b, cfg: ArenaConfig, seeds, labels=("candidate", "incumbent")) -> dict:
    """Run `a` and `b` in the same book over shared seeds, seats rotating."""
    n = cfg.n_agents
    out = {labels[0]: [], labels[1]: []}
    for e, seed in enumerate(seeds):
        # Alternate which policy owns the even seats so neither keeps an edge.
        assign = [(a if (i + e) % 2 == 0 else b) for i in range(n)]
        names = [labels[0] if (i + e) % 2 == 0 else labels[1] for i in range(n)]
        res = _run_episode(assign, cfg, seed)
        for i, nm in enumerate(names):
            out[nm].append(res[i])
    return {k: {"pnl": float(np.mean([r["final_pnl"] for r in v])),
                "sharpe": float(np.mean([r["sharpe"] for r in v])),
                "abs_inv": float(np.mean([r["mean_abs_inventory"] for r in v])),
                "fills": float(np.mean([r["n_fills"] for r in v])),
                "episodes": len(v)}
            for k, v in out.items()}


def env_config(cfg: ArenaConfig, n: int) -> MultiEnvConfig:
    return MultiEnvConfig(n_agents=n, max_steps=cfg.episode_steps,
                          inventory_penalty=cfg.inventory_penalty,
                          variable_size=cfg.variable_size,
                          min_quote_size=cfg.min_quote_size,
                          max_quote_size=cfg.max_quote_size,
                          flow_features=cfg.flow_features,
                          gap_predictor=cfg.gap_predictor,
                          flow=FlowConfig(informed_frac=cfg.informed_frac,
                                          fundamental_vol=cfg.fundamental_vol))


def _run_episode(policies, cfg: ArenaConfig, seed: int) -> list[dict]:
    n = len(policies)
    env = MultiAgentMarketMakingEnv(env_config(cfg, n), seed=seed)
    obs, _ = env.reset(seed=seed)
    for p in policies:
        p.reset()
    views = [_AgentView(env, i) for i in range(n)]
    # Seats sharing a policy object are evaluated in one forward pass; in a
    # two-policy match that halves the work per step.
    groups: dict[int, list[int]] = {}
    for i, p in enumerate(policies):
        groups.setdefault(id(p), []).append(i)
    eq = np.zeros((cfg.episode_steps, n))
    inv = np.zeros((cfg.episode_steps, n))
    mids = []
    t = 0
    while True:
        # A 2-d policy writes into the first two columns and leaves the size
        # columns at 0, which the env maps to exactly `quote_size` -- so old
        # policies keep their old behaviour in a variable-size book.
        acts = np.zeros((n, env.act_dim), np.float32)
        for seats in groups.values():
            p = policies[seats[0]]
            if isinstance(p, _Heuristic):
                for i in seats:
                    views[i].sync(i)
                    a = p.act_batch(obs[i:i + 1], views[i])[0]
                    acts[i, :len(a)] = a
            else:
                a = p.act_batch(obs[seats], deterministic=True)
                acts[seats, :a.shape[1]] = a
        obs, _, term, trunc, info = env.step(acts)
        eq[t] = info["equity"]
        inv[t] = info["inventory"]
        mids.append(info["mid"])
        t += 1
        if term or trunc:
            break
    res = []
    for i in range(n):
        fills = [(s, px, q) for (a_, s, px, q) in env.fill_log if a_ == i]
        res.append(summarize(eq[:t, i], mids, fills, inv[:t, i],
                             steps_per_year=cfg.episode_steps * 252))
    return res


# --------------------------------------------------------------------------
# the overnight loop
# --------------------------------------------------------------------------

class Arena:
    def __init__(self, cfg: ArenaConfig, outdir: Path):
        self.cfg = cfg
        self.dir = outdir
        self.pool_dir = outdir / "pool"
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)

        probe = MultiAgentMarketMakingEnv(env_config(cfg, cfg.n_agents), seed=cfg.seed)
        self.obs_dim = probe.observation_space.shape[0]
        self.act_dim = probe.action_space.shape[0]

        self.trainer = PPOTrainer(self.obs_dim, self.act_dim,
                                  PPOConfig(hidden=cfg.hidden, aux_coef=cfg.aux_coef,
                                            lean_col=cfg.lean_col, lean_init=cfg.lean_init))
        self.norm = RunningNorm(self.obs_dim)
        self.heuristics = make_heuristics()
        self.gen = 0
        self.history = []
        self.stop = False
        self._resume()

    # ---- persistence ---------------------------------------------------

    def _resume(self):
        best = self.dir / "best.pt"
        state = self.dir / "state.json"
        if best.exists():
            b = torch.load(best, map_location="cpu", weights_only=False)
            self.trainer.net.load_state_dict(b["model"], strict=False)
            self.norm.load_state_dict(b["norm"])
            print(f"resumed learner from {best}")
        if state.exists():
            st = json.loads(state.read_text())
            self.gen = st.get("generation", 0)
            print(f"resuming at generation {self.gen}")
        hist = self.dir / "history.jsonl"
        if hist.exists():
            self.history = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()]

    def _save_best(self, policy: "FrozenPolicy | None" = None):
        """Write `best.pt`. Pass the incumbent explicitly when the live learner
        is not the champion -- at the end of a run the learner has usually moved
        past the last promoted policy, and saving it would discard the ratchet."""
        net = policy.net if policy is not None else self.trainer.net
        norm = policy.norm if policy is not None else self.norm
        torch.save({"model": net.state_dict(), "norm": norm.state_dict(),
                    "obs_dim": self.obs_dim, "act_dim": self.act_dim,
                    "n_agents": self.cfg.n_agents, "generation": self.gen},
                   self.dir / "best.pt")

    def _snapshot(self):
        p = FrozenPolicy.from_live(self.trainer.net, self.norm, self.obs_dim,
                                   self.act_dim, f"gen{self.gen:04d}")
        p.save(self.pool_dir / f"gen_{self.gen:04d}.pt", self.obs_dim, self.act_dim)
        self._prune_pool()
        return p

    def _prune_pool(self):
        """Keep every recent snapshot and thin the old ones, so the pool stays
        diverse without growing without bound."""
        files = sorted(self.pool_dir.glob("gen_*.pt"))
        if len(files) <= self.cfg.pool_max:
            return
        keep = set(files[-self.cfg.pool_recent:])
        older = files[:-self.cfg.pool_recent]
        # Ceiling division: floor division rounds the stride down to 1 as the
        # pool grows, which silently keeps every snapshot and defeats the cap.
        budget = max(1, self.cfg.pool_max - self.cfg.pool_recent)
        stride = max(1, -(-len(older) // budget))
        keep |= set(older[::stride])
        for f in files:
            if f not in keep:
                f.unlink(missing_ok=True)

    def _write_status(self, extra=None):
        rows = self.history[-1] if self.history else {}
        status = {
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generation": self.gen,
            "elapsed_hours": round((time.time() - self.t0) / 3600, 3),
            "remaining_hours": round(max(0.0, (self.deadline - time.time()) / 3600), 3),
            "transitions": self.transitions,
            "promotions": sum(1 for h in self.history if h.get("promoted")),
            "pool_size": len(list(self.pool_dir.glob("gen_*.pt"))),
            "last": rows,
        }
        if extra:
            status.update(extra)
        (self.dir / "status.json").write_text(json.dumps(status, indent=2))

    # ---- opponents -----------------------------------------------------

    def _pool_files(self):
        return sorted(self.pool_dir.glob("gen_*.pt"))

    def _sample_opponent(self, live: FrozenPolicy):
        u = self.rng.random()
        if u < self.cfg.p_self:
            return live
        if u < self.cfg.p_self + self.cfg.p_baseline:
            return self.heuristics[self.rng.integers(len(self.heuristics))]
        files = self._pool_files()
        if not files:
            return live
        # Recent snapshots are the hardest opponents; old ones stop the learner
        # from forgetting what already beat it.
        if len(files) > self.cfg.pool_recent and self.rng.random() < 0.65:
            f = files[-self.cfg.pool_recent:][self.rng.integers(self.cfg.pool_recent)]
        else:
            f = files[self.rng.integers(len(files))]
        return FrozenPolicy.load(f)

    # ---- one generation of training ------------------------------------

    def train_chunk(self) -> dict:
        cfg = self.cfg
        L = cfg.learner_seats
        n = cfg.n_agents
        rollout = max(1, cfg.batch // L)
        env = MultiAgentMarketMakingEnv(env_config(cfg, n), seed=int(self.rng.integers(1 << 30)))
        obs, _ = env.reset(seed=int(self.rng.integers(1 << 30)))

        live = FrozenPolicy.from_live(self.trainer.net, self.norm, self.obs_dim,
                                      self.act_dim, "live")
        opponents = [self._sample_opponent(live) for _ in range(n - L)]
        views = [_AgentView(env, i) for i in range(n)]

        done_transitions = 0
        ep_ret, ep_stats = np.zeros(L), []
        stats = {}
        while done_transitions < cfg.chunk_transitions and not self.stop:
            buf = MultiRolloutBuffer(rollout, L, self.obs_dim, self.act_dim)
            while not buf.full():
                lobs = obs[:L]
                self.norm.update(lobs)
                nobs = self.norm(lobs)
                with torch.no_grad():
                    a, logp, v = self.trainer.net.act(torch.as_tensor(nobs))
                acts = np.zeros((n, env.act_dim), np.float32)
                acts[:L] = np.clip(a.numpy(), -1, 1)
                for j, opp in enumerate(opponents):
                    i = L + j
                    views[i].sync(i)
                    oa = (opp.act_batch(obs[i:i + 1], views[i])[0]
                          if isinstance(opp, _Heuristic) else opp.act_batch(obs[i:i + 1])[0])
                    acts[i, :len(oa)] = oa
                nxt, rew, term, trunc, info = env.step(acts)
                buf.add(nobs, acts[:L], logp.numpy(), rew[:L], v.numpy(),
                        np.full(L, 1.0 if trunc else 0.0, np.float32), mid=info["mid"],
                        gap=info.get("fundamental_gap", 0.0))
                ep_ret += rew[:L]
                obs = nxt
                if trunc:
                    ep_stats.append({
                        "size": float(info["sizes"][:L].mean()),
                        "ret": float(ep_ret.mean()),
                        "equity": float(info["equity"][:L].mean()),
                        "inv": float(np.abs(info["inventory"][:L]).mean()),
                        "fills": float(info["trades"][:L].mean()),
                        "offset": float(info["offsets"][:L].mean()),
                    })
                    ep_ret[:] = 0.0
                    obs, _ = env.reset()
                    # Fresh draw each episode keeps the opponent mix varied.
                    live = FrozenPolicy.from_live(self.trainer.net, self.norm,
                                                  self.obs_dim, self.act_dim, "live")
                    opponents = [self._sample_opponent(live) for _ in range(n - L)]
            with torch.no_grad():
                last_val = self.trainer.net.critic(
                    torch.as_tensor(self.norm(obs[:L]))).squeeze(-1).numpy()
            aux_t, aux_m = buf.aux_targets(self.trainer.cfg.aux_horizon,
                                           target=self.cfg.aux_target)
            stats = self.trainer.update(buf, last_val, aux_target=aux_t, aux_mask=aux_m)
            done_transitions += rollout * L
            self.transitions += rollout * L
            if time.time() > self.deadline:
                self.stop = True

        recent = ep_stats[-10:]
        agg = {k: float(np.mean([e[k] for e in recent])) for k in
               ("ret", "equity", "inv", "fills", "offset", "size")} if recent else {}
        return {**agg, **{k: float(v) for k, v in stats.items()},
                "episodes": len(ep_stats)}

    # ---- main loop -----------------------------------------------------

    def run(self):
        cfg = self.cfg
        self.t0 = time.time()
        self.deadline = self.t0 + cfg.hours * 3600
        self.transitions = 0

        def handle(signum, frame):
            print(f"\nsignal {signum} -- finishing this generation and saving", flush=True)
            self.stop = True
        signal.signal(signal.SIGTERM, handle)
        signal.signal(signal.SIGINT, handle)

        incumbent = FrozenPolicy.from_live(self.trainer.net, self.norm, self.obs_dim,
                                           self.act_dim, "gen0000")
        since_promotion = 0
        if not self._pool_files():
            self._snapshot()
        self._save_best()
        self._write_status({"phase": "starting"})
        print(f"arena: {cfg.n_agents} seats ({cfg.learner_seats} learner), "
              f"{cfg.hours}h budget, {cfg.chunk_transitions:,} transitions/generation",
              flush=True)

        while time.time() < self.deadline and not self.stop:
            gen_t0 = time.time()
            self.gen += 1
            train = self.train_chunk()

            candidate = FrozenPolicy.from_live(self.trainer.net, self.norm, self.obs_dim,
                                               self.act_dim, f"gen{self.gen:04d}")
            seeds = [int(self.rng.integers(1 << 30)) for _ in range(cfg.eval_episodes)]
            h2h = head_to_head(candidate, incumbent, cfg, seeds)
            metric = cfg.promote_metric
            gain = h2h["candidate"][metric] - h2h["incumbent"][metric]
            cand_inv = h2h["candidate"]["abs_inv"]
            inv_ok = (cand_inv <= cfg.max_abs_inventory and
                      cand_inv <= cfg.max_inventory_ratio * max(h2h["incumbent"]["abs_inv"], 1.0))
            promoted = gain > cfg.promote_margin and inv_ok

            reverted = False
            if promoted:
                incumbent = candidate
                since_promotion = 0
                self._save_best()
                self._snapshot()
            else:
                # Keep training from the candidate weights -- reverting every
                # failed generation would stall progress -- but the recorded
                # best stays the policy that actually won its match.
                since_promotion += 1
                if self.gen % 3 == 0:
                    self._snapshot()
                if since_promotion >= cfg.patience:
                    # Unattended safety valve: the learner has drifted for a
                    # while without beating the incumbent, so restart it from
                    # the best known weights instead of burning hours adrift.
                    self.trainer.net.load_state_dict(copy.deepcopy(incumbent.net.state_dict()))
                    self.norm.load_state_dict(copy.deepcopy(incumbent.norm.state_dict()))
                    since_promotion = 0
                    reverted = True

            row = {
                "generation": self.gen,
                "wall_minutes": round((time.time() - gen_t0) / 60, 2),
                "elapsed_hours": round((time.time() - self.t0) / 3600, 3),
                "transitions": self.transitions,
                "promoted": bool(promoted),
                "reverted": bool(reverted),
                "since_promotion": int(since_promotion),
                "gain_vs_incumbent": round(float(gain), 2),
                "gate_metric": metric,
                "inventory_ok": bool(inv_ok),
                "candidate": {k: round(v, 3) for k, v in h2h["candidate"].items()},
                "incumbent": {k: round(v, 3) for k, v in h2h["incumbent"].items()},
                "train": {k: round(v, 4) for k, v in train.items()},
                "pool_size": len(self._pool_files()),
            }
            self.history.append(row)
            with (self.dir / "history.jsonl").open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            (self.dir / "state.json").write_text(json.dumps(
                {"generation": self.gen, "transitions": self.transitions}))
            self._write_status({"phase": "training"})

            print(f"gen {self.gen:3d} | {row['wall_minutes']:5.1f}m | "
                  f"{self.transitions/1e6:6.2f}M trans | "
                  f"cand sharpe {h2h['candidate']['sharpe']:6.1f} PnL {h2h['candidate']['pnl']:7.0f} "
                  f"|inv| {cand_inv:5.1f} | inc sharpe {h2h['incumbent']['sharpe']:6.1f} "
                  f"| gain {gain:+7.2f}{'' if inv_ok else ' INV!'} "
                  f"| {'PROMOTED' if promoted else ('REVERTED' if reverted else 'held')} "
                  f"| pool {row['pool_size']} "
                  f"| {(self.deadline - time.time())/3600:4.2f}h left", flush=True)

        # The champion is the last policy that actually won its match, not
        # wherever the learner happened to stop.
        self._save_best(incumbent)
        self._write_status({"phase": "finished"})
        print(f"\nfinished: {self.gen} generations, {self.transitions:,} transitions, "
              f"{sum(1 for h in self.history if h['promoted'])} promotions, "
              f"{(time.time()-self.t0)/3600:.2f}h", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--learner-seats", type=int, default=2)
    ap.add_argument("--chunk-transitions", type=int, default=150_000)
    ap.add_argument("--eval-episodes", type=int, default=16)
    ap.add_argument("--episode-steps", type=int, default=1000)
    ap.add_argument("--inventory-penalty", type=float, default=2e-2,
                    help="phi in the training reward")
    ap.add_argument("--p-baseline", type=float, default=0.15,
                    help="chance an opponent seat is a heuristic quoter")
    ap.add_argument("--promote-metric", choices=["sharpe", "pnl"], default="sharpe",
                    help="what a candidate must beat the incumbent on")
    ap.add_argument("--patience", type=int, default=8,
                    help="failed generations before reverting the learner to best.pt")
    ap.add_argument("--aux-target", choices=["gap", "fwd"], default="gap")
    ap.add_argument("--lean-col", type=int, default=None,
                    help="observation column wired straight to the action mean")
    ap.add_argument("--lean-init", type=float, default=0.4)
    ap.add_argument("--gap-predictor", default=None,
                    help="path to a trained fundamental-gap estimator")
    ap.add_argument("--aux-coef", type=float, default=2.0,
                    help="weight on the auxiliary forward-price prediction loss")
    ap.add_argument("--flow-features", action="store_true",
                    help="add tape features to the observation")
    ap.add_argument("--informed-frac", type=float, default=0.0,
                    help="fraction of takers who trade on the latent fundamental")
    ap.add_argument("--fundamental-vol", type=float, default=0.0)
    ap.add_argument("--variable-size", action="store_true",
                    help="let the policy choose quote size as well as price")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/arena")
    args = ap.parse_args()

    cfg = ArenaConfig(n_agents=args.n_agents, learner_seats=args.learner_seats,
                      hours=args.hours, chunk_transitions=args.chunk_transitions,
                      eval_episodes=args.eval_episodes, episode_steps=args.episode_steps,
                      patience=args.patience, seed=args.seed,
                      promote_metric=args.promote_metric,
                      inventory_penalty=args.inventory_penalty,
                      p_baseline=args.p_baseline,
                      variable_size=args.variable_size, hidden=args.hidden,
                      flow_features=args.flow_features,
                      informed_frac=args.informed_frac,
                      fundamental_vol=args.fundamental_vol, aux_coef=args.aux_coef,
                      aux_target=args.aux_target, gap_predictor=args.gap_predictor,
                      lean_col=args.lean_col, lean_init=args.lean_init)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    Arena(cfg, outdir).run()


if __name__ == "__main__":
    main()
