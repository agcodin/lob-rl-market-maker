#!/usr/bin/env python
"""Supervised estimator of the latent fundamental gap from public market state.

Reinforcement learning is a bad way to learn this: the reward signal for
"read the tape" is indirect and buried in variance, and measured training left
the tape input weights near zero. Supervised learning on the same data finds it
in seconds. The estimator uses only market-visible features -- never inventory,
PnL or queue position -- so its output is a property of the market, computed once
per step and shared by every agent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lobrl.gap import MARKET_COLS, GapNet

# Market-visible slice of the observation: book levels, imbalance, spread,
# volatility and the tape. Deliberately excludes the agent-private tail.


def collect(steps: int, seed: int, informed_frac: float, fundamental_vol: float,
            driver: str | None = None, predictor: str | None = None):
    """Roll the market forward and record (market features, true gap).

    `driver` is a policy checkpoint that quotes while the data is collected. It
    matters: a market where nobody quotes is not the market the estimate is used
    in, and measured, an estimator trained on an idle book scores corr 0.502
    there but only 0.406 in the book the champion actually makes.
    """
    from lobrl.flow import FlowConfig
    from lobrl.multi_env import MultiAgentMarketMakingEnv, MultiEnvConfig

    fc = FlowConfig(informed_frac=informed_frac, fundamental_vol=fundamental_vol)
    n = 4 if driver else 2
    env = MultiAgentMarketMakingEnv(
        MultiEnvConfig(n_agents=n, max_steps=steps + 10, flow_features=True, flow=fc,
                       gap_predictor=predictor), seed=seed)
    pol = None
    if driver:
        from lobrl.arena import FrozenPolicy
        pol = FrozenPolicy.load(driver)
    obs, _ = env.reset(seed=seed)
    X, y = [], []
    for _ in range(steps):
        X.append(obs[0][MARKET_COLS].copy())
        y.append(env.flow.fundamental - env.book.mid())
        a = (pol.act_batch(obs, deterministic=False) if pol is not None
             else np.zeros((n, env.act_dim), np.float32))
        obs, *_ = env.step(a)
    return np.array(X, np.float32), np.array(y, np.float32)




def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=120_000)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--informed-frac", type=float, default=0.35)
    ap.add_argument("--fundamental-vol", type=float, default=0.35)
    ap.add_argument("--driver", default=None,
                    help="policy checkpoint that quotes while data is collected")
    ap.add_argument("--driver-predictor", default=None,
                    help="gap predictor the driver policy expects in its observation")
    ap.add_argument("--out", default="runs/gap_predictor.pt")
    args = ap.parse_args()

    print(f"collecting {args.steps:,} steps ...", flush=True)
    kw = dict(driver=args.driver, predictor=args.driver_predictor)
    X, y = collect(args.steps, 4242, args.informed_frac, args.fundamental_vol, **kw)
    Xv, yv = collect(args.steps // 4, 8888, args.informed_frac, args.fundamental_vol, **kw)

    net = GapNet(X.shape[1])
    with torch.no_grad():
        net.mean.copy_(torch.tensor(X.mean(0)))
        net.std.copy_(torch.tensor(X.std(0) + 1e-6))
    opt = torch.optim.Adam(net.net.parameters(), 1e-3)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    Xvt, yvt = torch.tensor(Xv), torch.tensor(yv)

    best, best_state = -1.0, None
    for ep in range(args.epochs):
        idx = torch.randperm(len(Xt))[:8192]
        opt.zero_grad()
        loss = ((net(Xt[idx]) - yt[idx]) ** 2).mean()
        loss.backward()
        opt.step()
        if (ep + 1) % 40 == 0:
            with torch.no_grad():
                p = net(Xvt).numpy()
            c = float(np.corrcoef(p, yv)[0, 1])
            if c > best:
                best, best_state = c, {k: v.clone() for k, v in net.state_dict().items()}
            print(f"  epoch {ep+1:4d}  train mse {loss.item():.3f}  held-out corr {c:+.3f}", flush=True)

    net.load_state_dict(best_state)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "dim": X.shape[1], "cols": MARKET_COLS,
                "corr": best, "gap_sd": float(y.std())}, args.out)
    print(f"saved {args.out}  held-out corr {best:+.3f}  (ridge ceiling was ~+0.29)")


if __name__ == "__main__":
    main()
