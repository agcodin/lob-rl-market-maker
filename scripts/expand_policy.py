#!/usr/bin/env python
"""Widen a trained policy's action head so it can also choose quote size.

The two new outputs start at zero weight and zero bias, and the env maps a zero
size action to exactly `quote_size` -- so the expanded policy is behaviourally
identical to its parent at step 0 and can only diverge by learning something the
parent could not express. Optionally widens the hidden layers too, copying the
parent's weights into the top-left block and leaving the rest small and random.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lobrl.ppo import ActorCritic, PPOConfig


def expand(src: str, dst: str, act_dim: int = 4, hidden: int | None = None,
           new_log_std: float = -0.7, obs_dim: int | None = None) -> None:
    b = torch.load(src, map_location="cpu", weights_only=False)
    old_act, old_obs = b["act_dim"], b["obs_dim"]
    obs_dim = obs_dim or old_obs
    if obs_dim < old_obs:
        raise SystemExit(f"cannot shrink the observation ({old_obs} -> {obs_dim})")
    old_hidden = b["model"]["actor.0.weight"].shape[0]
    hidden = hidden or old_hidden
    if act_dim < old_act:
        raise SystemExit(f"cannot shrink the action head ({old_act} -> {act_dim})")

    net = ActorCritic(obs_dim, act_dim, PPOConfig(hidden=hidden))
    sd, old = net.state_dict(), b["model"]
    norm = dict(b["norm"])
    if obs_dim > old_obs:
        # New observation slots start with an identity normalization; their
        # input weights are zeroed below, so they cannot move the output yet.
        pad = obs_dim - old_obs
        norm = {"mean": np.concatenate([norm["mean"], np.zeros(pad)]),
                "var": np.concatenate([norm["var"], np.ones(pad)]),
                "count": norm["count"]}

    with torch.no_grad():
        for key, t in sd.items():
            o = old.get(key)
            if o is None:
                continue
            if t.shape == o.shape:
                t.copy_(o)
                continue
            # Copy the parent into the leading block...
            sl = tuple(slice(0, m) for m in (min(a, b_) for a, b_ in zip(t.shape, o.shape)))
            t[sl].copy_(o[sl])
            # ...then silence every new *input* column. New hidden units still
            # get random incoming weights (so they are real units with real
            # activations and real gradients) but contribute nothing to the
            # output yet, which is what makes the expansion function-preserving.
            if t.dim() == 2 and t.shape[1] > o.shape[1]:
                t[:, o.shape[1]:].zero_()
        # New action outputs start dead: zero weight, zero bias, so the env maps
        # them to exactly `quote_size`.
        sd["actor.4.weight"][old_act:].zero_()
        sd["actor.4.bias"][old_act:].zero_()
        sd["log_std"][old_act:] = new_log_std

    net.load_state_dict(sd)
    torch.save({"model": net.state_dict(), "norm": norm, "obs_dim": obs_dim,
                "act_dim": act_dim, "n_agents": b.get("n_agents", 4),
                "expanded_from": str(src)}, dst)
    print(f"{src}  obs {old_obs}->{obs_dim}  act {old_act}->{act_dim}  "
          f"hidden {old_hidden}->{hidden}  ->  {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--act-dim", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--obs-dim", type=int, default=None)
    args = ap.parse_args()
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    expand(args.src, args.dst, args.act_dim, args.hidden, obs_dim=args.obs_dim)


if __name__ == "__main__":
    main()
