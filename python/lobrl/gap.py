"""Supervised estimator of the latent fundamental gap from public market state."""

from __future__ import annotations

import torch
import torch.nn as nn

# Market-visible slice of the observation: book levels, imbalance, spread,
# volatility and the tape. Deliberately excludes the agent-private tail
# (inventory, PnL, queue position) so the estimate is a property of the market
# and can be computed once per step and shared by every agent.
MARKET_COLS = list(range(0, 21)) + [22, 23] + list(range(28, 32))


class GapNet(nn.Module):
    def __init__(self, dim: int, hidden: int = 96):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.std = nn.Parameter(torch.ones(dim), requires_grad=False)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std).squeeze(-1)


def load(path: str):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net = GapNet(blob["dim"])
    net.load_state_dict(blob["state"])
    net.eval()
    return net, blob["cols"], max(float(blob.get("gap_sd", 1.0)), 1e-6)
