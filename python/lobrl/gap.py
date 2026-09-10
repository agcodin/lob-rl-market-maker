"""Supervised estimator of the latent fundamental gap from public market state."""

from __future__ import annotations

import torch
import torch.nn as nn

# Market-visible slice of the observation: book levels, imbalance, spread,
# volatility and the tape. Deliberately excludes the agent-private tail
# (inventory, PnL, queue position) so the estimate is a property of the market
# and can be computed once per step and shared by every agent.
MARKET_COLS = list(range(0, 21)) + [22, 23] + list(range(28, 32))

# Lags fed to the estimator. History lives INSIDE the estimator, not in the
# observation: the policy still sees one scalar estimate, so the observation
# width and every existing champion stay valid. Measured offline, lags take the
# estimate from corr 0.495 to 0.543.
HISTORY_LAGS = (2, 4, 8, 16, 32)


def stack(rows):
    """Build one estimator input from a history buffer.

    `rows` is a sequence of market-feature vectors, oldest first, with the
    current row last. Short buffers repeat the oldest available row, so the
    estimator sees a well-formed input from the first step of an episode.
    """
    import numpy as np

    cur = rows[-1]
    out = [cur]
    for lag in HISTORY_LAGS:
        out.append(rows[max(0, len(rows) - 1 - lag)])
    return np.concatenate(out).astype("float32")


def input_dim(n_market: int) -> int:
    return n_market * (1 + len(HISTORY_LAGS))


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


class GRUGapNet(nn.Module):
    """Recursive estimator of the latent fundamental gap.

    The gap is a latent random walk observed through noisy signed order flow --
    a linear-Gaussian state-space problem whose optimal estimator is recursive,
    not a fixed window. A lag stack truncates that recursion; a GRU learns it.
    Measured on identical on-policy data: lag stack corr 0.547, GRU 0.608.

    It also costs less to run: one hidden vector carried forward per step, no
    history buffer.
    """

    def __init__(self, dim: int, hidden: int = 96):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.std = nn.Parameter(torch.ones(dim), requires_grad=False)
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, h=None):
        """x: (batch, time, dim). Returns (batch, time) predictions and state."""
        z, h = self.gru((x - self.mean) / self.std, h)
        return self.head(z).squeeze(-1), h


def load(path: str):
    """Returns (net, columns, gap_sd). `net.recursive` says how to drive it."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    arch = blob.get("arch", "mlp")
    if arch == "gru":
        net = GRUGapNet(blob["dim"], blob.get("hidden", 96))
    else:
        net = GapNet(blob["dim"])
    net.load_state_dict(blob["state"])
    net.eval()
    net.recursive = arch == "gru"
    return net, blob["cols"], max(float(blob.get("gap_sd", 1.0)), 1e-6)
