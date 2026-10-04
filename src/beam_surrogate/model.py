"""MLP 8 → 32 → 16 → 8 → 4 with GELU, as in the course notebook.

Import this module only after ``apply_cpu_limit``.
"""

from __future__ import annotations

import torch
from torch import nn


class BeamPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...] | list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden)))
            layers.append(nn.GELU())
            prev_dim = int(hidden)
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
