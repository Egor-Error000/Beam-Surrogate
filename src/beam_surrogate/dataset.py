"""Feature contract shared with the course notebook (block IX).

Import this module only after ``apply_cpu_limit``: it loads PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from beam_surrogate.config import TrainingConfig


@dataclass(frozen=True)
class Norms:
    l_drift: float = 1.5
    mu_in: float = 0.0
    std_in: float = 1e-3
    mu_out: float = 0.0
    std_out: float = 1e-4
    j_ref: float = 6.5e6
    z_ref: float = 1.0

    @classmethod
    def from_training(cls, training: TrainingConfig) -> Norms:
        return cls(
            l_drift=training.l_drift,
            std_in=training.std_in,
            std_out=training.std_out,
            j_ref=training.j_ref,
            z_ref=training.z_ref,
        )


def make_features(
    input_state: np.ndarray,
    output_state: np.ndarray,
    setup_params: np.ndarray,
    norms: Norms,
) -> tuple[np.ndarray, np.ndarray]:
    """Build normalized ``X (N, 8)`` and residual targets ``Y (N, 4)``."""
    x0 = input_state[:, 0]
    y0 = input_state[:, 1]
    xp0 = input_state[:, 3] / input_state[:, 5]
    yp0 = input_state[:, 4] / input_state[:, 5]

    x1 = output_state[:, 0]
    y1 = output_state[:, 1]
    xp1 = output_state[:, 3] / output_state[:, 5]
    yp1 = output_state[:, 4] / output_state[:, 5]

    dx = x1 - (x0 + norms.l_drift * xp0)
    dy = y1 - (y0 + norms.l_drift * yp0)
    dxp = xp1 - xp0
    dyp = yp1 - yp0

    beam_in = np.stack([x0, y0, xp0, yp0], axis=1).astype(np.float32)
    beam_in = (beam_in - np.float32(norms.mu_in)) / np.float32(norms.std_in)

    params = setup_params.astype(np.float32, copy=True)
    params[:, 0] /= np.float32(norms.j_ref)
    params[:, 1] /= np.float32(norms.j_ref)
    params[:, 2] /= np.float32(norms.z_ref)
    params[:, 3] /= np.float32(norms.z_ref)

    features = np.concatenate([beam_in, params], axis=1).astype(np.float32)
    targets = np.stack([dx, dy, dxp, dyp], axis=1).astype(np.float32)
    targets = (targets - np.float32(norms.mu_out)) / np.float32(norms.std_out)
    return features, targets


class BeamDataset(Dataset):
    """HDF5 rows kept in memory, matching the notebook loader."""

    def __init__(self, file_path, norms: Norms | None = None):
        import h5py

        self.norms = norms or Norms()
        with h5py.File(file_path, "r") as handle:
            input_state = handle["input_state"][:]
            output_state = handle["output_state"][:]
            setup_params = handle["setup_params"][:]
        features, targets = make_features(input_state, output_state, setup_params, self.norms)
        self.X = torch.from_numpy(features)
        self.Y = torch.from_numpy(targets)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
