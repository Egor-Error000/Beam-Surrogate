"""Smoke checks: CPU cap, one Beam Tracing batch, one training step."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from beam_surrogate.config import load_config
from beam_surrogate.limits import apply_cpu_limit


def test_cpu_limit_caps_threads():
    n = apply_cpu_limit(12)
    assert n <= 12
    import numba
    import torch

    apply_cpu_limit(12)
    assert numba.get_num_threads() <= 12
    assert torch.get_num_threads() <= 12
    assert torch.get_num_interop_threads() == 1


def test_one_batch_writes_hdf5(tmp_path: Path):
    cfg = load_config("configs/smoke.yaml")
    cfg.n_batches = 1
    cfg.dataset_path = tmp_path / "beam.h5"

    from beam_surrogate.generate import generate_dataset

    path = generate_dataset(cfg, h5_path=tmp_path / "beam.h5", track=False)
    with h5py.File(path, "r") as handle:
        assert handle["input_state"].shape == (cfg.particles_per_batch, 6)
        assert handle["output_state"].shape == (cfg.particles_per_batch, 6)
        assert handle["setup_params"].shape == (cfg.particles_per_batch, 4)
        assert handle.attrs["max_cpus"] == 12
        assert np.isfinite(handle["output_state"][:]).all()
        assert np.isfinite(handle["input_state"][:]).all()


def test_one_training_step_lowers_loss():
    apply_cpu_limit(12)
    import torch
    from torch import nn

    from beam_surrogate.model import BeamPredictor

    apply_cpu_limit(12)
    torch.manual_seed(0)
    model = BeamPredictor(8, 4, [32, 16, 8])
    features = torch.randn(64, 8)
    targets = model(features).detach() + 0.1 * torch.randn(64, 4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    before = float(criterion(model(features), targets).detach())
    optimizer.zero_grad()
    criterion(model(features), targets).backward()
    optimizer.step()
    after = float(criterion(model(features), targets).detach())
    assert after < before
