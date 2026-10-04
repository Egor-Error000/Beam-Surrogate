"""Compare the trained MLP with one Beam Tracing run."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from beam_surrogate.config import SurrogateConfig, load_config
from beam_surrogate.generate import write_lattice
from beam_surrogate.limits import apply_cpu_limit
from beam_surrogate.paths import beam_tracing_root, default_run_config
from beam_surrogate.tracking import tracked_run

logger = logging.getLogger(__name__)

INPUT_DIM = 8
OUTPUT_DIM = 4


def _track_case(cfg: SurrogateConfig, scratch: Path):
    from beam_tracing.beam.source import sample_beam
    from beam_tracing.config.loader import load_simulation_config
    from beam_tracing.pipelines.simulation import build_scaling, run_simulation

    ev = cfg.eval
    lattice_path = scratch / "eval_lattice.yaml"
    write_lattice(lattice_path, ev.z1, ev.j1, ev.z2, ev.j2)
    sim = load_simulation_config(default_run_config(), root=beam_tracing_root())
    sim.lattice_file = str(lattice_path.resolve())
    sim.beam.n_particles = ev.n_particles
    sim.beam.sigma_x = ev.sigma_x
    sim.beam.spread_frac = ev.spread
    sim.beam.seed = ev.seed
    sim.beam.length_z = 0.0
    sim.time.n_steps = cfg.n_steps
    sim.field_grid.nr = cfg.field_nr
    sim.field_grid.nz = cfg.field_nz
    sim.space_charge.enabled = False
    sim.output.create_plots = False
    sim.output.save_plots = False
    sim.output.results_dir = str(scratch)
    sim.heatmap.every = 1_000_000_000
    sim.heatmap.k = 2
    sim.heatmap.nz_bins = 2
    _beam_phys, scaling, _beam_num = build_scaling(sim)
    x0, p0 = sample_beam(sim.beam, scaling)
    result = run_simulation(sim, with_plots=False, root=beam_tracing_root())
    input_state = np.concatenate([x0, p0], axis=1).astype(np.float32)
    output_state = np.concatenate([result.x_final, result.p_final], axis=1).astype(np.float32)
    setup = np.tile(np.array([ev.j1, ev.j2, ev.z1, ev.z2], dtype=np.float32), (x0.shape[0], 1))
    return input_state, output_state, setup


def evaluate(cfg: SurrogateConfig, *, track: bool = True) -> dict[str, float]:
    """Score the checkpoint against a fresh Beam Tracing beam."""
    apply_cpu_limit(cfg.max_cpus)
    import torch

    from beam_surrogate.dataset import Norms, make_features
    from beam_surrogate.metrics import physical_errors
    from beam_surrogate.model import BeamPredictor

    apply_cpu_limit(cfg.max_cpus)
    logging.getLogger("beam_tracing").setLevel(logging.WARNING)
    if not cfg.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}. Train first.")

    started = time.perf_counter()
    with tracked_run("evaluate", cfg, enabled=track) as log:
        scratch = cfg.checkpoint_path.parent / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        input_state, output_state, setup = _track_case(cfg, scratch)
        norms = Norms.from_training(cfg.training)
        features, targets = make_features(input_state, output_state, setup, norms)

        device = torch.device("cpu")
        model = BeamPredictor(INPUT_DIM, OUTPUT_DIM, cfg.training.hidden_dims).to(device)
        state = torch.load(cfg.checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            preds = model(torch.from_numpy(features)).numpy()

        mse = float(np.mean((preds - targets) ** 2))
        phys = physical_errors(preds, targets, features, norms)
        wall_s = time.perf_counter() - started
        report = {"mse": mse, **phys, "wall_time_s": wall_s}
        log.metric("mse", mse)
        log.metric("mean_abs_dr_over_r", phys["mean"])
        log.metric("median_abs_dr_over_r", phys["median"])
        log.metric("p95_abs_dr_over_r", phys["p95"])
        log.metric("r_rms_mm", phys["r_rms_mm"])
        log.metric("wall_time_s", wall_s)
        log.metric("n_particles", float(cfg.eval.n_particles))
    logger.info(
        "Eval MSE: %.8f | mean|dr|/r: %.3e | median: %.3e | p95: %.3e | r_rms: %.4f mm | %.2f s",
        report["mse"],
        report["mean"],
        report["median"],
        report["p95"],
        report["r_rms_mm"],
        report["wall_time_s"],
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare the CPU surrogate with Beam Tracing")
    parser.add_argument("--config", required=True, help="Path to cpu.yaml or smoke.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
