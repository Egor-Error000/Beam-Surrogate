"""Generate the HDF5 dataset by calling Beam Tracing once per batch."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from beam_surrogate.config import SurrogateConfig, load_config
from beam_surrogate.limits import apply_cpu_limit
from beam_surrogate.paths import beam_tracing_root, default_run_config
from beam_surrogate.tracking import tracked_run

logger = logging.getLogger(__name__)


def _sample(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    if lo == hi:
        return float(lo)
    return float(rng.uniform(lo, hi))


def write_lattice(path: Path, z1: float, j1: float, z2: float, j2: float) -> None:
    """Two-solenoid line matching ``Beam Tracing/lattices/default.yaml``."""
    import yaml

    document = {
        "origin_z": 0,
        "beamline": [
            {"type": "drift", "length": 0.2},
            {
                "type": "solenoid",
                "position": float(z1),
                "length": 0.4,
                "J": float(j1),
                "R_inner": 0.08,
                "R_outer": 0.12,
            },
            {"type": "drift", "length": 0.4},
            {
                "type": "solenoid",
                "position": float(z2),
                "length": 0.4,
                "J": float(j2),
                "R_inner": 0.08,
                "R_outer": 0.12,
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _append(handle, name: str, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array, dtype=np.float32)
    if name not in handle:
        handle.create_dataset(
            name,
            data=array,
            maxshape=(None, array.shape[1]),
            chunks=(min(8192, array.shape[0]), array.shape[1]),
        )
        return
    dataset = handle[name]
    start = dataset.shape[0]
    dataset.resize(start + array.shape[0], axis=0)
    dataset[start:] = array


def _simulation(cfg: SurrogateConfig, lattice_path: Path, scratch: Path, seed: int, sigma_x: float, spread: float):
    from beam_tracing.config.loader import load_simulation_config

    sim = load_simulation_config(default_run_config(), root=beam_tracing_root())
    sim.lattice_file = str(lattice_path.resolve())
    sim.beam.n_particles = cfg.particles_per_batch
    sim.beam.sigma_x = sigma_x
    sim.beam.spread_frac = spread
    sim.beam.seed = seed
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
    return sim


def generate_dataset(cfg: SurrogateConfig, *, h5_path: Path | None = None, track: bool = True) -> Path:
    """Run batches one after another and store particle states in HDF5."""
    apply_cpu_limit(cfg.max_cpus)
    import h5py
    from beam_tracing.beam.source import sample_beam
    from beam_tracing.pipelines.simulation import build_scaling, run_simulation

    apply_cpu_limit(cfg.max_cpus)
    logging.getLogger("beam_tracing").setLevel(logging.WARNING)

    out = Path(h5_path) if h5_path is not None else cfg.dataset_path
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = out.parent / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    lattice_path = scratch / "batch_lattice.yaml"

    rng = np.random.default_rng(cfg.seed)
    n_samples = cfg.n_batches * cfg.particles_per_batch
    logger.info(
        "Generating %d batches x %d particles, %d steps, <= %d CPUs -> %s",
        cfg.n_batches,
        cfg.particles_per_batch,
        cfg.n_steps,
        cfg.max_cpus,
        out,
    )

    started = time.perf_counter()
    with tracked_run("generate", cfg, enabled=track) as log, h5py.File(out, "w") as handle:
        handle.attrs["max_cpus"] = cfg.max_cpus
        handle.attrs["seed"] = cfg.seed
        handle.attrs["n_steps"] = cfg.n_steps
        for index in range(cfg.n_batches):
            batch_started = time.perf_counter()
            sigma_x = _sample(rng, cfg.sigma_x)
            spread = _sample(rng, cfg.spread)
            j1 = _sample(rng, cfg.j1)
            j2 = _sample(rng, cfg.j2)
            z1 = _sample(rng, cfg.z1)
            z2 = _sample(rng, cfg.z2)
            seed = cfg.seed + index
            write_lattice(lattice_path, z1, j1, z2, j2)
            sim = _simulation(cfg, lattice_path, scratch, seed, sigma_x, spread)
            _beam_phys, scaling, _beam_num = build_scaling(sim)
            x0, p0 = sample_beam(sim.beam, scaling)
            result = run_simulation(sim, with_plots=False, root=beam_tracing_root())
            if x0.shape != result.x_final.shape or p0.shape != result.p_final.shape:
                raise RuntimeError(
                    f"Batch {index}: sampled {x0.shape} but tracker returned {result.x_final.shape}"
                )
            input_state = np.concatenate([x0, p0], axis=1)
            output_state = np.concatenate([result.x_final, result.p_final], axis=1)
            setup = np.tile(
                np.array([j1, j2, z1, z2], dtype=np.float32),
                (x0.shape[0], 1),
            )
            _append(handle, "input_state", input_state)
            _append(handle, "output_state", output_state)
            _append(handle, "setup_params", setup)
            batch_s = time.perf_counter() - batch_started
            log.metric("batch_time_s", batch_s, step=index)
            logger.info(
                "batch %d/%d seed=%d time=%.2f s",
                index + 1,
                cfg.n_batches,
                seed,
                batch_s,
            )
        wall_s = time.perf_counter() - started
        log.metric("wall_time_s", wall_s)
        log.metric("n_samples", float(n_samples))
        log.metric("samples_per_s", n_samples / wall_s if wall_s > 0 else 0.0)
    logger.info(
        "Generation finished in %.2f s | %d samples | %.1f samples/s",
        wall_s,
        n_samples,
        n_samples / wall_s if wall_s > 0 else 0.0,
    )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a beam surrogate dataset with Beam Tracing")
    parser.add_argument("--config", required=True, help="Path to cpu.yaml or smoke.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    path = generate_dataset(cfg)
    print(path)


if __name__ == "__main__":
    main()
