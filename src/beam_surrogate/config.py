"""YAML profile for dataset generation and CPU training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beam_surrogate.paths import project_root


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a pair [lo, hi]")
    lo, hi = float(value[0]), float(value[1])
    if lo > hi:
        raise ValueError(f"{name} lower bound {lo} is above upper bound {hi}")
    return lo, hi


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path


@dataclass
class TrainingConfig:
    hidden_dims: tuple[int, ...]
    epochs: int
    batch_size: int
    train_aug_factor: int
    base_lr: float
    weight_decay: float
    split_seed: int
    l_drift: float
    std_in: float
    std_out: float
    j_ref: float
    z_ref: float


@dataclass
class EvalConfig:
    n_particles: int
    seed: int
    sigma_x: float
    spread: float
    j1: float
    j2: float
    z1: float
    z2: float


@dataclass
class SurrogateConfig:
    max_cpus: int
    seed: int
    particles_per_batch: int
    n_batches: int
    n_steps: int
    field_nr: int
    field_nz: int
    sigma_x: tuple[float, float]
    spread: tuple[float, float]
    j1: tuple[float, float]
    j2: tuple[float, float]
    z1: tuple[float, float]
    z2: tuple[float, float]
    dataset_path: Path
    checkpoint_path: Path
    training: TrainingConfig
    eval: EvalConfig
    source_path: Path | None = None

    @property
    def history_path(self) -> Path:
        return self.checkpoint_path.with_name(self.checkpoint_path.stem + "_history.json")


def load_config(path: str | Path) -> SurrogateConfig:
    """Load a surrogate YAML. Relative paths are resolved from this project."""
    import yaml

    raw_path = Path(path)
    if not raw_path.is_file():
        alt = project_root() / raw_path
        if alt.is_file():
            raw_path = alt
        else:
            raise FileNotFoundError(f"Config not found: {path}")

    data = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {raw_path}")

    ranges = data.get("ranges") or {}
    field = data.get("field_grid") or {}
    train = data.get("training") or {}
    ev = data.get("eval") or {}

    hidden = train.get("hidden_dims", [32, 16, 8])
    if not isinstance(hidden, (list, tuple)) or not hidden:
        raise ValueError("training.hidden_dims must be a non-empty list")

    cfg = SurrogateConfig(
        max_cpus=int(data.get("max_cpus", 12)),
        seed=int(data.get("seed", 104729)),
        particles_per_batch=int(data["particles_per_batch"]),
        n_batches=int(data["n_batches"]),
        n_steps=int(data["n_steps"]),
        field_nr=int(field.get("nr", 300)),
        field_nz=int(field.get("nz", 2000)),
        sigma_x=_pair(ranges["sigma_x"], "ranges.sigma_x"),
        spread=_pair(ranges["spread"], "ranges.spread"),
        j1=_pair(ranges["J1"], "ranges.J1"),
        j2=_pair(ranges["J2"], "ranges.J2"),
        z1=_pair(ranges["z1"], "ranges.z1"),
        z2=_pair(ranges["z2"], "ranges.z2"),
        dataset_path=_resolve(data.get("dataset", "data/structured_beam_data.h5")),
        checkpoint_path=_resolve(data.get("checkpoint", "artifacts/best_beam_model.pth")),
        training=TrainingConfig(
            hidden_dims=tuple(int(h) for h in hidden),
            epochs=int(train.get("epochs", 25)),
            batch_size=int(train.get("batch_size", 2048)),
            train_aug_factor=int(train.get("train_aug_factor", 4)),
            base_lr=float(train.get("base_lr", 1e-3)),
            weight_decay=float(train.get("weight_decay", 1e-5)),
            split_seed=int(train.get("split_seed", 42)),
            l_drift=float(train.get("l_drift", 1.5)),
            std_in=float(train.get("std_in", 1e-3)),
            std_out=float(train.get("std_out", 1e-4)),
            j_ref=float(train.get("j_ref", 6.5e6)),
            z_ref=float(train.get("z_ref", 1.0)),
        ),
        eval=EvalConfig(
            n_particles=int(ev.get("n_particles", 2000)),
            seed=int(ev.get("seed", 2024)),
            sigma_x=float(ev.get("sigma_x", 1e-4)),
            spread=float(ev.get("spread", 1e-4)),
            j1=float(ev.get("J1", 2.5e6)),
            j2=float(ev.get("J2", 2.5e6)),
            z1=float(ev.get("z1", 0.4)),
            z2=float(ev.get("z2", 1.2)),
        ),
        source_path=raw_path.resolve(),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: SurrogateConfig) -> None:
    if cfg.max_cpus < 1:
        raise ValueError("max_cpus must be >= 1")
    if cfg.particles_per_batch < 1 or cfg.n_batches < 1 or cfg.n_steps < 1:
        raise ValueError("particles_per_batch, n_batches and n_steps must be >= 1")
    if cfg.field_nr < 4 or cfg.field_nz < 4:
        raise ValueError("field_grid.nr and nz must be >= 4")
    tr = cfg.training
    if tr.epochs < 1 or tr.batch_size < 1 or tr.train_aug_factor < 1:
        raise ValueError("training epochs, batch_size and train_aug_factor must be >= 1")
    if tr.std_in <= 0 or tr.std_out <= 0 or tr.j_ref == 0 or tr.z_ref == 0:
        raise ValueError("normalization scales must be non-zero")
    if cfg.eval.n_particles < 1:
        raise ValueError("eval.n_particles must be >= 1")
