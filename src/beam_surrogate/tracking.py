"""Local MLflow tracking for generation, training and evaluation."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from beam_surrogate.config import SurrogateConfig
from beam_surrogate.paths import project_root


class RunLog:
    """Metric and artifact writer. A no-op when tracking is disabled."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def metric(self, key: str, value: float, step: int | None = None) -> None:
        if not self.enabled:
            return
        import mlflow

        if step is None:
            mlflow.log_metric(key, float(value))
        else:
            mlflow.log_metric(key, float(value), step=int(step))

    def artifact(self, path: Path) -> None:
        if not self.enabled or not path.is_file():
            return
        import mlflow

        mlflow.log_artifact(str(path))


def tracking_uri() -> str:
    """SQLite store in the project directory. A file store is rejected by MLflow 3."""
    override = os.environ.get("BEAM_SURROGATE_MLFLOW_URI")
    if override:
        return override
    database = (project_root() / "mlflow.db").resolve()
    return "sqlite:///" + database.as_posix()


def _artifact_root() -> str:
    override = os.environ.get("BEAM_SURROGATE_MLFLOW_ARTIFACTS")
    root = Path(override) if override else project_root() / "mlartifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve().as_uri()


def _params(cfg: SurrogateConfig) -> dict[str, str | int | float]:
    train = cfg.training
    params: dict[str, str | int | float] = {
        "max_cpus": cfg.max_cpus,
        "seed": cfg.seed,
        "particles_per_batch": cfg.particles_per_batch,
        "n_batches": cfg.n_batches,
        "n_steps": cfg.n_steps,
        "field_nr": cfg.field_nr,
        "field_nz": cfg.field_nz,
        "sigma_x_min": cfg.sigma_x[0],
        "sigma_x_max": cfg.sigma_x[1],
        "spread_min": cfg.spread[0],
        "spread_max": cfg.spread[1],
        "J1_min": cfg.j1[0],
        "J1_max": cfg.j1[1],
        "J2_min": cfg.j2[0],
        "J2_max": cfg.j2[1],
        "epochs": train.epochs,
        "batch_size": train.batch_size,
        "train_aug_factor": train.train_aug_factor,
        "base_lr": train.base_lr,
        "weight_decay": train.weight_decay,
        "hidden_dims": "-".join(str(h) for h in train.hidden_dims),
        "l_drift": train.l_drift,
        "eval_particles": cfg.eval.n_particles,
    }
    if cfg.source_path is not None:
        params["config"] = cfg.source_path.name
    return params


@contextmanager
def tracked_run(stage: str, cfg: SurrogateConfig, *, enabled: bool = True) -> Iterator[RunLog]:
    """Open one MLflow run in the local SQLite store ``mlflow.db``."""
    if not enabled:
        yield RunLog(enabled=False)
        return

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    if mlflow.get_experiment_by_name("beam-surrogate") is None:
        mlflow.create_experiment("beam-surrogate", artifact_location=_artifact_root())
    mlflow.set_experiment("beam-surrogate")
    try:
        mlflow.set_system_metrics_sampling_interval(1)
        mlflow.set_system_metrics_samples_before_logging(1)
        mlflow.enable_system_metrics_logging()
    except Exception:
        pass

    with mlflow.start_run(run_name=stage):
        mlflow.set_tag("stage", stage)
        mlflow.log_params(_params(cfg))
        if cfg.source_path is not None and cfg.source_path.is_file():
            mlflow.log_artifact(str(cfg.source_path))
        yield RunLog(enabled=True)
