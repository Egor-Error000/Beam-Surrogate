"""MLflow file store receives a metric without starting a server."""

from __future__ import annotations

from pathlib import Path

from beam_surrogate.config import load_config
from beam_surrogate.tracking import tracked_run


def test_tracked_run_writes_metric(tmp_path: Path, monkeypatch):
    uri = "sqlite:///" + (tmp_path / "mlflow.db").resolve().as_posix()
    monkeypatch.setenv("BEAM_SURROGATE_MLFLOW_URI", uri)
    monkeypatch.setenv("BEAM_SURROGATE_MLFLOW_ARTIFACTS", str(tmp_path / "mlartifacts"))
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    cfg = load_config("configs/smoke.yaml")
    with tracked_run("unit", cfg) as log:
        log.metric("wall_time_s", 1.5)
        log.metric("batch_time_s", 0.4, step=0)

    import mlflow

    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name("beam-surrogate")
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert float(runs.iloc[0]["metrics.wall_time_s"]) == 1.5
    assert runs.iloc[0]["tags.stage"] == "unit"
