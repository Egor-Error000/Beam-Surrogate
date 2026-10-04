"""Locations of this project and the sibling Beam Tracing repository."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Directory that contains ``pyproject.toml`` of Beam Surrogate."""
    return Path(__file__).resolve().parents[2]


def beam_tracing_root() -> Path:
    """Sibling ``Beam Tracing`` checkout used as the calculator."""
    root = project_root().parent / "Beam Tracing"
    if not (root / "configs" / "default.yaml").is_file():
        raise FileNotFoundError(
            f"Beam Tracing repo not found at {root}. "
            "Place Beam Surrogate next to the Beam Tracing directory."
        )
    return root


def default_run_config() -> Path:
    return beam_tracing_root() / "configs" / "default.yaml"
