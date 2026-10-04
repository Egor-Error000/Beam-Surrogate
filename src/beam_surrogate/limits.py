"""Cap Numba, BLAS and PyTorch at a fixed number of logical CPUs.

Environment variables are set before those libraries are imported. If they
are already loaded, the thread pools are lowered in place. Batches stay
sequential so per-process thread pools are not multiplied.
"""

from __future__ import annotations

import os
import sys

_ENV_KEYS = (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)

_applied_interop = False


def resolve_cpu_limit(max_cpus: int) -> int:
    """Return how many CPUs this process may use."""
    if max_cpus < 1:
        raise ValueError(f"max_cpus must be >= 1, got {max_cpus}")
    machine = os.cpu_count() or 1
    return max(1, min(int(max_cpus), machine))


def apply_cpu_limit(max_cpus: int = 12) -> int:
    """Restrict this process to ``max_cpus`` logical CPUs (default 12)."""
    n = resolve_cpu_limit(max_cpus)
    for key in _ENV_KEYS:
        os.environ[key] = str(n)
    _tighten_imported_runtimes(n)
    return n


def _tighten_imported_runtimes(n: int) -> None:
    if "numba" in sys.modules:
        import numba

        numba.set_num_threads(n)
    if "torch" in sys.modules:
        import torch

        torch.set_num_threads(n)
        global _applied_interop
        if not _applied_interop:
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            _applied_interop = True
