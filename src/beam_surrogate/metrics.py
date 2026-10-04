"""Physical validation metrics from the course notebook."""

from __future__ import annotations

import numpy as np

from beam_surrogate.dataset import Norms


def physical_errors(preds: np.ndarray, trues: np.ndarray, inputs: np.ndarray, norms: Norms) -> dict[str, float]:
    """Relative position error ``|Δr| / r_rms`` after undoing the fixed scaling."""
    pred_dx = preds[:, 0] * norms.std_out
    pred_dy = preds[:, 1] * norms.std_out
    true_dx = trues[:, 0] * norms.std_out
    true_dy = trues[:, 1] * norms.std_out

    delta_r = np.sqrt((pred_dx - true_dx) ** 2 + (pred_dy - true_dy) ** 2)

    x0 = inputs[:, 0] * norms.std_in
    y0 = inputs[:, 1] * norms.std_in
    xp0 = inputs[:, 2] * norms.std_in
    yp0 = inputs[:, 3] * norms.std_in
    x1_true = x0 + norms.l_drift * xp0 + true_dx
    y1_true = y0 + norms.l_drift * yp0 + true_dy
    r_true = np.sqrt(x1_true**2 + y1_true**2)
    r_rms = float(np.sqrt(np.mean(r_true**2)))
    rel_err = delta_r / r_rms if r_rms > 0 else delta_r

    return {
        "mean": float(np.mean(rel_err)),
        "median": float(np.median(rel_err)),
        "p95": float(np.percentile(rel_err, 95)),
        "r_rms_mm": r_rms * 1000.0,
    }
