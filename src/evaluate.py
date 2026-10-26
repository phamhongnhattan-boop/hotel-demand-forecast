"""Forecast metrics."""
from __future__ import annotations

import numpy as np


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    nonzero = y_true != 0
    return {
        "MAE": round(float(np.mean(np.abs(err))), 3),
        "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 3),
        "MAPE_%": round(float(np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100), 2),
    }
