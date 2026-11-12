"""80% prediction intervals: quantile LightGBM (P10, P90) + conformal calibration.

Why calibration: raw quantile models are often over-confident. In our 12-month
backtest the raw [P10, P90] band held the actual value only ~65% of the time
instead of 80%. Conformalized quantile regression (CQR) widens the band by the
amount needed on the validation months, so the test coverage comes back to ~80%.
Reference: Romano, Patterson & Candes (2019), "Conformalized Quantile Regression".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models import QuantileResidualLGBM


def conformal_offset(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, coverage: float = 0.8) -> float:
    """How much to widen [lo, hi] so that it holds `coverage` of the calibration values.

    Score per point = how far the actual value is outside the band (negative when inside).
    The offset is the `coverage` quantile of the scores, with the usual (n + 1) / n correction.
    """
    lo, hi, y = (np.asarray(a, dtype=float) for a in (lo, hi, y))
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    if n == 0:
        return 0.0
    level = min(1.0, coverage * (1 + 1 / n))
    return float(np.quantile(scores, level))


class IntervalForecaster:
    """Fits P10 and P90 residual LightGBM models, then calibrates them on the validation data."""

    def __init__(self, horizon: int, coverage: float = 0.8, random_state: int = 42):
        tail = (1 - coverage) / 2
        self.coverage = coverage
        self.low = QuantileResidualLGBM(horizon, tail, random_state)
        self.high = QuantileResidualLGBM(horizon, 1 - tail, random_state)
        self.offset = 0.0

    def _raw(self, X) -> tuple[pd.Series, pd.Series]:
        a, b = self.low.predict(X), self.high.predict(X)
        return np.minimum(a, b), np.maximum(a, b)  # quantile models can cross: keep lo <= hi

    def fit(self, X_train, y_train, X_valid, y_valid):
        self.low.fit(X_train, y_train, X_valid, y_valid)
        self.high.fit(X_train, y_train, X_valid, y_valid)
        lo, hi = self._raw(X_valid)
        self.offset = conformal_offset(lo, hi, y_valid, self.coverage)
        return self

    def predict(self, X) -> pd.DataFrame:
        lo, hi = self._raw(X)
        return pd.DataFrame({
            "p10": (lo - self.offset).clip(lower=0),
            "p90": hi + self.offset,
        }, index=X.index)


def interval_metrics(y, p10, p90) -> dict:
    y, p10, p90 = (np.asarray(a, dtype=float) for a in (y, p10, p90))
    inside = (y >= p10) & (y <= p90)
    return {"coverage_%": round(float(inside.mean() * 100), 1),
            "mean_width_rooms": round(float((p90 - p10).mean()), 1)}
