"""Baseline and LightGBM forecasters."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import pandas as pd

from features import naive_lag

# New LightGBM versions warn about `eval_set`; it still works in all 4.x versions.
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")


def seasonal_naive(X: pd.DataFrame, horizon: int) -> pd.Series:
    """Predict with the value from the same weekday, at least `horizon` days earlier."""
    return X[f"lag_{naive_lag(horizon)}"]


class ResidualLGBM:
    """LightGBM that learns the CORRECTION to the seasonal naive forecast.

    prediction = naive + model(features)

    Why: with few training rows, early stopping can stop after a few trees.
    A model trained on the raw target then predicts close to the average
    (bad in high season). Learning the residual means a weak model falls back
    to the naive forecast instead of the average.
    """

    def __init__(self, horizon: int, random_state: int = 42):
        self.base = f"lag_{naive_lag(horizon)}"
        self.model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
        )

    def fit(self, X_train, y_train, X_valid, y_valid):
        self.model.fit(
            X_train, y_train - X_train[self.base],
            eval_set=[(X_valid, y_valid - X_valid[self.base])],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return self

    def predict(self, X) -> pd.Series:
        pred = X[self.base] + self.model.predict(X)
        return pd.Series(pred, index=X.index).clip(lower=0)

    @property
    def best_iteration(self) -> int:
        return self.model.best_iteration_

    @property
    def feature_importances(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.model.feature_name_)
