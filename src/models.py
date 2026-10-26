"""Baseline and LightGBM forecasters."""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd


def seasonal_naive(series: pd.Series, index: pd.Index, horizon: int) -> pd.Series:
    """Predict with the value from the same weekday, at least `horizon` days earlier."""
    lag = -(-horizon // 7) * 7
    return series.shift(lag).reindex(index)


def train_lgbm(X_train, y_train, X_valid, y_valid, random_state: int = 42) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
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
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model
