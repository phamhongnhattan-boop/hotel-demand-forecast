"""Calendar + lag features. Every lag is >= horizon, so there is no data leakage."""
from __future__ import annotations

import pandas as pd


def make_features(daily: pd.DataFrame, horizon: int, target: str = "rooms_sold") -> pd.DataFrame:
    y = daily[target]
    X = pd.DataFrame(index=daily.index)

    # Calendar
    idx = daily.index
    X["dow"] = idx.dayofweek
    X["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    X["month"] = idx.month
    X["dayofyear"] = idx.dayofyear
    X["weekofyear"] = idx.isocalendar().week.astype(int).to_numpy()

    # Lags: same weekday, at least `horizon` days back
    first_week_lag = -(-horizon // 7) * 7  # round up to a multiple of 7
    for lag in (horizon, first_week_lag, first_week_lag + 7, first_week_lag + 14, 364):
        X[f"lag_{lag}"] = y.shift(lag)

    # Rolling stats that end `horizon` days before the target date
    shifted = y.shift(horizon)
    for w in (7, 28):
        X[f"roll_mean_{w}"] = shifted.rolling(w).mean()
    X["roll_std_28"] = shifted.rolling(28).std()

    X = X.loc[:, ~X.columns.duplicated()]
    X[target] = y
    return X


def split(X: pd.DataFrame, valid_start: str, test_start: str, target: str = "rooms_sold"):
    """Time-based split. Rows with missing lags (start of the series) are dropped from train."""
    train = X[X.index < valid_start].dropna()
    valid = X[(X.index >= valid_start) & (X.index < test_start)]
    test = X[X.index >= test_start]
    cols = [c for c in X.columns if c != target]
    return (train[cols], train[target]), (valid[cols], valid[target]), (test[cols], test[target])
