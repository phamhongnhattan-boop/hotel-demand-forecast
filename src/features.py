"""Calendar + lag features. Every lag is >= horizon, so there is no data leakage."""
from __future__ import annotations

import pandas as pd


def naive_lag(horizon: int) -> int:
    """Smallest multiple of 7 that is >= horizon (same weekday, already known at forecast time)."""
    return -(-horizon // 7) * 7


def make_features(daily: pd.DataFrame, horizon: int, target: str = "rooms_sold",
                  on_the_books: pd.Series | None = None) -> pd.DataFrame:
    """Build the feature table.

    on_the_books: rooms already booked for each date, known `horizon` days before
    (see data_prep.build_on_the_books). Optional, so the old model can still be compared.
    """
    y = daily[target]
    X = pd.DataFrame(index=daily.index)

    # Booking pace: what the hotel already has on the books at forecast time
    if on_the_books is not None:
        otb = on_the_books.reindex(daily.index).fillna(0)
        X[f"otb_{horizon}"] = otb
        # Same idea one year earlier: how much of last year's demand was booked this early
        otb_ly, y_ly = otb.shift(364), y.shift(364)
        X["otb_ratio_ly"] = (otb_ly / y_ly.replace(0, float("nan")))

    # Calendar
    idx = daily.index
    X["dow"] = idx.dayofweek
    X["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    X["month"] = idx.month
    X["dayofyear"] = idx.dayofyear
    X["weekofyear"] = idx.isocalendar().week.astype(int).to_numpy()

    # Lags: same weekday, at least `horizon` days back
    first_week_lag = naive_lag(horizon)
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


def split(X: pd.DataFrame, horizon: int, train_start: str, valid_start: str, test_start: str,
          target: str = "rooms_sold"):
    """Time-based split.

    Rows need the naive lag (it is the base of the forecast). Other missing lags
    (for example lag_364 in the first year) are kept as NaN: LightGBM handles them.
    """
    base = f"lag_{naive_lag(horizon)}"
    X = X[(X.index >= train_start) & X[base].notna()]
    train = X[X.index < valid_start]
    valid = X[(X.index >= valid_start) & (X.index < test_start)]
    test = X[X.index >= test_start]
    cols = [c for c in X.columns if c != target]
    return (train[cols], train[target]), (valid[cols], valid[target]), (test[cols], test[target])
