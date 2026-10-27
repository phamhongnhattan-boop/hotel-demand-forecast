"""AutoARIMA baseline with statsforecast (local alternative to BigQuery ML ARIMA_PLUS).

Fair comparison with LightGBM: every test day is forecast exactly `horizon` days
ahead (rolling origin, step = 1 day). The model is fitted once on data before the
first forecast origin, then updated with new observations (no refit).
"""
from __future__ import annotations

import pandas as pd


def arima_forecast(daily: pd.DataFrame, horizon: int, train_start: str, test_start: str,
                   test_end: str, season_length: int = 7) -> pd.Series:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    d = daily[(daily.index >= train_start) & (daily.index <= test_end)]
    df = pd.DataFrame({"unique_id": "hotel", "ds": d.index, "y": d["rooms_sold"].astype(float).values})

    n_test_days = (pd.Timestamp(test_end) - pd.Timestamp(test_start)).days + 1
    sf = StatsForecast(models=[AutoARIMA(season_length=season_length)], freq="D", n_jobs=1)
    cv = sf.cross_validation(df=df, h=horizon, step_size=1, n_windows=n_test_days, refit=False)

    # keep only the forecast made exactly `horizon` days before each date
    cv = cv[(cv["ds"] - cv["cutoff"]).dt.days == horizon]
    cv = cv[cv["ds"] >= test_start]
    return pd.Series(cv["AutoARIMA"].clip(lower=0).to_numpy(), index=pd.DatetimeIndex(cv["ds"]), name="auto_arima")
