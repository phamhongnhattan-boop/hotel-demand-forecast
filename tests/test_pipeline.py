import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_prep import build_daily, load_raw  # noqa: E402
from features import make_features, naive_lag, split  # noqa: E402
from pricing import multiplier  # noqa: E402
from sample_data import make_sample  # noqa: E402


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "hotel_bookings.csv"
    make_sample(n_bookings=8_000).to_csv(path, index=False)
    return load_raw(path)


def test_one_booking_counts_every_night():
    df = pd.DataFrame({
        "hotel": ["City Hotel"], "is_canceled": [0],
        "arrival_date": [pd.Timestamp("2016-01-01")], "total_nights": [3], "adr": [100.0],
    })
    daily = build_daily(df, "City Hotel", "2016-01-05")
    assert daily["rooms_sold"].tolist() == [1, 1, 1, 0, 0]
    assert daily["revenue"].sum() == 300


def test_daily_has_no_gaps(raw):
    daily = build_daily(raw, "City Hotel", "2017-08-31")
    assert daily.index.is_monotonic_increasing
    assert (daily.index.to_series().diff().dropna() == pd.Timedelta("1D")).all()


def test_lags_do_not_leak(raw):
    horizon = 14
    daily = build_daily(raw, "City Hotel", "2017-08-31")
    X = make_features(daily, horizon)
    lag_cols = [c for c in X.columns if c.startswith("lag_")]
    assert min(int(c.split("_")[1]) for c in lag_cols) >= horizon


def test_naive_lag_is_same_weekday():
    assert naive_lag(7) == 7
    assert naive_lag(10) == 14
    assert naive_lag(14) == 14


def test_split_keeps_rows_with_missing_long_lags(raw):
    daily = build_daily(raw, "City Hotel", "2017-08-31")
    X = make_features(daily, 14)
    (X_tr, _), _, (X_te, _) = split(X, 14, "2015-08-01", "2017-03-01", "2017-06-01")
    assert X_tr["lag_364"].isna().any()      # first year has no lag_364, rows are kept
    assert X_tr["lag_14"].notna().all()      # the naive base is always present
    assert X_te.index.min() == pd.Timestamp("2017-06-01")


def test_arima_forecasts_every_test_day(raw):
    pytest.importorskip("statsforecast")
    from arima import arima_forecast

    daily = build_daily(raw, "Resort Hotel", "2017-08-31")
    pred = arima_forecast(daily, 14, "2015-08-01", "2017-08-01", "2017-08-31")
    assert len(pred) == 31
    assert pred.index.min() == pd.Timestamp("2017-08-01")
    assert (pred >= 0).all()


def test_price_multiplier_order():
    assert multiplier(0.95) > multiplier(0.8) > multiplier(0.6) > multiplier(0.2)
