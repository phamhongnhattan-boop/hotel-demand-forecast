import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_prep import build_daily, build_on_the_books, load_raw  # noqa: E402
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


def _booking(lead_time, canceled=0, status_date="2016-01-12", arrival="2016-01-10", nights=2):
    return {"hotel": "City Hotel", "arrival_date": pd.Timestamp(arrival), "total_nights": nights,
            "adr": 100.0, "lead_time": lead_time, "is_canceled": canceled,
            "reservation_status_date": status_date}


def test_on_the_books_uses_only_info_known_at_cutoff():
    # Night 2016-01-10, horizon 14 -> cutoff 2015-12-27
    df = pd.DataFrame([
        _booking(lead_time=30),                                    # booked 2015-12-11: counts
        _booking(lead_time=5),                                     # booked after cutoff: no
        _booking(lead_time=30, canceled=1, status_date="2015-12-20"),  # canceled before cutoff: no
        _booking(lead_time=30, canceled=1, status_date="2016-01-05"),  # canceled after cutoff: counts
    ])
    otb = build_on_the_books(df, "City Hotel", horizon=14)
    assert otb.loc["2016-01-10"] == 2
    assert otb.loc["2016-01-11"] == 2   # second night of the same stays


def test_backtest_folds_do_not_overlap():
    from backtest import monthly_folds
    folds = monthly_folds("2016-09-01", "2017-08-31")
    assert len(folds) == 12
    for valid_start, test_start, test_end in folds:
        assert valid_start < test_start <= test_end
    # each test month starts the day after the previous one ends
    for (_, _, prev_end), (_, next_start, _) in zip(folds, folds[1:]):
        assert next_start == prev_end + pd.Timedelta("1D")


def test_features_include_on_the_books(raw):
    daily = build_daily(raw, "City Hotel", "2017-08-31")
    otb = build_on_the_books(raw, "City Hotel", 14)
    X = make_features(daily, 14, on_the_books=otb)
    assert "otb_14" in X and "otb_ratio_ly" in X
    assert X["otb_14"].notna().all()


def test_price_multiplier_order():
    assert multiplier(0.95) > multiplier(0.8) > multiplier(0.6) > multiplier(0.2)
