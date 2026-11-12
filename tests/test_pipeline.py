import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cancellation as canc  # noqa: E402
from data_prep import build_daily, build_on_the_books, load_raw  # noqa: E402
from features import make_features, naive_lag, split  # noqa: E402
from intervals import IntervalForecaster, conformal_offset, interval_metrics  # noqa: E402
from pricing import interval_actions, multiplier, score_actions  # noqa: E402
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


# ---------- Intervals + tested pricing ----------
def test_conformal_offset_reaches_target_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 2000)
    lo, hi = np.full(2000, -0.5), np.full(2000, 0.5)   # far too narrow: ~38% coverage
    off = conformal_offset(lo, hi, y, 0.8)
    assert off > 0
    cov = interval_metrics(y, lo - off, hi + off)["coverage_%"]
    assert 79 <= cov <= 82


def test_interval_forecaster_gives_ordered_band(raw):
    daily = build_daily(raw, "City Hotel", "2017-08-31")
    X = make_features(daily, 14).dropna(subset=["lag_14"])
    cols = [c for c in X.columns if c != "rooms_sold"]
    tr, va, te = X[:"2016-12-31"], X["2017-01-01":"2017-02-28"], X["2017-03-01":]
    band = IntervalForecaster(14, 0.8).fit(tr[cols], tr["rooms_sold"], va[cols], va["rooms_sold"]).predict(te[cols])
    assert (band["p10"] <= band["p90"]).all() and (band["p10"] >= 0).all()


def test_interval_actions_need_confidence():
    idx = pd.RangeIndex(4)
    p10 = pd.Series([95, 70, 10, 40], index=idx)
    p50 = pd.Series([98, 92, 30, 60], index=idx)
    p90 = pd.Series([100, 99, 45, 80], index=idx)
    acts = interval_actions(p10, p50, p90, capacity=100).tolist()
    # day 1: point forecast says 92% full, but P10 is only 70% -> small raise, not +15%
    assert acts == ["raise", "small_raise", "discount", "keep"]


def test_score_actions_counts():
    actions = pd.Series(["raise", "raise", "keep", "discount", "keep"])
    actual = pd.Series([95, 80, 92, 30, 40])          # rooms, capacity 100
    s = score_actions(actions, actual, 100)
    assert s["raise"] == {"days": 2, "precision_%": 50.0, "recall_%": 50.0, "wrong_days": 1}
    assert s["discount"] == {"days": 1, "precision_%": 100.0, "recall_%": 50.0, "wrong_days": 0}


# ---------- Cancellation prediction ----------
def _bookings(n=3000, seed=0):
    """Small fake booking table with every column the cancellation model uses."""
    rng = np.random.default_rng(seed)
    arrival = pd.Timestamp("2016-01-01") + pd.to_timedelta(rng.integers(0, 600, n), unit="D")
    deposit = rng.choice(["No Deposit", "Non Refund"], n, p=[0.85, 0.15])
    lead = rng.integers(0, 300, n)
    p = np.where(deposit == "Non Refund", 0.95, 0.1 + lead / 600)
    df = pd.DataFrame({
        "hotel": rng.choice(["City Hotel", "Resort Hotel"], n), "is_canceled": (rng.random(n) < p).astype(int),
        "arrival_date": arrival, "lead_time": lead, "deposit_type": deposit,
        "stays_in_weekend_nights": rng.integers(0, 3, n), "stays_in_week_nights": rng.integers(1, 5, n),
        "adults": 2, "children": 0.0, "babies": 0, "meal": "BB", "market_segment": "Online TA",
        "distribution_channel": "TA/TO", "reserved_room_type": "A", "assigned_room_type": "A",
        "customer_type": "Transient", "agent": rng.choice([9.0, 240.0, np.nan], n), "company": np.nan,
        "is_repeated_guest": 0, "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
        "days_in_waiting_list": 0, "adr": rng.uniform(50, 200, n), "country": "PRT",
        "required_car_parking_spaces": 0, "booking_changes": 0, "total_of_special_requests": 0,
    })
    return canc.prepare(df)


def test_cancellation_uses_no_leaky_columns():
    leaky = set(canc.LEAKY_EXTRA) | {"reservation_status", "reservation_status_date", "is_canceled"}
    assert not leaky & set(canc.SAFE_FEATURES)


def test_cancellation_split_is_by_arrival_time():
    train, valid, test = canc.split(_bookings())
    assert train["arrival_date"].max() < valid["arrival_date"].min()
    assert valid["arrival_date"].max() < test["arrival_date"].min()


def test_cancellation_model_beats_coin_flip():
    train, valid, test = canc.split(_bookings(6000))
    predict, _ = canc.fit_lgbm(train, valid, canc.SAFE_FEATURES)
    p = predict(test)
    assert ((p >= 0) & (p <= 1)).all()
    assert canc.classification_metrics(test["is_canceled"], p)["roc_auc"] > 0.7


def test_expected_cancellations_sum_probabilities():
    d = pd.DataFrame({"hotel": ["City Hotel"] * 3, "arrival_date": pd.to_datetime(["2017-03-01"] * 2 + ["2017-03-02"]),
                      "is_canceled": [1, 0, 1]})
    e = canc.expected_cancellations(d, np.array([0.5, 0.25, 0.9]))
    assert e["predicted"].round(2).tolist() == [0.75, 0.9]
    assert e["actual"].tolist() == [1, 1]
