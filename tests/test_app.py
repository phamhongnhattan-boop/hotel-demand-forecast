"""Checks for the demo's input validation and the saved model (no Streamlit needed)."""
import json
import sys
from datetime import date
from pathlib import Path

import lightgbm as lgb
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from validation import InvalidInput, booking_row, check_base_price, report_slug  # noqa: E402

META = json.loads((ROOT / "models" / "cancel_meta.json").read_text(encoding="utf-8"))


def good_form(**change):
    form = {"hotel": "City Hotel", "arrival_date": date(2017, 7, 15), "lead_time": 60,
            "stays_in_weekend_nights": 1, "stays_in_week_nights": 2, "adults": 2, "children": 0,
            "babies": 0, "adr": 100.0, "deposit_type": "No Deposit", "market_segment": "Online TA",
            "distribution_channel": "TA/TO", "customer_type": "Transient", "meal": "BB",
            "reserved_room_type": "A", "is_repeated_guest": 0, "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0, "days_in_waiting_list": 0, "agent": "none", "company": "none"}
    form.update(change)
    return form


def test_hotel_name_cannot_build_a_path():
    assert report_slug("City Hotel") == "city_hotel"
    for bad in ["../../etc/passwd", "city_hotel", "", "City Hotel\x00"]:
        with pytest.raises(InvalidInput):
            report_slug(bad)


@pytest.mark.parametrize("value", [-5, 0, 5000, float("nan"), float("inf"), "abc", None])
def test_bad_base_price_is_refused(value):
    with pytest.raises(InvalidInput):
        check_base_price(value)


@pytest.mark.parametrize("change", [
    {"deposit_type": "<script>alert(1)</script>"},   # value never seen in training
    {"hotel": "Hilton"},
    {"lead_time": -1}, {"lead_time": 10_000}, {"adr": float("nan")},
    {"stays_in_weekend_nights": 0, "stays_in_week_nights": 0},
    {"adults": 0, "children": 0},
    {"arrival_date": "2017-07-15"},                   # must be a real date
])
def test_bad_booking_is_refused(change):
    with pytest.raises(InvalidInput):
        booking_row(good_form(**change), META)


def test_saved_model_gives_a_probability():
    booster = lgb.Booster(model_file=str(ROOT / "models" / "cancel_lgbm.txt"))
    p_free = booster.predict(booking_row(good_form(), META))[0]
    p_nonrefund = booster.predict(booking_row(good_form(deposit_type="Non Refund"), META))[0]
    assert 0 <= p_free <= 1 and 0 <= p_nonrefund <= 1
    assert p_nonrefund > p_free   # non-refundable deposits almost always cancel in this data


def test_demo_runs_without_errors():
    """Start the real Streamlit app (no browser) and click through it."""
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    at.radio[0].set_value("Resort Hotel").run()
    assert not at.exception
    at.button[0].click().run()   # "Predict" in the cancellation tab
    assert not at.exception
    assert any(m.label == "Cancel probability" for m in at.metric)
