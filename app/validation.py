"""Input checks for the demo. Kept apart from the UI so they can be unit tested.

The Streamlit widgets already limit what a user can pick, but a request can be sent
without the UI, so every value is checked again here before it reaches the model:
- categories must be one of the values seen in training (no free text),
- numbers must be finite and inside the training range,
- hotel names come from a fixed list, so no user text is ever used to build a file path.
"""
from __future__ import annotations

import math
from datetime import date

import pandas as pd

HOTELS = {"City Hotel": "city_hotel", "Resort Hotel": "resort_hotel"}
BASE_PRICE_RANGE = (20.0, 1000.0)   # EUR per night, same range as the cleaned data
MAX_LEAD_DAYS = 737                 # longest lead time in the data


class InvalidInput(ValueError):
    pass


def report_slug(hotel: str) -> str:
    """Map a hotel name to its report file name. Anything else is refused."""
    if hotel not in HOTELS:
        raise InvalidInput(f"unknown hotel: {hotel!r}")
    return HOTELS[hotel]


def check_number(name: str, value, low: float, high: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise InvalidInput(f"{name} must be a number") from None
    if not math.isfinite(v) or not low <= v <= high:
        raise InvalidInput(f"{name} must be between {low:g} and {high:g}")
    return v


def check_base_price(value) -> float:
    return check_number("base price", value, *BASE_PRICE_RANGE)


def booking_row(form: dict, meta: dict) -> pd.DataFrame:
    """Turn the form values into one model row, or raise InvalidInput."""
    arrival = form.get("arrival_date")
    if not isinstance(arrival, date):
        raise InvalidInput("arrival date is missing")

    row = {}
    for col, allowed in meta["categories"].items():
        v = str(form.get(col, "none"))
        if v not in allowed:
            raise InvalidInput(f"{col}: value not allowed")
        row[col] = v

    row["arrival_month"] = arrival.month
    row["arrival_weekday"] = arrival.weekday()
    row["arrival_week"] = arrival.isocalendar()[1]
    for col, (low, high) in meta["numeric_ranges"].items():
        if col.startswith("arrival_"):
            continue
        if col == "lead_time":
            high = max(high, MAX_LEAD_DAYS)
        row[col] = check_number(col, form.get(col, 0), low, high)

    if row["stays_in_weekend_nights"] + row["stays_in_week_nights"] < 1:
        raise InvalidInput("the stay must be at least 1 night")
    if row["adults"] + row["children"] < 1:
        raise InvalidInput("at least 1 guest")

    df = pd.DataFrame([row])[meta["features"]]
    for col, allowed in meta["categories"].items():
        df[col] = pd.Categorical(df[col], categories=allowed)
    return df
