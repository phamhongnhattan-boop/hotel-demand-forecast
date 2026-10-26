"""Make a small fake dataset with the same columns as hotel_bookings.csv.

Use it to test the pipeline before downloading the real data from Kaggle.
    python src/sample_data.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


def make_sample(n_bookings: int = 40_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2015-07-01", "2017-08-31", freq="D")
    # More arrivals in summer and on Fridays/Saturdays
    season = 1 + 0.5 * np.sin(2 * np.pi * (days.dayofyear.to_numpy() - 100) / 365)
    weekday = np.where(days.dayofweek.isin([4, 5]), 1.3, 1.0)
    p = np.asarray(season * weekday, dtype=float)
    arrivals = pd.DatetimeIndex(rng.choice(days, size=n_bookings, p=p / p.sum()))

    weekend = rng.integers(0, 3, n_bookings)
    week = rng.integers(0, 5, n_bookings)
    base = 80 + 40 * np.sin(2 * np.pi * (arrivals.dayofyear - 100) / 365)
    return pd.DataFrame({
        "hotel": rng.choice(["City Hotel", "Resort Hotel"], n_bookings, p=[0.65, 0.35]),
        "is_canceled": (rng.random(n_bookings) < 0.3).astype(int),
        "lead_time": rng.integers(0, 300, n_bookings),
        "arrival_date_year": arrivals.year,
        "arrival_date_month": [MONTH_NAMES[m - 1] for m in arrivals.month],
        "arrival_date_week_number": arrivals.isocalendar().week.to_numpy(),
        "arrival_date_day_of_month": arrivals.day,
        "stays_in_weekend_nights": weekend,
        "stays_in_week_nights": week,
        "adults": rng.integers(1, 4, n_bookings),
        "adr": np.round(base + rng.normal(0, 15, n_bookings), 2),
    })


if __name__ == "__main__":
    from config import RAW_CSV

    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    make_sample().to_csv(RAW_CSV, index=False)
    print(f"Fake data saved to {RAW_CSV}. Replace it with the real Kaggle file later.")
