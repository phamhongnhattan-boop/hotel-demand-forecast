"""Turn booking rows into a daily time series of room-nights sold per hotel."""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["arrival_date"] = pd.to_datetime(dict(
        year=df["arrival_date_year"],
        month=df["arrival_date_month"].map(MONTHS),
        day=df["arrival_date_day_of_month"],
    ))
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep real stays only: not canceled, at least 1 night, ADR in a sane range."""
    out = df[(df["is_canceled"] == 0) & (df["total_nights"] > 0)]
    out = out[(out["adr"] > 0) & (out["adr"] < 1000)]
    return out.copy()


def build_daily(df: pd.DataFrame, hotel: str, end_date: str | None = None) -> pd.DataFrame:
    """One row per date: rooms_sold, revenue, adr.

    Each booking counts as 1 room for every night between arrival and departure.
    """
    d = clean(df[df["hotel"] == hotel])
    nights = d["total_nights"].to_numpy()
    start = np.repeat(d["arrival_date"].to_numpy(), nights)
    offset = np.concatenate([np.arange(n) for n in nights]) if len(nights) else np.array([], int)
    stay_dates = start + offset.astype("timedelta64[D]")
    adr = np.repeat(d["adr"].to_numpy(), nights)

    nightly = pd.DataFrame({"date": stay_dates, "adr": adr})
    daily = nightly.groupby("date").agg(rooms_sold=("adr", "size"), revenue=("adr", "sum"))

    last = pd.Timestamp(end_date) if end_date else d["arrival_date"].max()
    full_range = pd.date_range(daily.index.min(), last, freq="D")
    daily = daily.reindex(full_range, fill_value=0)
    daily.index.name = "date"
    daily["adr"] = (daily["revenue"] / daily["rooms_sold"].replace(0, np.nan)).round(2)
    daily.insert(0, "hotel", hotel)
    return daily


def build_on_the_books(df: pd.DataFrame, hotel: str, horizon: int) -> pd.Series:
    """Rooms already booked for each night, as known `horizon` days before that night.

    This is what a hotel's PMS shows at forecast time ("on the books").
    For a stay night D, the cutoff is D - horizon. A room-night counts if:
      - the booking was made on or before the cutoff (booking date = arrival - lead_time), and
      - it was not canceled by the cutoff (canceled bookings count until their cancel date).
    Bookings canceled AFTER the cutoff still count: at forecast time we did not know yet.
    Only information known at the cutoff is used, so there is no data leakage.
    """
    d = df[(df["hotel"] == hotel) & (df["total_nights"] > 0)]
    d = d[(d["adr"] > 0) & (d["adr"] < 1000)]  # same filter as clean(), but keep canceled rows
    nights = d["total_nights"].to_numpy()
    if len(nights) == 0:
        return pd.Series(dtype=float, name=f"otb_{horizon}")

    offset = np.concatenate([np.arange(n) for n in nights]).astype("timedelta64[D]")
    stay = np.repeat(d["arrival_date"].to_numpy(), nights) + offset
    booked_on = np.repeat(
        (d["arrival_date"] - pd.to_timedelta(d["lead_time"], unit="D")).to_numpy(), nights)
    canceled = np.repeat(d["is_canceled"].to_numpy(), nights) == 1
    # For canceled bookings, reservation_status_date is the cancel date
    status_date = np.repeat(pd.to_datetime(d["reservation_status_date"]).to_numpy(), nights)

    cutoff = stay - np.timedelta64(horizon, "D")
    on_books = (booked_on <= cutoff) & (~canceled | (status_date > cutoff))
    otb = pd.Series(on_books.astype(int), index=pd.DatetimeIndex(stay)).groupby(level=0).sum()
    otb.index.name = "date"
    return otb.rename(f"otb_{horizon}")


if __name__ == "__main__":
    from config import RAW_CSV, PROCESSED_DIR, DATA_END

    raw = load_raw(RAW_CSV)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames = [build_daily(raw, h, DATA_END) for h in raw["hotel"].unique()]
    out = pd.concat(frames).reset_index()
    out.to_csv(PROCESSED_DIR / "daily_demand.csv", index=False)
    print(out.groupby("hotel")["rooms_sold"].describe())
