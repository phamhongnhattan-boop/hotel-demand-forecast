"""Rule-based price suggestion from the demand forecast.

Two versions:
- suggest_prices():  v1, uses the point forecast only (kept to compare).
- decide_prices():   v2, uses the 80% interval [P10, P90] so the hotel only raises
                     prices when it is confident the date will be nearly full.

These are rules, not a trained pricing model: the dataset has no price elasticity
(we never see how many guests a higher price would lose). What we CAN test is
whether each decision was right, using the occupancy that actually happened.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------- v1: point forecast ----------
# (minimum occupancy, price multiplier). Checked from top to bottom.
RULES = [
    (0.90, 1.15),
    (0.75, 1.05),
    (0.50, 1.00),
    (0.00, 0.90),
]

FULL = 0.90   # a date counts as "nearly full" from 90% occupancy
LOW = 0.50    # and as "low demand" below 50%

ACTIONS = {"raise": 1.15, "small_raise": 1.05, "keep": 1.00, "discount": 0.90}


def estimate_capacity(rooms_sold: pd.Series, q: float = 0.95) -> float:
    """The dataset has no room count, so use a high percentile of rooms sold."""
    return float(np.quantile(rooms_sold.dropna(), q))


def multiplier(occupancy: float) -> float:
    for threshold, mult in RULES:
        if occupancy >= threshold:
            return mult
    return RULES[-1][1]


def suggest_prices(forecast: pd.Series, base_adr: pd.Series, capacity: float) -> pd.DataFrame:
    occ = (forecast / capacity).clip(0, 1.2)
    mult = occ.apply(multiplier)
    return pd.DataFrame({
        "forecast_rooms": forecast.round(1),
        "forecast_occupancy": occ.round(3),
        "base_adr": base_adr.round(2),
        "multiplier": mult,
        "suggested_adr": (base_adr * mult).round(2),
    })


def point_actions(p50: pd.Series, capacity: float) -> pd.Series:
    """v1 decisions written as actions, so v1 and v2 can be scored the same way."""
    occ = p50 / capacity
    return pd.Series(np.select(
        [occ >= FULL, occ >= 0.75, occ < LOW], ["raise", "small_raise", "discount"], "keep"),
        index=p50.index)


# ---------- v2: interval forecast ----------
def interval_actions(p10: pd.Series, p50: pd.Series, p90: pd.Series, capacity: float) -> pd.Series:
    """raise:       even the low estimate (P10) says nearly full -> +15%
    small_raise:  the point forecast says >= 75% full            -> +5%
    discount:     even the high estimate (P90) says < 50% full  -> -10%
    keep:         otherwise
    """
    lo, mid, hi = p10 / capacity, p50 / capacity, p90 / capacity
    return pd.Series(np.select(
        [lo >= FULL, mid >= 0.75, hi < LOW], ["raise", "small_raise", "discount"], "keep"),
        index=p50.index)


def decide_prices(p10, p50, p90, base_adr: pd.Series, capacity: float) -> pd.DataFrame:
    actions = interval_actions(p10, p50, p90, capacity)
    mult = actions.map(ACTIONS)
    return pd.DataFrame({
        "p10_rooms": p10.round(1), "p50_rooms": p50.round(1), "p90_rooms": p90.round(1),
        "action": actions, "multiplier": mult,
        "base_adr": base_adr.round(2), "suggested_adr": (base_adr * mult).round(2),
    })


def score_actions(actions: pd.Series, actual_rooms: pd.Series, capacity: float) -> dict:
    """Was each +15% raise / -10% discount right, given the occupancy that really happened?

    precision: of the days we raised (discounted), how many were really >= 90% (< 50%) full
    recall:    of the days that were really nearly full (low), how many we raised (discounted)
    wrong:     raises on days that were NOT nearly full = risk of losing guests
    """
    occ = actual_rooms / capacity
    full, low = occ >= FULL, occ < LOW
    out = {}
    for name, truth in (("raise", full), ("discount", low)):
        picked = actions == name
        n = int(picked.sum())
        out[name] = {
            "days": n,
            "precision_%": round(float(truth[picked].mean() * 100), 1) if n else None,
            "recall_%": round(float((picked & truth).sum() / truth.sum() * 100), 1) if truth.sum() else None,
            "wrong_days": int((picked & ~truth).sum()),
        }
    return out
