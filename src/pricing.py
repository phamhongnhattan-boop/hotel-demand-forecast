"""Simple rule-based price suggestion from the demand forecast.

This is a heuristic, not a trained pricing model. It is easy to explain in an
interview, and a good starting point before trying price elasticity models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (minimum occupancy, price multiplier). Checked from top to bottom.
RULES = [
    (0.90, 1.15),
    (0.75, 1.05),
    (0.50, 1.00),
    (0.00, 0.90),
]


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
