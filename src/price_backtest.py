"""Test the price suggestions over a full year (same 12 monthly folds as backtest.py).

For each test month: point forecast (P50) + calibrated 80% interval (P10, P90),
then two pricing rules, scored against the occupancy that really happened:
  v1: raise when the point forecast says >= 90% full
  v2: raise only when the LOW estimate (P10) says >= 90% full

    python src/price_backtest.py --hotel "City Hotel"
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from backtest import monthly_folds
from data_prep import build_daily, build_on_the_books, load_raw
from evaluate import metrics
from features import make_features, naive_lag
from intervals import IntervalForecaster, interval_metrics
from models import ResidualLGBM
from pricing import estimate_capacity, interval_actions, point_actions, score_actions


def run_price_backtest(hotel: str, horizon: int, data_path=config.RAW_CSV):
    raw = load_raw(data_path)
    daily = build_daily(raw, hotel, config.DATA_END)
    otb = build_on_the_books(raw, hotel, horizon)
    X = make_features(daily, horizon, on_the_books=otb)
    base = f"lag_{naive_lag(horizon)}"
    X = X[(X.index >= config.TRAIN_START) & X[base].notna()]
    target = "rooms_sold"
    cols = [c for c in X.columns if c != target]

    parts = []
    for valid_start, test_start, test_end in monthly_folds():
        train = X[X.index < valid_start]
        valid = X[(X.index >= valid_start) & (X.index < test_start)]
        test = X[(X.index >= test_start) & (X.index <= test_end)]
        # Capacity from past data only (the dataset has no room count)
        capacity = estimate_capacity(train[target])

        p50 = ResidualLGBM(horizon, config.RANDOM_STATE).fit(
            train[cols], train[target], valid[cols], valid[target]).predict(test[cols])
        band = IntervalForecaster(horizon, 0.8, config.RANDOM_STATE).fit(
            train[cols], train[target], valid[cols], valid[target]).predict(test[cols])
        band["p50"] = p50.clip(band["p10"], band["p90"])  # keep P10 <= P50 <= P90
        part = band.assign(actual=test[target], capacity=capacity)
        part["v1_action"] = point_actions(part["p50"], capacity)
        part["v2_action"] = interval_actions(part["p10"], part["p50"], part["p90"], capacity)
        parts.append(part)

    days = pd.concat(parts)
    cap = days["capacity"]
    summary = {
        "hotel": hotel,
        "horizon_days": horizon,
        "test_period": f"{days.index.min().date()} -> {days.index.max().date()}",
        "point_forecast": metrics(days["actual"], days["p50"]),
        "interval_80": interval_metrics(days["actual"], days["p10"], days["p90"]),
        "pricing_v1_point": score_actions(days["v1_action"], days["actual"] / cap, 1.0),
        "pricing_v2_interval": score_actions(days["v2_action"], days["actual"] / cap, 1.0),
    }
    return days, summary


def save_outputs(hotel: str, days: pd.DataFrame, summary: dict) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = hotel.lower().replace(" ", "_")
    days.round(2).to_csv(config.REPORTS_DIR / f"pricing_{slug}.csv")
    with open(config.REPORTS_DIR / f"pricing_{slug}.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(days.index, days["p10"], days["p90"], color="tab:green", alpha=0.2,
                    label=f"80% interval (coverage {summary['interval_80']['coverage_%']}%)")
    ax.plot(days.index, days["p50"], color="tab:green", lw=1, label="Forecast (P50)")
    ax.plot(days.index, days["actual"], color="black", lw=1, label="Actual")
    ax.plot(days.index, days["capacity"] * 0.9, color="tab:red", ls="--", lw=1, label="90% of capacity")
    raised = days[days["v2_action"] == "raise"]
    ax.scatter(raised.index, raised["actual"], s=8, color="tab:red", zorder=3, label="v2: raise +15%")
    ax.set_title(f"{hotel}: rooms sold, forecast 14 days ahead, Sep 2016 - Aug 2017")
    ax.set_ylabel("rooms")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / f"pricing_{slug}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotel", default=config.DEFAULT_HOTEL)
    parser.add_argument("--horizon", type=int, default=config.HORIZON)
    parser.add_argument("--data", default=str(config.RAW_CSV))
    args = parser.parse_args()

    days, summary = run_price_backtest(args.hotel, args.horizon, args.data)
    save_outputs(args.hotel, days, summary)
    print(json.dumps(summary, indent=2))
