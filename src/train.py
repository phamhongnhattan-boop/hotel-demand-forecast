"""Full pipeline: data -> features -> baseline vs LightGBM -> price suggestion.

    python src/train.py --hotel "City Hotel" --horizon 14
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from data_prep import build_daily, load_raw
from evaluate import metrics
from features import make_features, split
from models import seasonal_naive, train_lgbm
from pricing import estimate_capacity, suggest_prices


def base_adr_table(daily: pd.DataFrame, before: str) -> pd.Series:
    """Average ADR by (month, weekday) from training data only."""
    hist = daily[(daily.index < before) & (daily["rooms_sold"] > 0)]
    return hist.groupby([hist.index.month, hist.index.dayofweek])["adr"].mean()


def run(hotel: str, horizon: int, data_path=config.RAW_CSV) -> dict:
    raw = load_raw(data_path)
    daily = build_daily(raw, hotel, config.DATA_END)

    X = make_features(daily, horizon)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split(X, config.VALID_START, config.TEST_START)

    # 1) Baseline
    naive = seasonal_naive(daily["rooms_sold"], X_te.index, horizon)
    # 2) LightGBM with early stopping on the validation period
    model = train_lgbm(X_tr, y_tr, X_va, y_va, config.RANDOM_STATE)
    gbm_pred = pd.Series(model.predict(X_te), index=X_te.index).clip(lower=0)

    results = {
        "hotel": hotel,
        "horizon_days": horizon,
        "test_period": f"{X_te.index.min().date()} -> {X_te.index.max().date()}",
        "seasonal_naive": metrics(y_te, naive),
        "lightgbm": metrics(y_te, gbm_pred),
    }

    # 3) Price suggestion on the test period
    capacity = estimate_capacity(y_tr)
    adr_table = base_adr_table(daily, config.TEST_START)
    base_adr = pd.Series(
        [adr_table.get((d.month, d.dayofweek), adr_table.mean()) for d in X_te.index],
        index=X_te.index,
    )
    prices = suggest_prices(gbm_pred, base_adr, capacity)
    prices.insert(0, "actual_rooms", y_te)

    # 4) Save outputs
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = hotel.lower().replace(" ", "_")
    prices.to_csv(config.REPORTS_DIR / f"forecast_{slug}.csv")
    with open(config.REPORTS_DIR / f"metrics_{slug}.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_te.index, y_te, label="Actual", lw=2)
    ax.plot(naive.index, naive, label="Seasonal naive", alpha=0.7)
    ax.plot(gbm_pred.index, gbm_pred, label="LightGBM", alpha=0.9)
    ax.set_title(f"{hotel} - rooms sold, {horizon}-day-ahead forecast")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / f"forecast_{slug}.png", dpi=120)

    importance = pd.Series(model.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    results["top_features"] = importance.head(5).to_dict()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotel", default=config.DEFAULT_HOTEL)
    parser.add_argument("--horizon", type=int, default=config.HORIZON)
    parser.add_argument("--data", default=str(config.RAW_CSV))
    args = parser.parse_args()
    print(json.dumps(run(args.hotel, args.horizon, args.data), indent=2, default=int))
