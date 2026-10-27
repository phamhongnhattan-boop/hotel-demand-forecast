"""Full pipeline: data -> features -> baseline vs AutoARIMA vs LightGBM -> price suggestion.

    python src/train.py --hotel "City Hotel" --horizon 14
    python src/train.py --hotel "City Hotel" --no-arima     # skip AutoARIMA (faster)
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
from models import ResidualLGBM, seasonal_naive
from pricing import estimate_capacity, suggest_prices


def base_adr_table(daily: pd.DataFrame, before: str) -> pd.Series:
    """Average ADR by (month, weekday) from training data only."""
    hist = daily[(daily.index < before) & (daily["rooms_sold"] > 0)]
    return hist.groupby([hist.index.month, hist.index.dayofweek])["adr"].mean()


def run(hotel: str, horizon: int, data_path=config.RAW_CSV, use_arima: bool = True) -> dict:
    raw = load_raw(data_path)
    daily = build_daily(raw, hotel, config.DATA_END)

    X = make_features(daily, horizon)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split(
        X, horizon, config.TRAIN_START, config.VALID_START, config.TEST_START)

    # 1) Baseline
    naive = seasonal_naive(X_te, horizon)
    preds = {"seasonal_naive": naive}

    # 2) AutoARIMA (optional, needs statsforecast)
    if use_arima:
        try:
            from arima import arima_forecast
            preds["auto_arima"] = arima_forecast(
                daily, horizon, config.TRAIN_START, config.TEST_START, config.DATA_END).reindex(X_te.index)
        except ImportError:
            print("statsforecast is not installed -> AutoARIMA skipped (pip install statsforecast)")

    # 3) LightGBM on the residual (actual - naive), early stopping on the validation period
    model = ResidualLGBM(horizon, config.RANDOM_STATE).fit(X_tr, y_tr, X_va, y_va)
    gbm_pred = model.predict(X_te)
    preds["lightgbm"] = gbm_pred

    results = {
        "hotel": hotel,
        "horizon_days": horizon,
        "train_rows": len(X_tr),
        "test_period": f"{X_te.index.min().date()} -> {X_te.index.max().date()}",
        **{name: metrics(y_te, p) for name, p in preds.items()},
        "lightgbm_best_iteration": model.best_iteration,
    }
    base_mae = results["seasonal_naive"]["MAE"]
    results["mae_improvement_%"] = round((base_mae - results["lightgbm"]["MAE"]) / base_mae * 100, 1)

    # 4) Price suggestion on the test period (from the LightGBM forecast)
    capacity = estimate_capacity(y_tr)
    adr_table = base_adr_table(daily, config.TEST_START)
    base_adr = pd.Series(
        [adr_table.get((d.month, d.dayofweek), adr_table.mean()) for d in X_te.index],
        index=X_te.index,
    )
    prices = suggest_prices(gbm_pred, base_adr, capacity)
    prices.insert(0, "actual_rooms", y_te)
    for name, p in preds.items():
        if name != "lightgbm":
            prices.insert(1, f"{name}_rooms", p.round(1))

    # 5) Save outputs
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = hotel.lower().replace(" ", "_")
    prices.to_csv(config.REPORTS_DIR / f"forecast_{slug}.csv")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_te.index, y_te, label="Actual", lw=2, color="black")
    styles = {"seasonal_naive": ("Seasonal naive", "tab:orange"),
              "auto_arima": ("AutoARIMA", "tab:purple"),
              "lightgbm": ("LightGBM", "tab:green")}
    for name, p in preds.items():
        label, color = styles[name]
        ax.plot(p.index, p, label=f"{label} (MAE {results[name]['MAE']:.2f})", color=color, alpha=0.8)
    ax.set_title(f"{hotel} - rooms sold, {horizon}-day-ahead forecast")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / f"forecast_{slug}.png", dpi=120)
    plt.close(fig)

    results["top_features"] = model.feature_importances.sort_values(ascending=False).head(5).to_dict()
    with open(config.REPORTS_DIR / f"metrics_{slug}.json", "w") as f:
        json.dump(results, f, indent=2, default=int)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotel", default=config.DEFAULT_HOTEL)
    parser.add_argument("--horizon", type=int, default=config.HORIZON)
    parser.add_argument("--data", default=str(config.RAW_CSV))
    parser.add_argument("--no-arima", action="store_true", help="skip AutoARIMA")
    args = parser.parse_args()
    print(json.dumps(run(args.hotel, args.horizon, args.data, not args.no_arima), indent=2, default=int))
