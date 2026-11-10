"""Rolling backtest: test every month from Sep 2016 to Aug 2017, not only the summer.

For each test month:
  - train on all data before (test month - 2 months)
  - early stopping on the 2 months just before the test month
  - forecast every day of the test month exactly `horizon` days ahead

    python src/backtest.py --hotel "City Hotel"
    python src/backtest.py --hotel "City Hotel" --arima     # also AutoARIMA (slow)
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from data_prep import build_daily, build_on_the_books, load_raw
from evaluate import metrics
from features import make_features, naive_lag
from models import ResidualLGBM

FIRST_TEST_MONTH = "2016-09-01"  # needs 1 year of history before it (lag_364)
VALID_MONTHS = 2


def monthly_folds(first_month: str = FIRST_TEST_MONTH, last_day: str = config.DATA_END):
    """List of (valid_start, test_start, test_end), one per month."""
    folds = []
    for start in pd.date_range(first_month, last_day, freq="MS"):
        end = min(start + pd.offsets.MonthEnd(0), pd.Timestamp(last_day))
        folds.append((start - pd.DateOffset(months=VALID_MONTHS), start, end))
    return folds


def run_backtest(hotel: str, horizon: int, data_path=config.RAW_CSV, use_arima: bool = False):
    raw = load_raw(data_path)
    daily = build_daily(raw, hotel, config.DATA_END)
    otb = build_on_the_books(raw, hotel, horizon)
    X = make_features(daily, horizon, on_the_books=otb)

    base = f"lag_{naive_lag(horizon)}"
    X = X[(X.index >= config.TRAIN_START) & X[base].notna()]
    target = "rooms_sold"
    otb_cols = [f"otb_{horizon}", "otb_ratio_ly"]
    feature_sets = {
        "lightgbm": [c for c in X.columns if c != target and c not in otb_cols],
        "lightgbm_otb": [c for c in X.columns if c != target],
    }

    rows, daily_preds = [], []
    for valid_start, test_start, test_end in monthly_folds():
        train = X[X.index < valid_start]
        valid = X[(X.index >= valid_start) & (X.index < test_start)]
        test = X[(X.index >= test_start) & (X.index <= test_end)]
        y = test[target]

        preds = {"seasonal_naive": test[base]}
        for name, cols in feature_sets.items():
            model = ResidualLGBM(horizon, config.RANDOM_STATE).fit(
                train[cols], train[target], valid[cols], valid[target])
            preds[name] = model.predict(test[cols])
        if use_arima:
            from arima import arima_forecast
            preds["auto_arima"] = arima_forecast(
                daily, horizon, config.TRAIN_START, str(test_start.date()), str(test_end.date())
            ).reindex(test.index)

        row = {"month": test_start.strftime("%Y-%m"), "mean_rooms": round(y.mean(), 1)}
        row.update({name: metrics(y, p)["MAE"] for name, p in preds.items()})
        rows.append(row)
        daily_preds.append(pd.DataFrame({"actual": y, **{k: v.round(1) for k, v in preds.items()}}))

    table = pd.DataFrame(rows)
    models = [c for c in table.columns if c not in ("month", "mean_rooms")]
    all_days = pd.concat(daily_preds)
    summary = {
        "hotel": hotel,
        "horizon_days": horizon,
        "test_period": f"{all_days.index.min().date()} -> {all_days.index.max().date()}",
        "n_folds": len(table),
        # MAE over all test days (not the mean of monthly MAE)
        "overall": {m: metrics(all_days["actual"], all_days[m]) for m in models},
        "months_better_than_naive": {
            m: int((table[m] < table["seasonal_naive"]).sum()) for m in models if m != "seasonal_naive"},
    }
    naive_mae = summary["overall"]["seasonal_naive"]["MAE"]
    summary["mae_improvement_vs_naive_%"] = {
        m: round((naive_mae - summary["overall"][m]["MAE"]) / naive_mae * 100, 1)
        for m in models if m != "seasonal_naive"}
    return table, all_days, summary


def save_outputs(hotel: str, table: pd.DataFrame, all_days: pd.DataFrame, summary: dict) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = hotel.lower().replace(" ", "_")
    table.to_csv(config.REPORTS_DIR / f"backtest_{slug}.csv", index=False)
    with open(config.REPORTS_DIR / f"backtest_{slug}.json", "w") as f:
        json.dump(summary, f, indent=2)

    styles = {"seasonal_naive": ("Seasonal naive", "tab:orange"),
              "auto_arima": ("AutoARIMA", "tab:purple"),
              "lightgbm": ("LightGBM (calendar + lags)", "tab:blue"),
              "lightgbm_otb": ("LightGBM + on-the-books", "tab:green")}
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, (label, color) in styles.items():
        if name in table:
            ax.plot(table["month"], table[name], marker="o", label=label, color=color)
    ax.set_title(f"{hotel}: MAE by test month (forecast {summary['horizon_days']} days ahead)")
    ax.set_ylabel("MAE (rooms)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / f"backtest_{slug}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotel", default=config.DEFAULT_HOTEL)
    parser.add_argument("--horizon", type=int, default=config.HORIZON)
    parser.add_argument("--data", default=str(config.RAW_CSV))
    parser.add_argument("--arima", action="store_true", help="also run AutoARIMA (slow)")
    args = parser.parse_args()

    table, all_days, summary = run_backtest(args.hotel, args.horizon, args.data, args.arima)
    save_outputs(args.hotel, table, all_days, summary)
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))
