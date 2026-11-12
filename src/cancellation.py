"""Predict, at booking time, the chance that a booking will be canceled.

Use case: the hotel knows how many rooms are on the books, but ~37% of bookings cancel.
Summing the cancel probability of every booking gives the expected cancellations per night,
which tells the revenue manager how many rooms can be safely overbooked or resold.

Rules to avoid leakage (data that is only known AFTER the booking is made):
- Features: only what the hotel knows when the booking comes in (SAFE_FEATURES).
  Dropped on purpose:
    required_car_parking_spaces  filled in at check-in (0% of bookings with parking cancel)
    assigned_room_type           decided at check-in
    booking_changes              changes happen after the booking
    total_of_special_requests    can be added after the booking
    country                      per the dataset authors, set at check-in for many guests
    reservation_status(_date)    this IS the answer
- Split by arrival date: we only train on bookings whose outcome is already known
  (arrival before the cutoff), and test on later arrivals.

    python src/cancellation.py
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from data_prep import load_raw

TARGET = "is_canceled"
CATEGORICAL = ["hotel", "meal", "market_segment", "distribution_channel", "reserved_room_type",
               "deposit_type", "customer_type", "agent", "company"]
NUMERIC = ["lead_time", "arrival_month", "arrival_weekday", "arrival_week",
           "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
           "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
           "days_in_waiting_list", "adr"]
SAFE_FEATURES = CATEGORICAL + NUMERIC
LEAKY_EXTRA = ["required_car_parking_spaces", "assigned_room_type", "booking_changes",
               "total_of_special_requests", "country"]

VALID_START = "2017-01-01"   # train: arrivals before this date (outcome known)
TEST_START = "2017-03-01"    # valid: Jan-Feb 2017, test: Mar-Aug 2017


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["arrival_date"] <= config.DATA_END].copy()
    df["arrival_month"] = df["arrival_date"].dt.month
    df["arrival_weekday"] = df["arrival_date"].dt.weekday
    df["arrival_week"] = df["arrival_date"].dt.isocalendar().week.astype(int)
    df["children"] = df["children"].fillna(0)
    for c in CATEGORICAL + ["assigned_room_type", "country"]:
        df[c] = df[c].fillna("none").astype(str)
    return df


def split(df: pd.DataFrame):
    train = df[df["arrival_date"] < VALID_START]
    valid = df[(df["arrival_date"] >= VALID_START) & (df["arrival_date"] < TEST_START)]
    test = df[df["arrival_date"] >= TEST_START]
    return train, valid, test


def _to_lgb(X: pd.DataFrame, cats: list[str], ref: pd.DataFrame | None = None) -> pd.DataFrame:
    X = X.copy()
    for c in cats:
        levels = (ref if ref is not None else X)[c].astype("category").cat.categories
        X[c] = pd.Categorical(X[c], categories=levels)  # unseen values -> NaN
    return X


def fit_lgbm(train, valid, features, random_state=config.RANDOM_STATE):
    cats = [c for c in features if train[c].dtype == object]
    model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63,
                               min_child_samples=50, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, random_state=random_state, verbose=-1)
    Xtr = _to_lgb(train[features], cats)
    Xva = _to_lgb(valid[features], cats, ref=train)
    model.fit(Xtr, train[TARGET], eval_set=[(Xva, valid[TARGET])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return lambda d: model.predict_proba(_to_lgb(d[features], cats, ref=train))[:, 1], model


def fit_logreg(train):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    model = make_pipeline(pre, LogisticRegression(max_iter=2000))
    model.fit(train[SAFE_FEATURES], train[TARGET])
    return lambda d: model.predict_proba(d[SAFE_FEATURES])[:, 1]


def fit_rate_baseline(train):
    """Historical cancel rate per hotel x deposit type x market segment."""
    keys = ["hotel", "deposit_type", "market_segment"]
    rate = train.groupby(keys)[TARGET].mean().rename("p")
    overall = train[TARGET].mean()
    return lambda d: d[keys].join(rate, on=keys)["p"].fillna(overall).to_numpy()


def classification_metrics(y, p) -> dict:
    return {"roc_auc": round(float(roc_auc_score(y, p)), 3),
            "pr_auc": round(float(average_precision_score(y, p)), 3),
            "brier": round(float(brier_score_loss(y, p)), 4)}


def expected_cancellations(test: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Per arrival date: predicted vs real number of canceled bookings."""
    d = test.assign(p=p)
    return d.groupby(["hotel", "arrival_date"]).agg(actual=(TARGET, "sum"), predicted=("p", "sum"),
                                                    bookings=(TARGET, "size"))


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": p})
    d["bin"] = pd.qcut(d["p"], bins, labels=False, duplicates="drop")
    return d.groupby("bin").agg(mean_predicted=("p", "mean"), actual_rate=("y", "mean"), n=("y", "size"))


def run(data_path=config.RAW_CSV):
    df = prepare(load_raw(data_path))
    train, valid, test = split(df)
    y = test[TARGET]

    rate = fit_rate_baseline(train)
    logreg = fit_logreg(train)
    lgbm, lgbm_model = fit_lgbm(train, valid, SAFE_FEATURES)
    leaky, _ = fit_lgbm(train, valid, SAFE_FEATURES + LEAKY_EXTRA)

    preds = {"rate_baseline": rate(test), "logistic_regression": logreg(test),
             "lightgbm": lgbm(test), "lightgbm_LEAKY": leaky(test)}
    models = {name: classification_metrics(y, p) for name, p in preds.items()}

    # Business check: expected cancellations per arrival date (both hotels)
    constant = np.full(len(test), train[TARGET].mean())
    daily = {}
    for name, p in (("constant_rate", constant), ("rate_baseline", preds["rate_baseline"]),
                    ("lightgbm", preds["lightgbm"])):
        e = expected_cancellations(test, p)
        daily[name] = round(float((e["predicted"] - e["actual"]).abs().mean()), 2)

    importance = pd.Series(lgbm_model.booster_.feature_importance("gain"), index=SAFE_FEATURES)
    summary = {
        "train": f"arrivals before {VALID_START} ({len(train)} bookings)",
        "valid": f"{VALID_START} -> {TEST_START} ({len(valid)} bookings)",
        "test": f"{TEST_START} -> {config.DATA_END} ({len(test)} bookings)",
        "test_cancel_rate": round(float(y.mean()), 3),
        "models": models,
        "daily_cancellations_MAE": daily,
        "top_features": (importance / importance.sum()).sort_values(ascending=False).head(8).round(3).to_dict(),
    }
    calib = calibration_table(y, preds["lightgbm"])
    exp = expected_cancellations(test, preds["lightgbm"])
    return summary, calib, exp, (lgbm_model, train)


def export_model(model, train: pd.DataFrame, summary: dict, out_dir=config.MODELS_DIR) -> None:
    """Save the model as LightGBM TEXT (not pickle: loading a pickle can run any code)
    plus the allowed values of every input, used by the demo to validate user input."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out_dir / "cancel_lgbm.txt"))
    meta = {
        "features": SAFE_FEATURES,
        "categories": {c: sorted(train[c].astype(str).unique().tolist()) for c in CATEGORICAL},
        "numeric_ranges": {c: [float(train[c].min()), float(train[c].max())] for c in NUMERIC},
        "trained_on": summary["train"],
        "test_metrics": summary["models"]["lightgbm"],
    }
    with open(out_dir / "cancel_meta.json", "w") as f:
        json.dump(meta, f, indent=1)


def save_outputs(summary, calib, exp) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / "cancellation.json", "w") as f:
        json.dump(summary, f, indent=2)
    calib.round(3).to_csv(config.REPORTS_DIR / "cancellation_calibration.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 2]})
    a.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    a.plot(calib["mean_predicted"], calib["actual_rate"], marker="o", color="tab:blue")
    a.set(title="Calibration (test, LightGBM)", xlabel="predicted cancel probability",
          ylabel="real cancel rate", xlim=(0, 1), ylim=(0, 1))
    city = exp.loc["City Hotel"]
    b.plot(city.index, city["actual"], color="black", lw=1, label="Real cancellations")
    b.plot(city.index, city["predicted"], color="tab:blue", lw=1, label="Expected (sum of probabilities)")
    b.set(title="City Hotel: canceled bookings per arrival date, Mar - Aug 2017", ylabel="bookings")
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "cancellation.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(config.RAW_CSV))
    args = parser.parse_args()
    summary, calib, exp, (model, train) = run(args.data)
    save_outputs(summary, calib, exp)
    export_model(model, train, summary)
    print(json.dumps(summary, indent=2))
