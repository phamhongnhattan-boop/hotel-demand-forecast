"""Demo: demand forecast, tested price suggestions and cancellation risk.

    streamlit run app/streamlit_app.py

The app only READS files that are in the repo (reports/*.csv|json, models/*.txt|json).
It does not train anything and does not need the raw dataset.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import altair as alt
import lightgbm as lgb
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from pricing import ACTIONS, score_actions  # noqa: E402
from validation import HOTELS, BASE_PRICE_RANGE, InvalidInput, booking_row, check_base_price, report_slug  # noqa: E402

REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
ACTION_LABEL = {"raise": "Raise +15%", "small_raise": "Raise +5%", "keep": "Keep", "discount": "Discount -10%"}

st.set_page_config(page_title="Hotel demand & pricing demo", layout="wide")


# ---------- cached loaders (fixed file names only) ----------
@st.cache_data
def load_pricing(slug: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / f"pricing_{slug}.csv", parse_dates=["date"])


@st.cache_data
def load_json(name: str) -> dict:
    with open(REPORTS / name, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_cancel_model():
    meta = json.loads((MODELS / "cancel_meta.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(MODELS / "cancel_lgbm.txt"))  # text format, not pickle
    return booster, meta


st.title("Hotel demand forecast & pricing")
st.caption("Hotel Booking Demand dataset (Portugal, 2015-2017). Forecasts are 14 days ahead, "
           "tested on Sep 2016 - Aug 2017 with a monthly rolling backtest.")

tab_price, tab_cancel, tab_results = st.tabs(["Price suggestion", "Cancellation risk", "Model results"])

# ---------- Tab 1: price suggestion ----------
with tab_price:
    c1, c2, c3 = st.columns(3)
    hotel = c1.radio("Hotel", list(HOTELS), horizontal=True)
    days = load_pricing(report_slug(hotel))
    months = days["date"].dt.to_period("M").astype(str).unique().tolist()
    month = c2.selectbox("Month", months, index=len(months) - 1)
    base = c3.number_input("Base price (EUR / night)", min_value=BASE_PRICE_RANGE[0],
                           max_value=BASE_PRICE_RANGE[1], value=100.0, step=5.0)
    try:
        base = check_base_price(base)
    except InvalidInput as e:
        st.error(str(e))
        st.stop()

    m = days[days["date"].dt.to_period("M").astype(str) == month].copy()
    m["action"] = m["v2_action"].map(ACTION_LABEL)
    m["suggested_price"] = (base * m["v2_action"].map(ACTIONS)).round(0)
    m["occupancy_real_%"] = (m["actual"] / m["capacity"] * 100).round(0)

    band = alt.Chart(m).mark_area(opacity=0.25, color="#2ca02c").encode(
        x=alt.X("date:T", title=None), y=alt.Y("p10:Q", title="rooms sold", scale=alt.Scale(zero=False)), y2="p90:Q")
    p50 = alt.Chart(m).mark_line(color="#2ca02c").encode(x="date:T", y="p50:Q")
    actual = alt.Chart(m).mark_line(color="black").encode(x="date:T", y="actual:Q")
    full = alt.Chart(m).mark_rule(color="#d62728", strokeDash=[4, 4]).encode(y=alt.datum(float(m["capacity"].iloc[0]) * 0.9))
    st.altair_chart((band + p50 + actual + full).properties(height=300), width="stretch")
    st.caption("Green band: 80% interval (P10-P90). Green line: forecast. Black: what really happened. "
               "Red dashed: 90% of capacity. A +15% raise is suggested only when P10 is above the red line.")

    s = score_actions(m["v2_action"], m["actual"], float(m["capacity"].iloc[0]))["raise"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Days with +15%", s["days"])
    k2.metric("Raises that were right", f"{s['precision_%']}%" if s["precision_%"] is not None else "-")
    k3.metric("Wrong raises (not full)", s["wrong_days"])

    m["date"] = m["date"].dt.date
    st.dataframe(
        m[["date", "p10", "p50", "p90", "actual", "occupancy_real_%", "action", "suggested_price"]]
        .rename(columns={"actual": "real"}).round(1),
        hide_index=True, width="stretch")

# ---------- Tab 2: cancellation risk ----------
with tab_cancel:
    booster, meta = load_cancel_model()
    cats = meta["categories"]
    st.write("Cancel probability of one new booking, using only what the hotel knows when the booking comes in.")
    with st.form("booking"):
        a, b, c = st.columns(3)
        form = {
            "hotel": a.selectbox("Hotel", cats["hotel"]),
            "arrival_date": a.date_input("Arrival date", value=date(2017, 7, 15)),
            "lead_time": a.number_input("Days between booking and arrival", 0, 737, 60),
            "stays_in_weekend_nights": a.number_input("Weekend nights", 0, 19, 1),
            "stays_in_week_nights": a.number_input("Week nights", 0, 50, 2),
            "adults": b.number_input("Adults", 0, 10, 2),
            "children": b.number_input("Children", 0, 10, 0),
            "babies": b.number_input("Babies", 0, 10, 0),
            "adr": b.number_input("Price per night (EUR)", 0.0, 1000.0, 100.0, 5.0),
            "deposit_type": b.selectbox("Deposit", cats["deposit_type"]),
            "market_segment": c.selectbox("Market segment", cats["market_segment"],
                                          index=cats["market_segment"].index("Online TA")),
            "distribution_channel": c.selectbox("Channel", cats["distribution_channel"],
                                                index=cats["distribution_channel"].index("TA/TO")),
            "customer_type": c.selectbox("Customer type", cats["customer_type"],
                                         index=cats["customer_type"].index("Transient")),
            "meal": c.selectbox("Meal", cats["meal"]),
            "reserved_room_type": c.selectbox("Room type", cats["reserved_room_type"]),
            "is_repeated_guest": int(c.checkbox("Repeat guest")),
            "previous_cancellations": c.number_input("Past cancellations of this guest", 0, 26, 0),
            "previous_bookings_not_canceled": 0, "days_in_waiting_list": 0,
            "agent": "none", "company": "none",
        }
        sent = st.form_submit_button("Predict")
    if sent:
        try:
            row = booking_row(form, meta)
        except InvalidInput as e:
            st.error(f"Input not accepted: {e}")
        else:
            p = float(booster.predict(row)[0])
            st.metric("Cancel probability", f"{p:.0%}")
            st.progress(min(max(p, 0.0), 1.0))
            st.caption(f"Model: LightGBM, trained on {meta['trained_on']}. "
                       f"Test ROC-AUC {meta['test_metrics']['roc_auc']}, PR-AUC {meta['test_metrics']['pr_auc']}. "
                       "Agent and company are set to 'none' in this demo.")

# ---------- Tab 3: results ----------
with tab_results:
    rows = []
    for h in HOTELS:
        bt = load_json(f"backtest_{report_slug(h)}.json")["overall"]
        pr = load_json(f"pricing_{report_slug(h)}.json")
        rows.append({"hotel": h, "naive MAE": bt["seasonal_naive"]["MAE"], "AutoARIMA MAE": bt["auto_arima"]["MAE"],
                     "LightGBM + on-the-books MAE": bt["lightgbm_otb"]["MAE"],
                     "80% interval coverage": pr["interval_80"]["coverage_%"],
                     "wrong raises v1 -> v2": f"{pr['pricing_v1_point']['raise']['wrong_days']} -> "
                                              f"{pr['pricing_v2_interval']['raise']['wrong_days']}"})
    st.subheader("Demand forecast (12-month backtest)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    cm = load_json("cancellation.json")
    st.subheader("Cancellation model (test Mar - Aug 2017)")
    st.dataframe(pd.DataFrame(cm["models"]).T, width="stretch")
    st.caption("The 'LEAKY' row uses columns only known after booking (parking, assigned room...). "
               "It is shown as a warning, not used.")
