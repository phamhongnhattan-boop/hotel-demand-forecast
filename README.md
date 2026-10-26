# Hotel Demand Forecasting & Price Suggestion

Forecast how many rooms a hotel will sell each day, 14 days ahead, then suggest a room price from that forecast.

> Status: work in progress. Fill in the result table after running on the real data.

## Problem

Hotels need to know future demand early to set prices and plan staff. This project:

1. Turns 119k booking records into a daily "rooms sold" time series per hotel.
2. Compares a seasonal naive baseline, LightGBM, and BigQuery ML ARIMA_PLUS.
3. Uses the forecast to suggest a daily price (simple, explainable rules).

## Data

[Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (Antonio, Almeida & Nunes, 2019): bookings of a city hotel and a resort hotel in Portugal, arrivals from Jul 2015 to Aug 2017.

Download `hotel_bookings.csv` and put it in `data/raw/`.

## Method

| Step | Detail |
|---|---|
| Daily series | Non-canceled bookings, each counts as 1 room per night of stay |
| Split | Train 2015-08 → 2017-02 (first month skipped: incomplete), Valid 2017-03 → 05, Test 2017-06 → 08 |
| Features | Weekday, month, day of year, lags ≥ horizon (no leakage), rolling mean/std |
| Models | Seasonal naive, LightGBM on the residual (actual − naive, early stopping), BigQuery ML ARIMA_PLUS |
| Pricing | Occupancy = forecast / capacity (95th percentile of rooms sold) → price multiplier |

## Results (test period, horizon = 14 days)

| Hotel | Model | MAE | RMSE | MAPE |
|---|---|---|---|---|
| City Hotel | Seasonal naive | 7.11 | 9.64 | 3.37% |
| City Hotel | **LightGBM (residual)** | **5.84** | **7.60** | **2.78%** |
| City Hotel | BigQuery ARIMA_PLUS | – | – | – |
| Resort Hotel | Seasonal naive | 5.67 | 7.52 | 3.25% |
| Resort Hotel | **LightGBM (residual)** | **4.76** | **6.36** | **2.75%** |
| Resort Hotel | BigQuery ARIMA_PLUS | – | – | – |

LightGBM lowers MAE by 18% (City) and 16% (Resort) vs the seasonal naive baseline.
Note: the test period is high season, when both hotels are almost full, so errors are small for every model.

![forecast](reports/forecast_city_hotel.png)

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# Optional: test with fake data first
python src/sample_data.py

python src/data_prep.py           # -> data/processed/daily_demand.csv
python src/train.py --hotel "City Hotel" --horizon 14
python -m pytest -q
```

BigQuery ML part: see `bigquery/upload_to_bq.py` and `bigquery/forecast_arima.sql`.

## Project structure

```
src/
  config.py       settings (paths, split dates, horizon)
  data_prep.py    bookings -> daily series
  features.py     calendar + lag features
  models.py       seasonal naive, LightGBM
  evaluate.py     MAE, RMSE, MAPE
  pricing.py      price suggestion rules
  train.py        full pipeline
bigquery/         upload script + ARIMA_PLUS SQL
tests/            unit tests
```

## Lesson learned

The first version trained LightGBM on the raw target and dropped every row with a missing `lag_364`.
That left only 245 training rows; for the Resort Hotel, early stopping kept 1 tree and the model
predicted close to the yearly average (MAE 28 vs 5.7 for the baseline). Fix: keep rows with missing
long lags (LightGBM handles NaN) and learn the correction to the naive forecast, so a weak model
falls back to the baseline instead of the average.

## Limitations

- The dataset has no room count, so capacity is estimated.
- Pricing rules are hand-made, not learned from price elasticity.
- Data is from Portugal (2015–2017), so patterns may differ from Vietnamese hotels.
