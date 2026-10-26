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
| Split | Train < 2017-03-01, Valid 2017-03 → 05, Test 2017-06 → 08 |
| Features | Weekday, month, day of year, lags ≥ horizon (no leakage), rolling mean/std |
| Models | Seasonal naive, LightGBM (early stopping), BigQuery ML ARIMA_PLUS |
| Pricing | Occupancy = forecast / capacity (95th percentile of rooms sold) → price multiplier |

## Results (test period, horizon = 14 days)

| Hotel | Model | MAE | RMSE | MAPE |
|---|---|---|---|---|
| City Hotel | Seasonal naive | – | – | – |
| City Hotel | LightGBM | – | – | – |
| City Hotel | BigQuery ARIMA_PLUS | – | – | – |

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

## Limitations

- The dataset has no room count, so capacity is estimated.
- Pricing rules are hand-made, not learned from price elasticity.
- Data is from Portugal (2015–2017), so patterns may differ from Vietnamese hotels.
