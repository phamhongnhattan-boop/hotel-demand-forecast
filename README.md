# Hotel Demand Forecasting & Price Suggestion

Forecast how many rooms a hotel will sell each day, 14 days ahead, then suggest a room price from that forecast.

## Problem

Hotels need to know future demand early to set prices and plan staff. This project:

1. Turns 119k booking records into a daily "rooms sold" time series per hotel.
2. Compares a seasonal naive baseline, AutoARIMA and LightGBM, with and without **on-the-books** data
   (rooms already booked, the main signal hotels use in practice).
3. Uses the forecast to suggest a daily price (simple, explainable rules).

## Data

[Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (Antonio, Almeida & Nunes, 2019): bookings of a city hotel and a resort hotel in Portugal, arrivals from Jul 2015 to Aug 2017.

Download `hotel_bookings.csv` and put it in `data/raw/`.

## Key findings from EDA

Full analysis: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)

- 42% of City Hotel and 28% of Resort Hotel bookings are canceled; the longer the lead time, the higher the cancel rate (10% → 68%).
- Strong yearly cycle: the City Hotel sells ~105 rooms/day in January vs ~218 in August. Both hotels are often full in summer.
- Price follows demand: Resort Hotel ADR goes from ~53 EUR in winter to ~190 EUR in August. When the hotel is full, ADR still varies a lot, so there is room to raise prices on forecast "full" days.

## Method

| Step | Detail |
|---|---|
| Daily series | Non-canceled bookings, each counts as 1 room per night of stay |
| Split | Train 2015-08 → 2017-02 (first month skipped: incomplete), Valid 2017-03 → 05, Test 2017-06 → 08 |
| Features | Weekday, month, day of year, lags ≥ horizon (no leakage), rolling mean/std |
| On-the-books | Rooms already booked for the date, as known 14 days before (booking date = arrival − lead time; canceled bookings count until their cancel date) + the same booked/actual ratio one year earlier |
| Models | Seasonal naive (same weekday, 14 days earlier) · AutoARIMA (weekly season, rolling origin) · LightGBM on the residual (actual − naive, early stopping) |
| Evaluation | Every test day is forecast exactly 14 days ahead, for all models. Two tests: summer only (Jun–Aug 2017) and a 12-month rolling backtest (Sep 2016 → Aug 2017) |
| Intervals | Quantile LightGBM (P10, P90) on the residual, calibrated with conformalized quantile regression (CQR) on the validation months |
| Pricing | Occupancy = forecast / capacity (95th percentile of rooms sold). v1 uses the point forecast; v2 raises only when P10 says ≥ 90% full. Both are scored against what really happened |

## Results

### 1. Full year: rolling backtest (Sep 2016 → Aug 2017, horizon = 14 days)

Each month is a test fold: train on everything before, early stopping on the 2 months before, then forecast every day of the month 14 days ahead.
MAE is over all 365 test days.

| Hotel | Seasonal naive | AutoARIMA | LightGBM (calendar + lags) | **LightGBM + on-the-books** |
|---|---|---|---|---|
| City Hotel | 22.5 | 20.2 | 20.2 | **9.3** (−59%) |
| Resort Hotel | 18.9 | 17.5 | 15.6 | **9.3** (−51%) |

- With on-the-books, LightGBM beats the baseline in 12/12 months (City) and 11/12 months (Resort).
- Without it, the model is weak in low season: in Dec 2016 the City Hotel MAE is 56 rooms vs 18 with on-the-books.
  In Oct 2016 the first LightGBM was even worse than the baseline (21.5 vs 12.1).

![backtest](reports/backtest_city_hotel.png)

### 2. Summer only (Jun–Aug 2017, horizon = 14 days)

| Hotel | Model | MAE | RMSE | MAPE |
|---|---|---|---|---|
| City Hotel | Seasonal naive | 7.11 | 9.64 | 3.37% |
| City Hotel | AutoARIMA | 6.35 | 8.21 | 3.01% |
| City Hotel | LightGBM (calendar + lags) | 5.84 | 7.60 | 2.78% |
| City Hotel | **LightGBM + on-the-books** | **4.29** | **5.51** | **2.03%** |
| Resort Hotel | Seasonal naive | 5.67 | 7.52 | 3.25% |
| Resort Hotel | AutoARIMA | 4.88 | 6.12 | 2.78% |
| Resort Hotel | LightGBM (calendar + lags) | 4.76 | 6.36 | 2.75% |
| Resort Hotel | **LightGBM + on-the-books** | **3.92** | **5.16** | **2.25%** |

In summer both hotels are almost full, so every model looks good. This is why the full-year backtest matters:
the first version showed −18% MAE on summer data but only −10% (City) over a whole year.

![forecast](reports/forecast_city_hotel.png)

### 3. Price suggestion, tested (Sep 2016 → Aug 2017)

A price rule is only useful if its decisions are right. The backtest checks every +15% raise
against the real occupancy of that day: a raise on a day that ended below 90% full is a **wrong raise**
(the hotel risks losing guests).

**80% interval.** Raw quantile models were over-confident: the [P10, P90] band held the real value only
66% (City) and 63% (Resort) of the time. After CQR calibration on the 2 validation months of each fold:

| Hotel | Point MAE | Interval coverage (target 80%) | Mean width |
|---|---|---|---|
| City Hotel | 9.3 | **80.3%** | 32 rooms |
| Resort Hotel | 9.2 | **80.0%** | 29 rooms |

**Raise +15% (day really ≥ 90% full?)**

| Hotel | Rule | Days raised | Precision | Recall | Wrong raises |
|---|---|---|---|---|---|
| City Hotel | v1: point forecast ≥ 90% | 212 | 93.9% | 86.9% | 13 |
| City Hotel | **v2: P10 ≥ 90%** | 149 | **98.0%** | 63.8% | **3** |
| Resort Hotel | v1: point forecast ≥ 90% | 208 | 89.4% | 86.9% | 22 |
| Resort Hotel | **v2: P10 ≥ 90%** | 144 | **97.9%** | 65.9% | **3** |

v2 cuts wrong raises from 13 → 3 (City) and 22 → 3 (Resort). The cost: it misses about a third of the
full days, which get a +5% raise instead of +15%. Which rule is better depends on what the hotel fears more:
losing guests (v2) or leaving money on the table (v1). The data has no price elasticity, so this choice
cannot be learned here; the backtest only shows the trade-off clearly.

![pricing](reports/pricing_city_hotel.png)

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# Optional: test with fake data first
python src/sample_data.py

python src/data_prep.py           # -> data/processed/daily_demand.csv
python src/train.py --hotel "City Hotel"
python src/train.py --hotel "Resort Hotel"
python src/backtest.py --hotel "City Hotel"            # 12-month backtest
python src/backtest.py --hotel "City Hotel" --arima    # + AutoARIMA (slow)
python src/price_backtest.py --hotel "City Hotel"      # intervals + pricing test
python -m pytest -q
```

Outputs go to `reports/`: metrics (JSON), forecast + suggested price per day (CSV), charts (PNG),
`backtest_*.csv/json/png` for the 12-month backtest, and `pricing_*.csv/json/png` for the pricing test.

### Optional: BigQuery ML

`bigquery/` has an upload script and ARIMA_PLUS SQL to run the same comparison in BigQuery ML
(works in the free BigQuery sandbox). Not run yet.

## Project structure

```
src/
  config.py       settings (paths, split dates, horizon)
  data_prep.py    bookings -> daily series, on-the-books series
  features.py     calendar, lag and on-the-books features
  models.py       seasonal naive, residual LightGBM
  arima.py        AutoARIMA (statsforecast), rolling origin
  evaluate.py     MAE, RMSE, MAPE
  intervals.py    P10/P90 quantile models + conformal calibration
  pricing.py      price rules (v1 point, v2 interval) and their scoring
  price_backtest.py  12-month test of intervals and price decisions
  train.py        full pipeline (summer test + price suggestion)
  backtest.py     12-month rolling backtest
notebooks/        EDA
bigquery/         optional BigQuery ML scripts
tests/            unit tests
```

## Lesson learned

The first version trained LightGBM on the raw target and dropped every row with a missing `lag_364`.
That left only 245 training rows; for the Resort Hotel, early stopping kept 1 tree and the model
predicted close to the yearly average (MAE 28 vs 5.7 for the baseline). Fix: keep rows with missing
long lags (LightGBM handles NaN) and learn the correction to the naive forecast, so a weak model
falls back to the baseline instead of the average.

The second lesson came from testing on a full year instead of only summer. The summer test hid a weak model:
in winter the error was 5–8 times larger, and in some months LightGBM was worse than the baseline.
Calendar and past demand cannot tell a quiet December week from a busy one, but the booking list can.
Adding "rooms already booked 14 days before" (only data known at forecast time, checked by a unit test)
halved the full-year error.

## Limitations and next steps

- The dataset has no room count, so capacity is estimated.
- Pricing rules are hand-made, not learned from price elasticity; the backtest scores decisions, not revenue.
- On-the-books is rebuilt from `lead_time` and the cancel date; a real PMS would give exact daily snapshots.
- Only one year of backtest is possible with this data (2 years in total, and the model needs 1 year of history).
- Data is from Portugal (2015–2017), so patterns may differ from Vietnamese hotels.
