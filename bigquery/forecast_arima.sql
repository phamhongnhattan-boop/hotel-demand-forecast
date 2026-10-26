-- BigQuery ML: ARIMA_PLUS model per hotel.
-- Replace YOUR_PROJECT with your GCP project id, then run each block in the BigQuery console.
-- Data is from Portugal hotels, so holiday_region = 'PT'.

-- 1) Train on data before the test period
CREATE OR REPLACE MODEL `YOUR_PROJECT.hotel.demand_arima`
OPTIONS (
  model_type = 'ARIMA_PLUS',
  time_series_timestamp_col = 'date',
  time_series_data_col = 'rooms_sold',
  time_series_id_col = 'hotel',
  horizon = 92,
  holiday_region = 'PT',
  auto_arima = TRUE
) AS
SELECT date, hotel, rooms_sold
FROM `YOUR_PROJECT.hotel.daily_demand`
WHERE date < '2017-06-01';

-- 2) Forecast the test period (2017-06-01 -> 2017-08-31)
SELECT hotel, DATE(forecast_timestamp) AS date,
       ROUND(forecast_value, 1) AS forecast,
       ROUND(prediction_interval_lower_bound, 1) AS lower,
       ROUND(prediction_interval_upper_bound, 1) AS upper
FROM ML.FORECAST(MODEL `YOUR_PROJECT.hotel.demand_arima`,
                 STRUCT(92 AS horizon, 0.9 AS confidence_level))
ORDER BY hotel, date;

-- 3) Compare with actual values (MAE per hotel) to put next to the LightGBM result
SELECT f.hotel,
       ROUND(AVG(ABS(a.rooms_sold - f.forecast_value)), 3) AS mae
FROM ML.FORECAST(MODEL `YOUR_PROJECT.hotel.demand_arima`, STRUCT(92 AS horizon)) AS f
JOIN `YOUR_PROJECT.hotel.daily_demand` AS a
  ON a.hotel = f.hotel AND a.date = DATE(f.forecast_timestamp)
GROUP BY f.hotel;

-- 4) See what the model learned (trend, seasonality, holidays)
SELECT * FROM ML.ARIMA_EVALUATE(MODEL `YOUR_PROJECT.hotel.demand_arima`);
