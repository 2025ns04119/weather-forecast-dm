-- ============================================================
-- Migration v3: multi-target forecasting (safe to re-run)
-- ============================================================

-- Feature version v3: extra weather-memory features
ALTER TABLE features.weather_features ADD COLUMN IF NOT EXISTS humidity_lag_24h   REAL;
ALTER TABLE features.weather_features ADD COLUMN IF NOT EXISTS cloud_lag_24h      REAL;
ALTER TABLE features.weather_features ADD COLUMN IF NOT EXISTS wind_lag_24h       REAL;
ALTER TABLE features.weather_features ADD COLUMN IF NOT EXISTS pressure_change_3h REAL;
ALTER TABLE features.weather_features ADD COLUMN IF NOT EXISTS rain_6h            REAL;
ALTER TABLE features.weather_features ADD COLUMN IF NOT EXISTS rain_24h           REAL;

-- Predictions now hold every forecast variable
ALTER TABLE ml.predictions ADD COLUMN IF NOT EXISTS predicted_humidity REAL;
ALTER TABLE ml.predictions ADD COLUMN IF NOT EXISTS predicted_wind     REAL;
ALTER TABLE ml.predictions ADD COLUMN IF NOT EXISTS predicted_cloud    REAL;
ALTER TABLE ml.predictions ADD COLUMN IF NOT EXISTS rain_probability   REAL;
