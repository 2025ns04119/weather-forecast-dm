# Exploratory Data Analysis - summary

## 1. Overview

- Rows: **290,177** hourly records, **32** locations
- Period: 2024-11-02 to 2026-11-16
- Hours per location: min 8,784, max 17,873
- Missing values in the silver layer: 0

| Variable | mean | std | min | max | missing |
|---|---|---|---|---|---|
| temperature_2m | 20.18 | 9.31 | -25.6 | 46.3 | 0 |
| relative_humidity_2m | 67.52 | 22.14 | 3.0 | 100.0 | 0 |
| dew_point_2m | 12.82 | 9.18 | -32.6 | 31.0 | 0 |
| pressure_msl | 1013.6 | 7.02 | 974.9 | 1047.5 | 0 |
| cloud_cover | 52.58 | 42.01 | 0.0 | 100.0 | 0 |
| wind_speed_10m | 8.55 | 5.05 | 0.0 | 43.8 | 0 |
| wind_direction_10m | 182.51 | 103.96 | 0.0 | 360.0 | 0 |
| precipitation | 0.13 | 0.62 | 0.0 | 37.4 | 0 |

## 2. Findings

- **Distributions** (reports/figures/01_distributions.png): most hours have no rain, so precipitation is skewed; temperature is wide because the corpus covers every climate.
- **Correlations** (reports/figures/02_correlation_heatmap.png): strongest pairs - dew_point_2m vs temperature_2m: 0.73; pressure_msl vs temperature_2m: -0.51; dew_point_2m vs pressure_msl: -0.48.
- **Seasonality** (reports/figures/03_monthly_by_climate_zone.png): monthly curves differ by climate zone and are reversed in the southern hemisphere - reason for the *month* and *latitude* features.
- **Daily cycle** (reports/figures/04_bengaluru_daily_cycle.png): Bengaluru is warmest around 14:00 IST while humidity is lowest - reason for the *solar hour* feature.
- **Rain sparsity** (reports/figures/05_rain_sparsity.png): on average only 17.6% of hours are rainy, so rain is modelled as a probability and evaluated with Brier score and AUC.
- **Completeness** (reports/figures/06_completeness_heatmap.png): Bengaluru has 2 years of history, other cities 1 year; light cells inside the range would mean gaps.
- **Features vs target** (reports/figures/07_feature_target_correlation.png): most related to temperature 24 h later - temp_lag_24h (0.93), temp_lag_1h (0.92), temp_roll_mean_6h (0.81).

## 3. Data preparation applied

| Step | What was done | Where |
|---|---|---|
| Validation | Required fields, physical ranges, no future hours | `pipeline/profiling_validation/clean_weather.py` |
| Duplicates | One row per location + hour | silver layer primary key |
| Missing values | Gaps up to 3 h interpolated; rows still missing a feature dropped | `pipeline/feature_engineering/build_features.py` |
| Encoding | No categorical model inputs: location enters through latitude and solar hour; climate zone is used only for analysis | - |
| Scaling | StandardScaler for linear/logistic models; tree models use raw values | `pipeline/model_training/train.py` |
| Prepared dataset | Silver layer exported to `data/prepared/` and versioned with DVC | `pipeline/versioning_lineage/export_datasets.py` |