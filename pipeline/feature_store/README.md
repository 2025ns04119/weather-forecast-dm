# Feature store

A simple custom feature store with three parts:

| Part | Implementation |
|---|---|
| **Registry** (feature metadata) | `pipeline/feature_store/registry.yaml`, versioned in Git and copied to the table `features.feature_registry` (`python -m pipeline.feature_store.store --sync`) |
| **Offline store** | `features.weather_features` in Neon: one row per location + hour + `feature_version` |
| **Retrieval API** | `pipeline/feature_store/store.py` - `FeatureStore.get_training_features(version, ...)` and `get_online_features(location, version, n_hours)` |

## Feature versions

| Version | Change |
|---|---|
| v1 | Local clock hour (IST), temperature lags and rolling means, current conditions - Bengaluru only |
| v2 | Solar hour instead of clock hour, plus latitude - one global model for any city |
| v3 (current) | Adds 24 h lags of humidity, cloud and wind, 3 h pressure change, 6 h and 24 h rain - multi-target model |

## Features (v3)

| Feature | Source column | Transformation |
|---|---|---|
| hour | ts, longitude | floor((UTC hour + longitude / 15) mod 24) - solar hour |
| month | ts | month(ts) |
| latitude | clean.locations.latitude | copied |
| temp_lag_1h / 3h / 24h | temperature_2m | value 1 / 3 / 24 hours earlier |
| temp_roll_mean_6h / 24h | temperature_2m | mean of the last 6 / 24 hours |
| humidity, pressure, wind_speed, cloud_cover | relative_humidity_2m, pressure_msl, wind_speed_10m, cloud_cover | current value |
| humidity_lag_24h, cloud_lag_24h, wind_lag_24h | same columns | value 24 hours earlier |
| pressure_change_3h | pressure_msl | now minus 3 hours earlier |
| rain_6h, rain_24h | precipitation | sum over the last 6 / 24 hours |

Rules: gaps up to 3 h interpolated; rows with a missing feature dropped; only past values used (no leakage).

## Who uses which retrieval

| Consumer | Method |
|---|---|
| Training (`pipeline/model_training/train.py`) | `get_training_features("v3")` - all locations, full history |
| Hourly forecast (`pipeline/model_deployment/predict.py`) | `get_online_features("Bengaluru", "v3", 24)` - latest 24 hours |
| App search for any city (`pipeline/model_deployment/on_demand.py`) | Same `build_features()` function computed on demand (no storage) |

`python -m pipeline.feature_store.store --check` stops the pipeline if the registry and the code disagree.
Demo: `python -m pipeline.feature_store.demo_retrieval`.
