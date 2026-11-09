"""
On-demand 7-day forecast for ANY location (used by the dashboard search).

Same steps as the stored pipeline, run in memory:
    Open-Meteo (last 7 days) -> same validation rules -> same v2 features -> global model
"""
import pandas as pd

from features.build_features import build_features
from ingestion.open_meteo import fetch_recent
from validation.clean_weather import apply_rules, drop_future_and_duplicates, payload_to_df


def forecast_location(model, name: str, lat: float, lon: float) -> dict:
    """
    Returns {"forecast": DataFrame[target_ts, day_ahead, predicted_temp],
             "observed": DataFrame[ts, temperature_2m, ...],
             "current": dict | None, "elevation": float | None,
             "rejected": int}
    """
    payload = fetch_recent(lat, lon, past_days=7, with_current=True)
    raw = {"id": 0, "location": name, "endpoint": "forecast", "payload": payload}
    df = payload_to_df(raw)
    good, rejected = apply_rules(df)
    good, _, _ = drop_future_and_duplicates(good)

    feats = build_features(good, lat, lon)
    if len(feats) < 24:
        raise RuntimeError(f"Not enough recent data for {name} ({len(feats)} usable hours)")
    fc = model.forecast(feats.tail(24))
    return {
        "forecast": fc,
        "observed": good.reset_index(drop=True),
        "current": payload.get("current"),
        "elevation": payload.get("elevation"),
        "rejected": len(rejected),
        "made_at": pd.Timestamp.now(tz="UTC"),
    }
