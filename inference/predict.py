"""
Batch inference: 7-day hourly temperature forecast.

1. Load the champion model (weather-temp-7day) from the MLflow registry
2. Read the latest 24 hourly feature rows from features.weather_features
3. Forecast the next 168 hours (7 days)
4. Save to ml.predictions (with model name + version -> lineage)

Usage (from the project root):
    python -m inference.predict
"""
import os

import pandas as pd
from psycopg2.extras import execute_values
from sqlalchemy import text

from common.db import finish_run, get_engine, start_run
from common.model_loader import MODEL_NAME, load_champion
from features.build_features import FEATURE_COLS, FEATURE_VERSION

MAX_FEATURE_AGE_HOURS = 6   # refuse to forecast from stale data


def latest_features(engine, location: str) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT ts, {", ".join(FEATURE_COLS)} FROM features.weather_features
            WHERE location = :loc AND feature_version = :v
            ORDER BY ts DESC LIMIT 24"""), conn,
            params={"loc": location, "v": FEATURE_VERSION})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def save_predictions(engine, fc: pd.DataFrame, location: str, version: str, made_at):
    rows = [(location, made_at, r.target_ts.to_pydatetime(), float(r.predicted_temp),
             MODEL_NAME, version) for r in fc.itertuples(index=False)]
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO ml.predictions
                    (location, made_at, target_ts, predicted_temp, model_name, model_version)
                VALUES %s""", rows, page_size=1000)
        raw_conn.commit()
    finally:
        raw_conn.close()


def main():
    location = os.getenv("LOCATION", "Bengaluru")
    engine = get_engine()
    run_id = start_run(engine, "predict_7day")
    try:
        feats = latest_features(engine, location)
        if len(feats) < 24:
            raise RuntimeError(f"Need 24 feature rows, found {len(feats)}")
        age_h = (pd.Timestamp.now(tz="UTC") - feats["ts"].max()).total_seconds() / 3600
        if age_h > MAX_FEATURE_AGE_HOURS:
            raise RuntimeError(f"Latest features are {age_h:.1f} h old - run ingestion/cleaning/features first")

        model, version = load_champion()
        fc = model.forecast(feats)
        made_at = pd.Timestamp.now(tz="UTC").to_pydatetime()
        save_predictions(engine, fc, location, version, made_at)

        finish_run(engine, run_id, "success", len(feats), len(fc),
                   f"{MODEL_NAME} v{version}; from {fc.target_ts.min()} to {fc.target_ts.max()}")

        local = fc.assign(date=fc["target_ts"].dt.tz_convert("Asia/Kolkata").dt.date)
        daily = local.groupby("date")["predicted_temp"].agg(["min", "max"]).round(1)
        print(f"Model: {MODEL_NAME} v{version} | forecast hours saved: {len(fc)}\n")
        print("7-day forecast for", location, "(IST)")
        print(daily.rename(columns={"min": "min C", "max": "max C"}).to_string())
    except Exception as e:
        finish_run(engine, run_id, "failed", notes=str(e)[:500])
        raise


if __name__ == "__main__":
    main()
