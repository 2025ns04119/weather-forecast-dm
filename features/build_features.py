"""
Feature engineering (SILVER -> GOLD).

Reads clean.weather_hourly, builds ML features and the label, and upserts
them into features.weather_features (versioned with feature_version).

Features (all use only past/current values -> no data leakage):
    hour, month, day_of_week                  time features
    temp_lag_1h, temp_lag_3h, temp_lag_24h    lag features
    temp_roll_mean_6h, temp_roll_mean_24h     rolling means
    humidity, pressure, wind_speed, cloud_cover
Label:
    target_temp_24h   temperature 24 hours ahead (null for the latest 24 h ->
                      these rows are used for prediction, not training)

Usage (from the project root):
    python -m features.build_features            # recompute the last 3 days
    python -m features.build_features --all      # recompute everything
"""
import argparse

import pandas as pd
from psycopg2.extras import execute_values
from sqlalchemy import text

from common.db import finish_run, get_engine, start_run

FEATURE_VERSION = "v1"
MAX_INTERPOLATE_HOURS = 3   # fill gaps up to 3 h; longer gaps stay empty

FEATURE_COLS = [
    "hour", "month", "day_of_week",
    "temp_lag_1h", "temp_lag_3h", "temp_lag_24h",
    "temp_roll_mean_6h", "temp_roll_mean_24h",
    "humidity", "pressure", "wind_speed", "cloud_cover",
]
TARGET = "target_temp_24h"


def load_clean(engine, location: str, since=None) -> pd.DataFrame:
    sql = """SELECT ts, temperature_2m, relative_humidity_2m, pressure_msl,
                    wind_speed_10m, cloud_cover
             FROM clean.weather_hourly WHERE location = :loc"""
    params = {"loc": location}
    if since is not None:
        sql += " AND ts >= :since"
        params["since"] = since
    sql += " ORDER BY ts"
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Pure function: clean hourly data -> feature table."""
    if clean.empty:
        return pd.DataFrame()

    # 1. regular hourly grid + short-gap interpolation
    df = clean.set_index("ts").sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    df = df.reindex(full)
    df = df.interpolate(method="time", limit=MAX_INTERPOLATE_HOURS, limit_area="inside")

    t = df["temperature_2m"]
    out = pd.DataFrame(index=df.index)

    # 2. time features (IST is what matters for daily weather patterns)
    local = df.index.tz_convert("Asia/Kolkata")
    out["hour"] = local.hour
    out["month"] = local.month
    out["day_of_week"] = local.dayofweek

    # 3. lag + rolling features (past only)
    out["temp_lag_1h"] = t.shift(1)
    out["temp_lag_3h"] = t.shift(3)
    out["temp_lag_24h"] = t.shift(24)
    out["temp_roll_mean_6h"] = t.rolling(6, min_periods=6).mean()
    out["temp_roll_mean_24h"] = t.rolling(24, min_periods=24).mean()

    # 4. current conditions
    out["humidity"] = df["relative_humidity_2m"]
    out["pressure"] = df["pressure_msl"]
    out["wind_speed"] = df["wind_speed_10m"]
    out["cloud_cover"] = df["cloud_cover"]

    # 5. label: temperature 24 h ahead (exact timestamp match)
    out[TARGET] = t.shift(-24)

    # rows where the current temperature is unknown, or features are incomplete, are dropped
    out = out[t.notna()]
    out = out.dropna(subset=FEATURE_COLS)
    out.index.name = "ts"
    return out.reset_index()


def save_features(engine, feats: pd.DataFrame, location: str):
    cols = FEATURE_COLS + [TARGET]
    rows = [
        (location, r.ts.to_pydatetime(),
         *[None if pd.isna(getattr(r, c)) else float(getattr(r, c)) for c in cols],
         FEATURE_VERSION)
        for r in feats.itertuples(index=False)
    ]
    sql = f"""
        INSERT INTO features.weather_features (location, ts, {", ".join(cols)}, feature_version)
        VALUES %s
        ON CONFLICT (location, ts, feature_version) DO UPDATE SET
            {", ".join(f"{c} = EXCLUDED.{c}" for c in cols)},
            created_at = now()
    """
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)
        raw_conn.commit()
    finally:
        raw_conn.close()


def main():
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="recompute every row")
    parser.add_argument("--days", type=int, default=3, help="days to recompute (default 3)")
    args = parser.parse_args()
    location = os.getenv("LOCATION", "Bengaluru")

    engine = get_engine()
    run_id = start_run(engine, "build_features")
    try:
        if args.all:
            clean = load_clean(engine, location)
            keep_from = None
        else:
            keep_from = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=args.days)
            # extra 2 days of history so 24 h lags / rolling means are complete
            clean = load_clean(engine, location, since=keep_from - pd.Timedelta(days=2))

        feats = build_features(clean)
        if keep_from is not None and not feats.empty:
            feats = feats[feats["ts"] >= keep_from]

        save_features(engine, feats, location)

        n_train = int(feats[TARGET].notna().sum()) if not feats.empty else 0
        n_pred = len(feats) - n_train
        finish_run(engine, run_id, "success", len(clean), len(feats),
                   f"version={FEATURE_VERSION}; with_label={n_train}; latest_without_label={n_pred}")
        print(f"Clean rows read        : {len(clean)}")
        print(f"Feature rows saved     : {len(feats)}  (version {FEATURE_VERSION})")
        print(f"  with label (training): {n_train}")
        print(f"  latest, no label yet : {n_pred}")
    except Exception as e:
        finish_run(engine, run_id, "failed", notes=str(e)[:500])
        raise


if __name__ == "__main__":
    main()
