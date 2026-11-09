"""
Feature engineering (SILVER -> GOLD), feature version v3 (location-agnostic, multi-target).

Reads clean.weather_hourly, builds ML features and upserts them into
features.weather_features. The SAME build_features() function is used by the
pipeline and by the dashboard's on-demand forecast (no training/serving skew).

Features (only past/current values -> no data leakage):
    hour                      local SOLAR hour (UTC hour + longitude/15) - works anywhere
    month, latitude           season + hemisphere/climate context
    temp_lag_1h/3h/24h        lag features
    temp_roll_mean_6h/24h     rolling means
    humidity, pressure, wind_speed, cloud_cover          current conditions
    humidity/cloud/wind_lag_24h, pressure_change_3h      weather memory & tendency
    rain_6h, rain_24h                                    recent rainfall (mm)
Labels (training only, added in training/train.py): temperature, humidity, wind,
cloud cover and rain (yes/no) 24*d hours ahead, d = 1..7.

Usage (from the project root):
    python -m features.build_features                    # monitored city, last 3 days
    python -m features.build_features --all              # monitored city, full history
    python -m features.build_features --all-locations    # every stored location, full history
"""
import argparse
import os

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values
from sqlalchemy import text

from common.db import finish_run, get_engine, start_run

FEATURE_VERSION = "v3"
MAX_INTERPOLATE_HOURS = 3   # fill gaps up to 3 h; longer gaps stay empty

FEATURE_COLS = [
    "hour", "month", "latitude",
    "temp_lag_1h", "temp_lag_3h", "temp_lag_24h",
    "temp_roll_mean_6h", "temp_roll_mean_24h",
    "humidity", "pressure", "wind_speed", "cloud_cover",
    "humidity_lag_24h", "cloud_lag_24h", "wind_lag_24h", "pressure_change_3h",
    "rain_6h", "rain_24h",
]
TARGET = "target_temp_24h"


def load_locations(engine, only: str | None = None) -> pd.DataFrame:
    sql = "SELECT location, latitude, longitude FROM clean.locations"
    params = {}
    if only:
        sql += " WHERE location = :loc"
        params["loc"] = only
    with engine.connect() as conn:
        return pd.read_sql(text(sql + " ORDER BY location"), conn, params=params)


def load_clean(engine, location: str, since=None) -> pd.DataFrame:
    sql = """SELECT ts, temperature_2m, relative_humidity_2m, pressure_msl,
                    wind_speed_10m, cloud_cover, precipitation
             FROM clean.weather_hourly WHERE location = :loc"""
    params = {"loc": location}
    if since is not None:
        sql += " AND ts >= :since"
        params["since"] = since
    with engine.connect() as conn:
        df = pd.read_sql(text(sql + " ORDER BY ts"), conn, params=params)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def build_features(clean: pd.DataFrame, latitude: float, longitude: float) -> pd.DataFrame:
    """Pure function: clean hourly data for ONE location -> feature table."""
    if clean.empty:
        return pd.DataFrame()

    # 1. regular hourly grid + short-gap interpolation
    df = clean.set_index("ts").sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    df = df.reindex(full)
    num = df.select_dtypes("number").columns
    df[num] = df[num].interpolate(method="time", limit=MAX_INTERPOLATE_HOURS, limit_area="inside")

    t = df["temperature_2m"]
    out = pd.DataFrame(index=df.index)

    # 2. time + place features (solar hour drives the daily temperature cycle)
    utc_hour = df.index.hour + df.index.minute / 60
    out["hour"] = np.floor((utc_hour + longitude / 15.0) % 24).astype(int)
    out["month"] = df.index.month
    out["latitude"] = float(latitude)

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

    # 5. weather memory, tendency and recent rain
    out["humidity_lag_24h"] = df["relative_humidity_2m"].shift(24)
    out["cloud_lag_24h"] = df["cloud_cover"].shift(24)
    out["wind_lag_24h"] = df["wind_speed_10m"].shift(24)
    out["pressure_change_3h"] = df["pressure_msl"] - df["pressure_msl"].shift(3)
    rain = df["precipitation"].fillna(0) if "precipitation" in df else pd.Series(0.0, index=df.index)
    out["rain_6h"] = rain.rolling(6, min_periods=6).sum()
    out["rain_24h"] = rain.rolling(24, min_periods=24).sum()

    # 6. day-1 temperature label kept in the table for convenience
    out[TARGET] = t.shift(-24)

    out = out[t.notna()].dropna(subset=FEATURE_COLS)
    out.index.name = "ts"
    return out.reset_index()


def save_features(engine, feats: pd.DataFrame, location: str):
    if feats.empty:
        return
    cols = FEATURE_COLS + [TARGET]
    rows = [(location, r.ts.to_pydatetime(),
             *[None if pd.isna(getattr(r, c)) else float(getattr(r, c)) for c in cols],
             FEATURE_VERSION)
            for r in feats.itertuples(index=False)]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="full history of the monitored city")
    parser.add_argument("--all-locations", action="store_true", help="full history of every location")
    parser.add_argument("--days", type=int, default=3, help="days to recompute (default 3)")
    args = parser.parse_args()
    monitored = os.getenv("LOCATION", "Bengaluru")

    engine = get_engine()
    run_id = start_run(engine, "build_features")
    try:
        locs = load_locations(engine, None if args.all_locations else monitored)
        if locs.empty:
            raise RuntimeError("No locations found - run infra/migrate_v2.sql and ingestion first")
        full = args.all or args.all_locations
        keep_from = None if full else pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=args.days)

        total_in = total_out = 0
        for loc in locs.itertuples(index=False):
            since = None if keep_from is None else keep_from - pd.Timedelta(days=2)
            clean = load_clean(engine, loc.location, since=since)
            feats = build_features(clean, loc.latitude, loc.longitude)
            if keep_from is not None and not feats.empty:
                feats = feats[feats["ts"] >= keep_from]
            save_features(engine, feats, loc.location)
            total_in += len(clean)
            total_out += len(feats)
            if len(locs) > 1:
                print(f"  {loc.location:<14} {len(feats):>7,} feature rows")

        finish_run(engine, run_id, "success", total_in, total_out,
                   f"version={FEATURE_VERSION}; locations={len(locs)}")
        print(f"Locations: {len(locs)} | clean rows read: {total_in:,} | "
              f"feature rows saved: {total_out:,} (version {FEATURE_VERSION})")
    except Exception as e:
        finish_run(engine, run_id, "failed", notes=str(e)[:500])
        raise


if __name__ == "__main__":
    main()
