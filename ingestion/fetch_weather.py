"""
Ingestion (BRONZE layer): pull weather data from Open-Meteo and store the raw
JSON payload, unchanged, in raw.weather_api. Locations are registered in the
clean.locations dimension table.

Usage (run from the project root):
    python -m ingestion.fetch_weather --mode live                 # monitored city, last 7 days
    python -m ingestion.fetch_weather --mode backfill --days 730  # monitored city history
    python -m ingestion.fetch_weather --mode corpus --days 365    # all training cities (global model)
"""
import argparse
import json
import os
import time

from sqlalchemy import text

from common.db import finish_run, get_engine, start_run
from config.locations import TRAINING_CITIES
from ingestion.open_meteo import fetch_history, fetch_recent

LAT = float(os.getenv("LAT", "12.97"))
LON = float(os.getenv("LON", "77.59"))
LOCATION = os.getenv("LOCATION", "Bengaluru")


def save_raw(engine, location: str, endpoint: str, payload: dict) -> int:
    with engine.begin() as conn:
        return conn.execute(
            text("INSERT INTO raw.weather_api (location, endpoint, payload) "
                 "VALUES (:loc, :ep, CAST(:payload AS JSONB)) RETURNING id"),
            {"loc": location, "ep": endpoint, "payload": json.dumps(payload)},
        ).scalar()


def upsert_location(engine, location: str, lat: float, lon: float,
                    elevation=None, role: str = "training"):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO clean.locations (location, latitude, longitude, elevation, role)
            VALUES (:loc, :lat, :lon, :elev, :role)
            ON CONFLICT (location) DO UPDATE SET
                latitude = EXCLUDED.latitude, longitude = EXCLUDED.longitude,
                elevation = COALESCE(EXCLUDED.elevation, clean.locations.elevation),
                role = CASE WHEN clean.locations.role = 'monitored' THEN 'monitored'
                            ELSE EXCLUDED.role END"""),
            {"loc": location, "lat": lat, "lon": lon, "elev": elevation, "role": role})


def ingest_one(engine, location, lat, lon, mode, days, role) -> tuple[int, int]:
    if mode == "live":
        endpoint, payload = "forecast", fetch_recent(lat, lon, past_days=7)
    else:
        endpoint, payload = "archive", fetch_history(lat, lon, days)
    upsert_location(engine, location, lat, lon, payload.get("elevation"), role)
    n_hours = len(payload.get("hourly", {}).get("time", []))
    return save_raw(engine, location, endpoint, payload), n_hours


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "backfill", "corpus"], default="live")
    parser.add_argument("--days", type=int, default=730)
    args = parser.parse_args()

    engine = get_engine()
    run_id = start_run(engine, f"ingest_{args.mode}")
    try:
        if args.mode == "corpus":
            total, ids = 0, []
            for i, (name, lat, lon) in enumerate(TRAINING_CITIES, start=1):
                role = "monitored" if name == LOCATION else "training"
                raw_id, n = ingest_one(engine, name, lat, lon, "backfill", args.days, role)
                total += n
                ids.append(raw_id)
                print(f"[{i:>2}/{len(TRAINING_CITIES)}] {name:<14} raw_id={raw_id}  {n} hours")
                time.sleep(1)   # be polite to the free API
            finish_run(engine, run_id, "success", total, len(ids),
                       f"cities={len(ids)}; raw_ids={ids[0]}-{ids[-1]}")
            print(f"\nCorpus ingested: {len(ids)} cities, {total:,} hourly records")
        else:
            raw_id, n = ingest_one(engine, LOCATION, LAT, LON, args.mode, args.days, "monitored")
            finish_run(engine, run_id, "success", rows_in=n, rows_out=1, notes=f"raw_id={raw_id}")
            print(f"Saved raw payload id={raw_id} with {n} hourly records")
    except Exception as e:
        finish_run(engine, run_id, "failed", notes=str(e)[:500])
        raise


if __name__ == "__main__":
    main()
