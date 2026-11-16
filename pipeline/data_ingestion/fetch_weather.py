"""
Ingestion (BRONZE layer): pull weather data from the Open-Meteo REST API.

Each API response is stored unchanged in TWO places:
  - the data lake on disk   data/raw/source=open_meteo/type=<forecast|archive>/date=YYYY-MM-DD/
  - the bronze table        raw.weather_api   (with lake_path for lineage)
Locations come from clean.locations, which is loaded from the city catalogue CSV
(pipeline/data_ingestion/ingest_csv.py); pipeline/data_collection/locations.py is only a fallback.

Usage (run from the project root):
    python -m pipeline.data_ingestion.fetch_weather --mode live                 # monitored city, last 7 days
    python -m pipeline.data_ingestion.fetch_weather --mode backfill --days 730  # monitored city history
    python -m pipeline.data_ingestion.fetch_weather --mode corpus --days 365    # all catalogue cities
"""
import argparse
import json
import os
import time

from sqlalchemy import text

from pipeline.raw_data_storage.datalake import save_json
from pipeline.common.db import finish_run, get_engine, start_run
from pipeline.common.logger import get_logger
from pipeline.data_collection.locations import TRAINING_CITIES
from pipeline.data_collection.open_meteo import fetch_history, fetch_recent

log = get_logger("ingestion")

LAT = float(os.getenv("LAT", "12.97"))
LON = float(os.getenv("LON", "77.59"))
LOCATION = os.getenv("LOCATION", "Bengaluru")


def save_raw(engine, location: str, endpoint: str, payload: dict, lake_path: str) -> int:
    with engine.begin() as conn:
        return conn.execute(
            text("INSERT INTO raw.weather_api (location, endpoint, payload, lake_path) "
                 "VALUES (:loc, :ep, CAST(:payload AS JSONB), :lake) RETURNING id"),
            {"loc": location, "ep": endpoint, "payload": json.dumps(payload), "lake": lake_path},
        ).scalar()


def upsert_location(engine, location: str, lat: float, lon: float,
                    elevation=None, role: str = "training"):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO clean.locations (location, latitude, longitude, elevation, role)
            VALUES (:loc, :lat, :lon, :elev, :role)
            ON CONFLICT (location) DO UPDATE SET
                elevation = COALESCE(EXCLUDED.elevation, clean.locations.elevation),
                role = CASE WHEN clean.locations.role = 'monitored' THEN 'monitored'
                            ELSE EXCLUDED.role END"""),
            {"loc": location, "lat": lat, "lon": lon, "elev": elevation, "role": role})


def catalogue_cities(engine) -> list[tuple[str, float, float, str]]:
    """Cities from the catalogue table; falls back to pipeline/data_collection/locations.py if empty."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT location, latitude, longitude, role FROM clean.locations "
                "WHERE role IN ('training', 'monitored') ORDER BY location")).all()
    except Exception:
        rows = []
    if rows:
        return [(r[0], float(r[1]), float(r[2]), r[3]) for r in rows]
    log.warning("clean.locations is empty - using pipeline/data_collection/locations.py instead")
    return [(n, la, lo, "monitored" if n == LOCATION else "training")
            for n, la, lo in TRAINING_CITIES]


def ingest_one(engine, location, lat, lon, mode, days, role) -> tuple[int, int]:
    if mode == "live":
        endpoint, payload = "forecast", fetch_recent(lat, lon, past_days=7)
    else:
        endpoint, payload = "archive", fetch_history(lat, lon, days)
    n_hours = len(payload.get("hourly", {}).get("time", []))
    lake_path, md5 = save_json("open_meteo", endpoint, location, payload,
                               {"lat": lat, "lon": lon, "hours": n_hours, "mode": mode})
    upsert_location(engine, location, lat, lon, payload.get("elevation"), role)
    raw_id = save_raw(engine, location, endpoint, payload, lake_path)
    log.info("saved %s %s: %d hours | raw_id=%s | lake=%s", location, endpoint, n_hours, raw_id, lake_path)
    return raw_id, n_hours


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "backfill", "corpus"], default="live")
    parser.add_argument("--days", type=int, default=730)
    args = parser.parse_args()

    engine = get_engine()
    run_id = start_run(engine, f"ingest_{args.mode}")
    log.info("API ingestion started: mode=%s run_id=%s", args.mode, run_id)
    try:
        if args.mode == "corpus":
            cities = catalogue_cities(engine)
            total, ids, failed = 0, [], []
            for i, (name, lat, lon, role) in enumerate(cities, start=1):
                try:
                    raw_id, n = ingest_one(engine, name, lat, lon, "backfill", args.days, role)
                    total += n
                    ids.append(raw_id)
                except Exception as e:                 # one bad city must not stop the rest
                    failed.append(name)
                    log.error("[%d/%d] %s FAILED: %s", i, len(cities), name, e)
                time.sleep(1)   # be polite to the free API
            status = "success" if not failed else ("partial" if ids else "failed")
            finish_run(engine, run_id, status, total, len(ids),
                       f"cities_ok={len(ids)}; failed={failed}")
            log.info("API ingestion %s: %d cities, %d hourly records, %d failed",
                     status.upper(), len(ids), total, len(failed))
            if not ids:
                raise RuntimeError("no city could be ingested")
        else:
            raw_id, n = ingest_one(engine, LOCATION, LAT, LON, args.mode, args.days, "monitored")
            finish_run(engine, run_id, "success", rows_in=n, rows_out=1, notes=f"raw_id={raw_id}")
            log.info("API ingestion SUCCESS: %s raw_id=%s, %d hourly records", LOCATION, raw_id, n)
    except Exception as e:
        finish_run(engine, run_id, "failed", notes=str(e)[:500])
        log.exception("API ingestion FAILED: %s", e)
        raise


if __name__ == "__main__":
    main()
