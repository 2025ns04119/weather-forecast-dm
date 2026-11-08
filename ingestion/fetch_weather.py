"""
Ingestion (BRONZE layer): pull weather data from Open-Meteo and store the raw
JSON payload, unchanged, in raw.weather_api.

Usage (run from the project root):
    python -m ingestion.fetch_weather --mode live              # last 2 days + today
    python -m ingestion.fetch_weather --mode backfill --days 730   # history for training
"""
import argparse
import json
import os
import time
from datetime import date, timedelta
import requests
from sqlalchemy import text
from common.db import finish_run, get_engine, start_run

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]

LAT = float(os.getenv("LAT", "12.97"))
LON = float(os.getenv("LON", "77.59"))
LOCATION = os.getenv("LOCATION", "Bengaluru")


def call_api(url: str, params: dict, retries: int = 3) -> dict:
    """GET with simple retry + backoff."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def fetch_live() -> tuple[str, dict]:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(HOURLY_VARS),
        "past_days": 7,
        "forecast_days": 1,
        "timezone": "UTC",
    }
    return "forecast", call_api(FORECAST_URL, params)


def fetch_backfill(days: int) -> tuple[str, dict]:
    end = date.today() - timedelta(days=6)        # archive lags by ~5 days
    start = end - timedelta(days=days)
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(HOURLY_VARS),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "UTC",
    }
    print(f"Backfilling {start} -> {end}")
    return "archive", call_api(ARCHIVE_URL, params)


def save_raw(engine, endpoint: str, payload: dict) -> int:
    with engine.begin() as conn:
        return conn.execute(
            text("INSERT INTO raw.weather_api (location, endpoint, payload) "
                 "VALUES (:loc, :ep, CAST(:payload AS JSONB)) RETURNING id"),
            {"loc": LOCATION, "ep": endpoint, "payload": json.dumps(payload)},
        ).scalar()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "backfill"], default="live")
    parser.add_argument("--days", type=int, default=730)
    args = parser.parse_args()

    engine = get_engine()
    run_id = start_run(engine, f"ingest_{args.mode}")
    try:
        if args.mode == "live":
            endpoint, payload = fetch_live()
        else:
            endpoint, payload = fetch_backfill(args.days)

        n_hours = len(payload.get("hourly", {}).get("time", []))
        raw_id = save_raw(engine, endpoint, payload)

        finish_run(engine, run_id, "success", rows_in=n_hours, rows_out=1,
                   notes=f"raw_id={raw_id}")
        print(f"Saved raw payload id={raw_id} with {n_hours} hourly records")
    except Exception as e:
        finish_run(engine, run_id, "failed", notes=str(e)[:500])
        raise


if __name__ == "__main__":
    main()
