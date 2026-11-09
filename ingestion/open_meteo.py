"""
Open-Meteo API client (no database code - used by the pipeline AND the dashboard).

Endpoints (free, no API key, CC BY 4.0):
    forecast   https://api.open-meteo.com/v1/forecast          recent + current weather
    archive    https://archive-api.open-meteo.com/v1/archive   history (ERA5 reanalysis)
    geocoding  https://geocoding-api.open-meteo.com/v1/search  place name -> coordinates
"""
import time
from datetime import date, timedelta

import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl",
    "cloud_cover", "wind_speed_10m", "wind_direction_10m", "precipitation",
]
CURRENT_VARS = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "weather_code", "wind_speed_10m", "is_day", "precipitation",
]


def call_api(url: str, params: dict, retries: int = 3, timeout: int = 60) -> dict:
    """GET with simple retry + backoff."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def fetch_recent(lat: float, lon: float, past_days: int = 7, with_current: bool = False) -> dict:
    """Observed hours for the last `past_days` days (+ today), times in UTC."""
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(HOURLY_VARS),
              "past_days": past_days, "forecast_days": 1, "timezone": "UTC"}
    if with_current:
        params["current"] = ",".join(CURRENT_VARS)
    return call_api(FORECAST_URL, params)


def fetch_history(lat: float, lon: float, days: int) -> dict:
    """Historical hourly data (archive lags real time by about 5 days)."""
    end = date.today() - timedelta(days=6)
    start = end - timedelta(days=days)
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(HOURLY_VARS),
              "start_date": start.isoformat(), "end_date": end.isoformat(), "timezone": "UTC"}
    return call_api(ARCHIVE_URL, params, timeout=120)


def geocode(query: str, count: int = 8) -> list[dict]:
    """Search places by name. Returns [{name, admin1, country, latitude, longitude, timezone, ...}]."""
    if not query or len(query.strip()) < 2:
        return []
    data = call_api(GEOCODING_URL, {"name": query.strip(), "count": count,
                                    "language": "en", "format": "json"}, retries=2, timeout=15)
    return data.get("results", []) or []
