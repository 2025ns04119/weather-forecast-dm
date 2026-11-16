"""
Values the app shows that are CALCULATED by this project (no external service):

    solar_elevation()      sun angle for any place/time   (NOAA solar-position equations)
    sun_times()            sunrise / sunset               (NOAA sunrise equation)
    apparent_temperature() "feels like"                   (Steadman / Australian BoM formula)
    uv_index()             UV estimate                    (clear-sky UV from sun angle x cloud factor)
    condition()            condition text + icon          (rules on predicted cloud cover & rain chance)
"""
import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- sun position
def _solar_terms(ts_utc: pd.DatetimeIndex):
    """Fractional-year based NOAA approximations: declination (rad), equation of time (min)."""
    doy = ts_utc.dayofyear.to_numpy()
    hour = ts_utc.hour.to_numpy() + ts_utc.minute.to_numpy() / 60
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    return decl, eqtime, hour


def solar_elevation(lat: float, lon: float, ts_utc) -> np.ndarray:
    """Sun elevation angle in degrees (negative = below the horizon)."""
    ts = pd.DatetimeIndex(pd.to_datetime(ts_utc, utc=True))
    decl, eqtime, hour = _solar_terms(ts)
    true_solar_min = hour * 60 + eqtime + 4 * lon
    ha = np.radians(true_solar_min / 4 - 180)
    phi = math.radians(lat)
    cos_zen = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    return 90 - np.degrees(np.arccos(np.clip(cos_zen, -1, 1)))


def sun_times(lat: float, lon: float, day, tz: str):
    """(sunrise, sunset) as local naive Timestamps for a local calendar day; None in polar day/night."""
    noon_utc = pd.Timestamp(day).tz_localize(tz).tz_convert("UTC") + pd.Timedelta(hours=12)
    decl, eqtime, _ = _solar_terms(pd.DatetimeIndex([noon_utc]))
    phi = math.radians(lat)
    cos_ha = (math.cos(math.radians(90.833)) / (math.cos(phi) * math.cos(decl[0]))
              - math.tan(phi) * math.tan(decl[0]))
    if cos_ha < -1 or cos_ha > 1:
        return None, None
    ha = math.degrees(math.acos(cos_ha))
    day_utc = pd.Timestamp(day).tz_localize("UTC")
    rise = day_utc + pd.Timedelta(minutes=720 - 4 * (lon + ha) - eqtime[0])
    sset = day_utc + pd.Timedelta(minutes=720 - 4 * (lon - ha) - eqtime[0])
    # the local date of these UTC times can differ by a day for far east/west places
    rise_l = rise.tz_convert(tz).tz_localize(None)
    set_l = sset.tz_convert(tz).tz_localize(None)
    shift = pd.Timedelta(days=(pd.Timestamp(day).date() - rise_l.date()).days)
    return rise_l + shift, set_l + shift


# ---------------------------------------------------------------- comfort & UV
def apparent_temperature(temp_c, rh_pct, wind_kmh):
    """'Feels like' (Steadman apparent temperature, as used by the Australian BoM)."""
    t = np.asarray(temp_c, dtype=float)
    e = np.asarray(rh_pct, dtype=float) / 100 * 6.105 * np.exp(17.27 * t / (237.7 + t))
    ws = np.asarray(wind_kmh, dtype=float) / 3.6
    return t + 0.33 * e - 0.70 * ws - 4.00


def uv_index(elevation_deg, cloud_pct):
    """
    UV index estimate: clear-sky UVI = 12.5 * cos(zenith)^2.42 (typical ozone),
    times a cloud modification factor 1 - 0.75 * (cloud fraction)^3.4.
    """
    cos_zen = np.clip(np.sin(np.radians(np.asarray(elevation_deg, dtype=float))), 0, 1)
    clear = 12.5 * cos_zen ** 2.42
    cmf = 1 - 0.75 * (np.clip(np.asarray(cloud_pct, dtype=float), 0, 100) / 100) ** 3.4
    return clear * cmf


def uv_level(uv: float) -> str:
    return ("Low" if uv < 3 else "Moderate" if uv < 6 else "High" if uv < 8
            else "Very high" if uv < 11 else "Extreme")


# ---------------------------------------------------------------- condition
def condition(cloud_pct: float, rain_prob: float, is_day: bool, rain_mm: float | None = None):
    """
    Returns (icon_kind, label). Forecast hours use rain probability (0-1);
    observed hours can pass the measured rain (mm) instead.
    """
    if rain_mm is not None and not pd.isna(rain_mm):
        if rain_mm >= 2.5:
            return "rain", "Heavy rain"
        if rain_mm >= 0.1:
            return "rain", "Light rain"
    elif rain_prob is not None and not pd.isna(rain_prob):
        if rain_prob >= 0.6:
            return "rain", "Rain likely"
        if rain_prob >= 0.35:
            return "rain", "Chance of rain"
    c = 0 if cloud_pct is None or pd.isna(cloud_pct) else cloud_pct
    if c >= 85:
        return "cloud", "Cloudy"
    if c >= 50:
        return ("partly" if is_day else "moonpartly"), "Partly cloudy"
    if c >= 20:
        return ("partly" if is_day else "moonpartly"), "Mostly clear"
    return ("sun" if is_day else "moon"), ("Sunny" if is_day else "Clear")


def compass(deg) -> str:
    if deg is None or pd.isna(deg):
        return ""
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((float(deg) + 22.5) // 45) % 8]
