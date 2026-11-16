"""
EDA results computed with SQL aggregates in the database - used by the web app (Project insights -> EDA tab).
Same analyses as eda.py (distributions, correlations, seasonality, daily cycle, rain, completeness, features).
"""
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from pipeline.feature_engineering.build_features import FEATURE_COLS, FEATURE_VERSION, TARGET

VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl",
        "cloud_cover", "wind_speed_10m", "wind_direction_10m", "precipitation"]
BINS = 40


def q(engine, sql: str, **params) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def overview(engine) -> dict:
    head = q(engine, """SELECT COUNT(*) AS rows, COUNT(DISTINCT location) AS locations,
                               MIN(ts) AS first_ts, MAX(ts) AS last_ts FROM clean.weather_hourly""").iloc[0]
    stats = q(engine, " UNION ALL ".join(
        f"""SELECT '{v}' AS variable, AVG({v}) AS mean, STDDEV({v}) AS std, MIN({v}) AS min,
                   MAX({v}) AS max, COUNT(*) - COUNT({v}) AS missing FROM clean.weather_hourly""" for v in VARS))
    return {"rows": int(head["rows"]), "locations": int(head["locations"]),
            "first_ts": str(head["first_ts"])[:10], "last_ts": str(head["last_ts"])[:10],
            "stats": stats.round(2)}


def histogram(engine, var: str) -> pd.DataFrame:
    """Counts in equal-width bins (precipitation: only hours with rain > 0)."""
    where = f"{var} > 0" if var == "precipitation" else f"{var} IS NOT NULL"
    return q(engine, f"""
        WITH b AS (SELECT MIN({var}) AS lo, MAX({var}) AS hi FROM clean.weather_hourly WHERE {where})
        SELECT LEAST(width_bucket({var}, b.lo, b.hi + 1e-9, {BINS}), {BINS}) AS bin,
               b.lo + (LEAST(width_bucket({var}, b.lo, b.hi + 1e-9, {BINS}), {BINS}) - 0.5)
                      * (b.hi - b.lo) / {BINS} AS mid,
               COUNT(*) AS n
        FROM clean.weather_hourly, b WHERE {where} GROUP BY 1, 2 ORDER BY 1""")


def correlation(engine) -> pd.DataFrame:
    pairs = [(a, b) for i, a in enumerate(VARS) for b in VARS[i + 1:]]
    row = q(engine, "SELECT " + ", ".join(f"CORR({a}, {b}) AS c{k}" for k, (a, b) in enumerate(pairs))
            + " FROM clean.weather_hourly").iloc[0]
    m = pd.DataFrame(1.0, index=VARS, columns=VARS)
    for k, (a, b) in enumerate(pairs):
        m.loc[a, b] = m.loc[b, a] = row[f"c{k}"]
    return m.astype(float).round(2)


def monthly_by_climate(engine) -> pd.DataFrame:
    return q(engine, """
        SELECT COALESCE(l.climate_zone, 'Unknown') AS climate_zone, EXTRACT(MONTH FROM c.ts)::int AS month,
               AVG(c.temperature_2m) AS temperature
        FROM clean.weather_hourly c LEFT JOIN clean.locations l USING (location)
        GROUP BY 1, 2 ORDER BY 1, 2""")


def daily_cycle(engine, location: str = "Bengaluru", tz: str = "Asia/Kolkata") -> pd.DataFrame:
    return q(engine, """
        SELECT EXTRACT(HOUR FROM ts AT TIME ZONE :tz)::int AS local_hour,
               AVG(temperature_2m) AS temperature, AVG(relative_humidity_2m) AS humidity
        FROM clean.weather_hourly WHERE location = :loc GROUP BY 1 ORDER BY 1""", loc=location, tz=tz)


def rain_share(engine) -> pd.DataFrame:
    return q(engine, """
        SELECT location, 100.0 * AVG(CASE WHEN precipitation >= 0.1 THEN 1 ELSE 0 END) AS rainy_pct
        FROM clean.weather_hourly GROUP BY location ORDER BY rainy_pct""")


def completeness(engine) -> pd.DataFrame:
    """Share of expected hourly rows present, per location and month."""
    df = q(engine, """SELECT location, TO_CHAR(ts AT TIME ZONE 'UTC', 'YYYY-MM') AS ym, COUNT(*) AS n
                      FROM clean.weather_hourly GROUP BY 1, 2""")
    have = df.pivot(index="location", columns="ym", values="n").fillna(0).sort_index(axis=1)
    hours = pd.to_datetime(have.columns + "-01").days_in_month.values * 24
    return (have / hours * 100).clip(upper=100).round(1)


def feature_target(engine, location: str = "Bengaluru") -> pd.Series:
    """Correlation of each gold feature with the temperature 24 h later."""
    row = q(engine, "SELECT " + ", ".join(f"CORR({f}, {TARGET}) AS {f}" for f in FEATURE_COLS)
            + " FROM features.weather_features WHERE feature_version = :v AND location = :loc",
            v=FEATURE_VERSION, loc=location).iloc[0]
    return row.astype(float).dropna().sort_values()


def collect(engine) -> dict:
    """Everything the EDA tab shows."""
    return {"generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "overview": overview(engine),
            "histograms": {v: histogram(engine, v) for v in VARS},
            "correlation": correlation(engine),
            "monthly": monthly_by_climate(engine),
            "daily": daily_cycle(engine),
            "rain": rain_share(engine),
            "completeness": completeness(engine),
            "feature_target": feature_target(engine),
            "feature_version": FEATURE_VERSION}
