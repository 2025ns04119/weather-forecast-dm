"""
Data quality metrics (profiling) computed with SQL - shared by the data quality report and the web app.
Checks: schema, rows per layer, missing values, duplicates, range violations, completeness, freshness, quarantine.
"""
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from pipeline.profiling_validation.clean_weather import COLUMNS, RANGES

EXPECTED_SCHEMA = ["location", "ts", *COLUMNS, "raw_id", "ingested_at"]


def q(engine, sql: str, **params) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def profile(engine) -> dict:
    """All data quality metrics except Great Expectations (see quality_report.py)."""
    out = {"generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}

    # 1. schema check
    cols = q(engine, """SELECT column_name FROM information_schema.columns
                        WHERE table_schema='clean' AND table_name='weather_hourly'""")["column_name"].tolist()
    out["schema"] = {"missing_columns": [c for c in EXPECTED_SCHEMA if c not in cols],
                     "unexpected_columns": [c for c in cols if c not in EXPECTED_SCHEMA]}

    # 2. row counts per layer
    out["row_counts"] = q(engine, """
        SELECT 'raw.weather_api' AS layer, COUNT(*) AS n FROM raw.weather_api
        UNION ALL SELECT 'raw.city_catalogue', COUNT(*) FROM raw.city_catalogue
        UNION ALL SELECT 'clean.weather_hourly', COUNT(*) FROM clean.weather_hourly
        UNION ALL SELECT 'clean.locations', COUNT(*) FROM clean.locations
        UNION ALL SELECT 'quarantine.weather_rejects', COUNT(*) FROM quarantine.weather_rejects
        UNION ALL SELECT 'quarantine.location_rejects', COUNT(*) FROM quarantine.location_rejects
        UNION ALL SELECT 'features.weather_features', COUNT(*) FROM features.weather_features
        """).set_index("layer")["n"].astype(int).to_dict()
    total = out["row_counts"]["clean.weather_hourly"]

    # 3. missing values per column
    nulls = q(engine, "SELECT " + ", ".join(f"COUNT(*) - COUNT({c}) AS {c}" for c in COLUMNS)
              + " FROM clean.weather_hourly").iloc[0].astype(int)
    out["missing_values"] = {c: {"count": int(nulls[c]), "pct": round(100 * nulls[c] / max(total, 1), 3)}
                             for c in COLUMNS}

    # 4. duplicates (primary key location+ts should make this 0)
    out["duplicates"] = int(q(engine, """SELECT COUNT(*) AS n FROM (SELECT location, ts FROM clean.weather_hourly
                                         GROUP BY location, ts HAVING COUNT(*) > 1) d""")["n"].iloc[0])

    # 5. range violations (should be 0 - bad rows are quarantined)
    rng = q(engine, "SELECT " + ", ".join(
        f"SUM(CASE WHEN {c} < {lo} OR {c} > {hi} THEN 1 ELSE 0 END) AS {c}" for c, (lo, hi) in RANGES.items())
        + " FROM clean.weather_hourly").iloc[0].fillna(0).astype(int)
    out["range_violations"] = {c: int(rng[c]) for c in RANGES}

    # 6. completeness and freshness per location
    comp = q(engine, """
        SELECT location, COUNT(*) AS hours, MIN(ts) AS first_ts, MAX(ts) AS last_ts,
               (EXTRACT(EPOCH FROM MAX(ts) - MIN(ts)) / 3600 + 1)::bigint AS expected
        FROM clean.weather_hourly GROUP BY location ORDER BY location""")
    comp["missing_hours"] = comp["expected"] - comp["hours"]
    comp["completeness_pct"] = (100 * comp["hours"] / comp["expected"]).round(2)
    comp["age_hours"] = ((pd.Timestamp.now(tz="UTC") - pd.to_datetime(comp["last_ts"], utc=True))
                         .dt.total_seconds() / 3600).round(1)
    out["completeness"] = comp[["location", "hours", "expected", "missing_hours",
                                "completeness_pct", "age_hours"]].to_dict("records")

    # 7. quarantine reasons
    out["quarantine_weather"] = q(engine, """SELECT reason, COUNT(*) AS n FROM quarantine.weather_rejects
                                             GROUP BY reason ORDER BY n DESC""").to_dict("records")
    out["quarantine_locations"] = q(engine, """SELECT reason, COUNT(*) AS n FROM quarantine.location_rejects
                                               GROUP BY reason ORDER BY n DESC""").to_dict("records")

    return out


def latest_gx_result(engine) -> str | None:
    """Great Expectations result of the latest validation run, e.g. 'GE 13/13 passed' (from the run log)."""
    notes = q(engine, """SELECT notes FROM ml.pipeline_runs WHERE pipeline = 'validate_clean'
                         AND status = 'success' AND notes LIKE '%GE %' ORDER BY run_id DESC LIMIT 1""")
    if notes.empty:
        return None
    part = [p.strip() for p in str(notes["notes"].iloc[0]).split(";") if p.strip().startswith("GE ")]
    return part[0] if part else None
