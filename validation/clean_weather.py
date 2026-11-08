"""
Validation + cleaning (BRONZE -> SILVER).

1. Read raw JSON payloads from raw.weather_api that have not been processed yet
2. Unpack them into one row per hour
3. Row-level quality rules  -> bad rows go to quarantine.weather_rejects
4. Batch-level checks with Great Expectations -> summary logged
5. Good rows are upserted into clean.weather_hourly (with raw_id for lineage)

Usage (from the project root):
    python -m validation.clean_weather          # only new raw payloads
    python -m validation.clean_weather --all    # reprocess every raw payload
"""
import argparse
import json

import great_expectations as gx
import pandas as pd
from psycopg2.extras import execute_values
from sqlalchemy import text

from common.db import finish_run, get_engine, start_run

COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]

# Valid physical ranges (min, max) - the data quality rules
RANGES = {
    "temperature_2m": (-10, 50),        # deg C (Bengaluru-safe bounds)
    "relative_humidity_2m": (0, 100),   # %
    "dew_point_2m": (-30, 40),          # deg C
    "pressure_msl": (870, 1085),        # hPa
    "cloud_cover": (0, 100),            # %
    "wind_speed_10m": (0, 200),         # km/h
    "wind_direction_10m": (0, 360),     # degrees
    "precipitation": (0, 300),          # mm per hour
}

REQUIRED = ["temperature_2m", "relative_humidity_2m", "pressure_msl"]


# ---------------------------------------------------------------- extract
def load_raw(engine, reprocess_all: bool) -> list[dict]:
    sql = "SELECT id, location, endpoint, payload FROM raw.weather_api"
    if not reprocess_all:
        sql += """
            WHERE NOT EXISTS (SELECT 1 FROM clean.weather_hourly c WHERE c.raw_id = raw.weather_api.id)
              AND NOT EXISTS (SELECT 1 FROM quarantine.weather_rejects q WHERE q.raw_id = raw.weather_api.id)
        """
    sql += " ORDER BY id"
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text(sql))]


def payload_to_df(raw: dict) -> pd.DataFrame:
    payload = raw["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    hourly = payload.get("hourly", {})
    df = pd.DataFrame(hourly)
    if df.empty:
        return df
    df = df.rename(columns={"time": "ts"})
    for c in COLUMNS:                       # make sure every column exists
        if c not in df.columns:
            df[c] = None
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df[COLUMNS] = df[COLUMNS].apply(pd.to_numeric, errors="coerce")
    df["location"] = raw["location"]
    df["raw_id"] = raw["id"]
    df["endpoint"] = raw["endpoint"]
    return df[["location", "ts", *COLUMNS, "raw_id", "endpoint"]]


# ---------------------------------------------------------------- row rules
def apply_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (good_rows, rejected_rows_with_reason)."""
    reasons = pd.Series("", index=df.index)

    def flag(mask, reason):
        nonlocal reasons
        reasons = reasons.where(~mask, reasons + reason + "; ")

    flag(df["ts"].isna(), "missing timestamp")
    for c in REQUIRED:
        flag(df[c].isna(), f"{c} is null")
    for c, (lo, hi) in RANGES.items():
        flag(df[c].notna() & ~df[c].between(lo, hi), f"{c} out of range [{lo},{hi}]")

    bad = reasons != ""
    rejected = df[bad].copy()
    rejected["reason"] = reasons[bad].str.rstrip("; ")
    return df[~bad].copy(), rejected


def drop_future_and_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Live payloads contain forecast hours: keep only observed hours (<= now)."""
    now = pd.Timestamp.now(tz="UTC")
    future = df["ts"] > now
    n_future = int(future.sum())
    df = df[~future]
    before = len(df)
    df = df.sort_values("raw_id").drop_duplicates(["location", "ts"], keep="last")
    return df, n_future, before - len(df)


# ---------------------------------------------------------------- GE checks
def run_great_expectations(df: pd.DataFrame) -> dict:
    """Batch-level data quality report using Great Expectations."""
    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas("weather_source")
    asset = source.add_dataframe_asset("clean_weather")
    batch_def = asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name="weather_clean_suite")
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="ts"))
    suite.add_expectation(
        gx.expectations.ExpectCompoundColumnsToBeUnique(column_list=["location", "ts"]))
    for c, (lo, hi) in RANGES.items():
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
            column=c, min_value=lo, max_value=hi))
    for c in REQUIRED:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=c))
    suite = context.suites.add(suite)

    result = batch.validate(suite)
    stats = result.statistics
    return {
        "success": bool(result.success),
        "evaluated": stats["evaluated_expectations"],
        "passed": stats["successful_expectations"],
        "failed": stats["unsuccessful_expectations"],
    }


def count_missing_hours(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    full = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
    return int(len(full) - df["ts"].nunique())


# ---------------------------------------------------------------- load
def save_clean(engine, df: pd.DataFrame):
    rows = [
        (r.location, r.ts.to_pydatetime(), *[None if pd.isna(v) else float(v)
                                             for v in (getattr(r, c) for c in COLUMNS)],
         int(r.raw_id))
        for r in df.itertuples(index=False)
    ]
    sql = f"""
        INSERT INTO clean.weather_hourly (location, ts, {", ".join(COLUMNS)}, raw_id)
        VALUES %s
        ON CONFLICT (location, ts) DO UPDATE SET
            {", ".join(f"{c} = EXCLUDED.{c}" for c in COLUMNS)},
            raw_id = EXCLUDED.raw_id,
            ingested_at = now()
    """
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)
        raw_conn.commit()
    finally:
        raw_conn.close()


def save_rejects(engine, rejected: pd.DataFrame):
    if rejected.empty:
        return
    rows = []
    for r in rejected.to_dict("records"):
        record = {k: (None if pd.isna(v) else (v.isoformat() if hasattr(v, "isoformat") else v))
                  for k, v in r.items() if k != "reason"}
        ts = None if pd.isna(r["ts"]) else r["ts"].to_pydatetime()
        rows.append((r["location"], ts, r["reason"], json.dumps(record), int(r["raw_id"])))
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO quarantine.weather_rejects (location, ts, reason, record, raw_id)
                VALUES %s""", rows, page_size=1000)
        raw_conn.commit()
    finally:
        raw_conn.close()


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="reprocess all raw payloads")
    args = parser.parse_args()

    engine = get_engine()
    run_id = start_run(engine, "validate_clean")
    try:
        raws = load_raw(engine, args.all)
        if not raws:
            finish_run(engine, run_id, "success", 0, 0, "no new raw payloads")
            print("No new raw payloads to process.")
            return

        df = pd.concat([payload_to_df(r) for r in raws], ignore_index=True)
        rows_in = len(df)
        print(f"Raw payloads: {len(raws)}  |  hourly rows: {rows_in}")

        good, rejected = apply_rules(df)
        good, n_future, n_dupes = drop_future_and_duplicates(good)

        ge = run_great_expectations(good)
        gaps = count_missing_hours(good)

        save_rejects(engine, rejected)
        save_clean(engine, good)

        notes = (f"future_dropped={n_future}; duplicates_dropped={n_dupes}; "
                 f"rejected={len(rejected)}; missing_hours={gaps}; "
                 f"GE {ge['passed']}/{ge['evaluated']} passed")
        finish_run(engine, run_id, "success", rows_in, len(good), notes)

        print(f"Future (forecast) hours dropped : {n_future}")
        print(f"Duplicates dropped              : {n_dupes}")
        print(f"Rejected -> quarantine          : {len(rejected)}")
        print(f"Clean rows saved                : {len(good)}")
        print(f"Missing hours (gaps)            : {gaps}")
        print(f"Great Expectations              : {ge['passed']}/{ge['evaluated']} passed"
              f"  ->  {'PASS' if ge['success'] else 'FAIL'}")
    except Exception as e:
        finish_run(engine, run_id, "failed", notes=str(e)[:500])
        raise


if __name__ == "__main__":
    main()
