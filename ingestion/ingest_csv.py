"""
Ingestion of the curated CITY CATALOGUE (CSV) - the project's second data source.

    data/sources/cities.csv  (structured CSV, curated reference data)
      1. copy the file unchanged into the data lake   data/raw/source=city_catalogue/type=csv/...
      2. validate: schema, data types, ranges, allowed values, duplicates
      3. BRONZE      every row as received          -> raw.city_catalogue
      4. QUARANTINE  rows that failed validation    -> quarantine.location_rejects
      5. SILVER      valid rows (dimension table)    -> clean.locations
    The file's MD5 is its version: an unchanged file is skipped (use --force to reload).

Usage (from the project root):
    python -m ingestion.ingest_csv                 # load if the file changed
    python -m ingestion.ingest_csv --force         # always reload
    python -m ingestion.ingest_csv --dry-run       # validate only, no database
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from common.datalake import save_file_copy
from common.logger import get_logger

log = get_logger("ingestion")

DEFAULT_FILE = Path("data/sources/cities.csv")
SOURCE = "city_catalogue"

EXPECTED_COLUMNS = ["city", "country", "continent", "latitude", "longitude",
                    "timezone", "climate_zone", "role"]
REQUIRED_TEXT = ["city", "country", "continent", "timezone", "role"]
ALLOWED_ROLES = {"training", "monitored"}
ALLOWED_CONTINENTS = {"Asia", "Europe", "Africa", "North America", "South America",
                      "Oceania", "Antarctica"}
TIMEZONE_FORMAT = re.compile(r"^[A-Za-z]+(/[A-Za-z0-9_+\-]+)+$")    # e.g. Asia/Kolkata


class SchemaError(Exception):
    """The whole file has the wrong structure - nothing is loaded."""


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[dict, str]]]:
    """Return (valid rows, [(rejected row, reason), ...]). Raises SchemaError for a bad file."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        log.warning("ignoring unexpected columns: %s", extra)

    df = df[EXPECTED_COLUMNS].copy()
    for col in ["city", "country", "continent", "timezone", "climate_zone", "role"]:
        df[col] = df[col].astype("string").str.strip()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    rejects, keep = [], []
    seen = set()
    for _, row in df.iterrows():
        record = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        reason = None
        empty = [c for c in REQUIRED_TEXT if not record.get(c)]
        if empty:
            reason = f"missing value: {empty}"
        elif record["latitude"] is None or not -90 <= record["latitude"] <= 90:
            reason = "latitude missing or outside -90..90"
        elif record["longitude"] is None or not -180 <= record["longitude"] <= 180:
            reason = "longitude missing or outside -180..180"
        elif record["role"] not in ALLOWED_ROLES:
            reason = f"role must be one of {sorted(ALLOWED_ROLES)}"
        elif record["continent"] not in ALLOWED_CONTINENTS:
            reason = "unknown continent"
        elif not TIMEZONE_FORMAT.match(record["timezone"]):
            reason = "timezone not in Area/City format"
        elif record["city"].lower() in seen:
            reason = "duplicate city"
        if reason:
            rejects.append((record, reason))
        else:
            seen.add(record["city"].lower())
            keep.append(record)

    good = pd.DataFrame(keep, columns=EXPECTED_COLUMNS)
    if (good["role"] == "monitored").sum() != 1:
        log.warning("expected exactly one 'monitored' city, found %d",
                    int((good["role"] == "monitored").sum()))
    return good, rejects


def last_loaded_md5(engine) -> str | None:
    with engine.connect() as conn:
        return conn.execute(text(
            "SELECT file_md5 FROM raw.city_catalogue ORDER BY id DESC LIMIT 1")).scalar()


def save_bronze(engine, df_raw: pd.DataFrame, source_file: str, md5: str, lake_path: str) -> int:
    rows = [{"f": source_file, "m": md5, "p": lake_path, "n": i + 1,
             "r": json.dumps({k: (None if pd.isna(v) else v) for k, v in rec.items()}, default=str)}
            for i, rec in enumerate(df_raw.to_dict("records"))]
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO raw.city_catalogue (source_file, file_md5, lake_path, row_num, record) "
            "VALUES (:f, :m, :p, :n, CAST(:r AS JSONB))"), rows)
    return len(rows)


def save_rejects(engine, rejects: list[tuple[dict, str]], md5: str) -> None:
    if not rejects:
        return
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO quarantine.location_rejects (city, reason, record, file_md5) "
            "VALUES (:c, :reason, CAST(:r AS JSONB), :m)"),
            [{"c": rec.get("city"), "reason": reason, "m": md5,
              "r": json.dumps(rec, default=str)} for rec, reason in rejects])


def upsert_locations(engine, good: pd.DataFrame) -> int:
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO clean.locations
                (location, latitude, longitude, role, country, continent, timezone,
                 climate_zone, source, updated_at)
            VALUES (:city, :latitude, :longitude, :role, :country, :continent, :timezone,
                    :climate_zone, 'city_catalogue', now())
            ON CONFLICT (location) DO UPDATE SET
                latitude = EXCLUDED.latitude, longitude = EXCLUDED.longitude,
                role = EXCLUDED.role, country = EXCLUDED.country,
                continent = EXCLUDED.continent, timezone = EXCLUDED.timezone,
                climate_zone = EXCLUDED.climate_zone, source = EXCLUDED.source,
                updated_at = now()"""), good.to_dict("records"))
    return len(good)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=str(DEFAULT_FILE))
    parser.add_argument("--force", action="store_true", help="reload even if unchanged")
    parser.add_argument("--dry-run", action="store_true", help="validate only, no database")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        log.error("CSV ingestion FAILED: file not found: %s", path)
        raise SystemExit(1)
    md5 = file_md5(path)
    log.info("CSV ingestion started: %s (md5 %s)", path.as_posix(), md5[:10])

    df_raw = pd.read_csv(path, dtype=str)
    try:
        good, rejects = validate(df_raw)
    except SchemaError as e:
        log.error("CSV ingestion FAILED: schema mismatch - %s", e)
        raise SystemExit(1)
    for rec, reason in rejects:
        log.warning("rejected row city=%s: %s", rec.get("city"), reason)
    log.info("validation: %d rows read, %d valid, %d rejected", len(df_raw), len(good), len(rejects))

    if args.dry_run:
        log.info("dry run - nothing written")
        return

    from common.db import finish_run, get_engine, start_run   # imported here so --dry-run needs no DB
    engine = get_engine()
    if not args.force and last_loaded_md5(engine) == md5:
        log.info("CSV unchanged since last load (md5 %s) - skipped", md5[:10])
        return

    run_id = start_run(engine, "ingest_csv")
    try:
        lake_path, _ = save_file_copy(SOURCE, "csv", path, {"rows": len(df_raw)})
        log.info("raw file saved to data lake: %s", lake_path)
        n_raw = save_bronze(engine, df_raw, path.as_posix(), md5, lake_path)
        save_rejects(engine, rejects, md5)
        n_loc = upsert_locations(engine, good)
        finish_run(engine, run_id, "success", rows_in=n_raw, rows_out=n_loc,
                   notes=f"file_md5={md5}; rejected={len(rejects)}; lake={lake_path}")
        log.info("CSV ingestion SUCCESS: %d raw rows, %d locations upserted, %d quarantined",
                 n_raw, n_loc, len(rejects))
    except Exception as e:
        finish_run(engine, run_id, "failed", notes=str(e)[:500])
        log.exception("CSV ingestion FAILED: %s", e)
        raise


if __name__ == "__main__":
    main()
