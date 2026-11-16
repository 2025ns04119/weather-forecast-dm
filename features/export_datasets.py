"""
Export the transformed layers to files so they can be versioned with DVC.

    data/prepared/clean_weather_hourly.parquet      silver layer (prepared dataset)
    data/features/weather_features_<version>.parquet gold layer (feature store, offline)
    data/<folder>/_metadata.json                     source table, rows, export time, md5

Usage (from the project root):
    python -m features.export_datasets
Then version them:  dvc add data/raw data/prepared data/features
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from common.db import finish_run, get_engine, start_run
from common.logger import get_logger
from features.build_features import FEATURE_VERSION

log = get_logger("export")


def export(engine, sql: str, path: Path, source: str, transformation: str) -> dict:
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    meta = {"file": path.name, "source_table": source, "transformation": transformation,
            "rows": len(df), "columns": list(df.columns),
            "exported_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "md5": hashlib.md5(path.read_bytes()).hexdigest()}
    (path.parent / "_metadata.json").write_text(json.dumps(meta, indent=2))
    log.info("exported %s: %d rows -> %s (md5 %s)", source, len(df), path.as_posix(), meta["md5"][:10])
    return meta


def main():
    engine = get_engine()
    run_id = start_run(engine, "export_datasets")
    try:
        a = export(engine, "SELECT * FROM clean.weather_hourly ORDER BY location, ts",
                   Path("data/prepared/clean_weather_hourly.parquet"), "clean.weather_hourly",
                   "validation + cleaning (validation/clean_weather.py)")
        b = export(engine, f"SELECT * FROM features.weather_features WHERE feature_version = '{FEATURE_VERSION}' "
                           "ORDER BY location, ts",
                   Path(f"data/features/weather_features_{FEATURE_VERSION}.parquet"), "features.weather_features",
                   f"feature engineering {FEATURE_VERSION} (features/build_features.py)")
        finish_run(engine, run_id, "success", rows_out=a["rows"] + b["rows"],
                   notes=f"prepared md5={a['md5']}; features md5={b['md5']}")
    except Exception as e:
        finish_run(engine, run_id, "failed", notes=str(e)[:500])
        log.exception("export FAILED: %s", e)
        raise


if __name__ == "__main__":
    main()
