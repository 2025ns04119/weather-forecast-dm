"""
Raw data lake on the file system (in addition to the bronze table in Neon).

Every file is stored exactly as received, in folders partitioned by
source, data type and ingestion date (UTC):

    data/raw/
      source=open_meteo/
        type=forecast/date=2026-11-16/bengaluru_131705.json
        type=archive/date=2026-11-10/london_101512.json
      source=city_catalogue/
        type=csv/date=2026-11-16/cities_131700.csv
      _manifest.csv          <- one line per file: when, source, type, path, size, md5

The lake folder can be changed with the DATA_LAKE_DIR environment variable.
"""
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

LAKE_DIR = Path(os.getenv("DATA_LAKE_DIR", "data/raw"))
MANIFEST = LAKE_DIR / "_manifest.csv"
MANIFEST_COLUMNS = ["ingested_at_utc", "source", "type", "name", "path",
                    "bytes", "md5", "details"]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "file"


def partition_dir(source: str, data_type: str, when: datetime) -> Path:
    return LAKE_DIR / f"source={source}" / f"type={data_type}" / f"date={when:%Y-%m-%d}"


def write_raw(source: str, data_type: str, name: str, content, ext: str,
              details: dict | None = None, when: datetime | None = None) -> tuple[str, str]:
    """
    Save one raw file unchanged and record it in the manifest.
    Returns (path as text, md5 of the content).
    """
    when = when or datetime.now(timezone.utc)
    folder = partition_dir(source, data_type, when)
    folder.mkdir(parents=True, exist_ok=True)

    data = content.encode("utf-8") if isinstance(content, str) else content
    path = folder / f"{_slug(name)}_{when:%H%M%S}.{ext}"
    path.write_bytes(data)
    md5 = hashlib.md5(data).hexdigest()

    new_file = not MANIFEST.exists()
    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(MANIFEST_COLUMNS)
        writer.writerow([when.strftime("%Y-%m-%d %H:%M:%S"), source, data_type, name,
                         path.as_posix(), len(data), md5, json.dumps(details or {})])
    return path.as_posix(), md5


def save_json(source: str, data_type: str, name: str, payload: dict,
              details: dict | None = None) -> tuple[str, str]:
    """Save an API response (dict) as a JSON file in the lake."""
    return write_raw(source, data_type, name, json.dumps(payload, separators=(",", ":")),
                     "json", details)


def save_file_copy(source: str, data_type: str, file_path: str | Path,
                   details: dict | None = None) -> tuple[str, str]:
    """Copy an input file (e.g. a CSV) into the lake, byte for byte."""
    file_path = Path(file_path)
    return write_raw(source, data_type, file_path.stem, file_path.read_bytes(),
                     file_path.suffix.lstrip(".") or "dat", details)
