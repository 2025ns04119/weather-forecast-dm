# Raw data storage and data sources

## 1. Data sources (two data types)

| # | Source | Type | Format | Ingestion script | Frequency |
|---|---|---|---|---|---|
| 1 | Open-Meteo REST API (forecast + archive endpoints) | Semi-structured | JSON | `pipeline/data_ingestion/fetch_weather.py` | Every hour (live); once (backfill / corpus) |
| 2 | City catalogue, curated reference data | Structured | CSV (`data/sources/cities.csv`) | `pipeline/data_ingestion/ingest_csv.py` | Every hour; reloaded only when the file changes (MD5) |

## 2. Raw data lake (file system)

Every file is stored **exactly as received**, partitioned by **source**, **type** and **ingestion date (UTC)**.
The file name carries the location (or file name) and the ingestion time `HHMMSS`.

```
data/raw/
├── _manifest.csv                                   one line per raw file (audit trail)
├── source=open_meteo/
│   ├── type=forecast/                              live data, last 7 days per call
│   │   └── date=2026-11-16/
│   │       ├── bengaluru_001705.json
│   │       └── bengaluru_011705.json
│   └── type=archive/                               history (backfill / corpus)
│       └── date=2026-11-16/
│           ├── bengaluru_120102.json
│           └── london_120140.json
└── source=city_catalogue/
    └── type=csv/
        └── date=2026-11-16/
            └── cities_183900.csv
```

`_manifest.csv` columns: `ingested_at_utc, source, type, name, path, bytes, md5, details`.

Code: `pipeline/raw_data_storage/datalake.py` (`save_json`, `save_file_copy`). The root folder can be changed with the
`DATA_LAKE_DIR` environment variable.

## 3. Raw data in the database (bronze layer, Neon PostgreSQL)

| Table | Content | Link to the lake |
|---|---|---|
| `raw.weather_api` | One row per API call: `fetched_at`, `location`, `source`, `endpoint`, `payload` (JSONB) | `lake_path` |
| `raw.city_catalogue` | One row per CSV line: `ingested_at`, `source_file`, `file_md5`, `row_num`, `record` (JSONB) | `lake_path` |

Rows that fail validation go to `quarantine.weather_rejects` and `quarantine.location_rejects`
with the reason. Valid data moves on to the silver layer (`clean.weather_hourly`, `clean.locations`).

## 4. Where the raw files live

| Where it runs | Raw files | How they are kept |
|---|---|---|
| Laptop (manual / Prefect runs) | `data/raw/` in the project folder | Versioned with DVC (`dvc add data/raw`, `dvc push` to DagsHub) |
| GitHub Actions (hourly) | `data/raw/` on the temporary runner | Uploaded as a run artifact (`run-<n>-logs-and-raw`, kept 30 days); the same data is always in `raw.weather_api` |

## 5. Logs (monitoring and audit trail)

`pipeline/common/logger.py` writes every step to the console and to `logs/<step>.log`, for example `logs/ingestion.log`:

```
2026-11-16 13:17:02 UTC | INFO    | ingestion | API ingestion started: mode=live run_id=812
2026-11-16 13:17:05 UTC | INFO    | ingestion | saved Bengaluru forecast: 192 hours | raw_id=1043 | lake=data/raw/...
2026-11-16 13:17:05 UTC | INFO    | ingestion | API ingestion SUCCESS: Bengaluru raw_id=1043, 192 hourly records
```

Failed API attempts are logged as `WARNING` (with retry number) and failed runs as `ERROR`.
Each run is also recorded in `ml.pipeline_runs` (status, rows in/out, notes).
