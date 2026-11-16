# Data versioning and lineage

## 1. What is versioned, and how

| Asset | Path / place | Tool | Version identifier |
|---|---|---|---|
| Code, pipeline definition, SQL schema | repository | Git (GitHub) | commit hash |
| Raw data lake (API JSON + CSV copies) | `data/raw/` | DVC (remote: DagsHub) | `data/raw.dvc` (MD5) |
| Prepared dataset (silver export) | `data/prepared/clean_weather_hourly.parquet` | DVC | `data/prepared.dvc` |
| Feature dataset (gold export) | `data/features/weather_features_v3.parquet` | DVC | `data/features.dvc` |
| Training snapshot (features + labels) | `data/training/train_7day_v3.parquet` | DVC | `.dvc` file + `data_md5` in MLflow |
| Feature definitions | `pipeline/feature_store/registry.yaml` | Git + table `features.feature_registry` | feature version v1 / v2 / v3 |
| Features in the database | `features.weather_features` | `feature_version` column | v1 / v2 / v3 |
| Models | MLflow Model Registry (DagsHub) | MLflow | `weather-7day` version + alias `champion` |
| City catalogue (source file) | `data/sources/cities.csv` | Git | file MD5 stored in `raw.city_catalogue.file_md5` |

Git stores only the small `.dvc` pointer files (MD5 hash + size); DVC stores the actual data on DagsHub.

## 2. Versioning workflow

```powershell
# 1. run the pipeline (creates/updates data/raw, Neon tables)
python -m pipeline.orchestration.flow --report
# 2. export the transformed layers
python -m pipeline.versioning_lineage.export_datasets
# 3. version the data with DVC and record the version in Git
dvc add data/raw data/prepared data/features
git add data/raw.dvc data/prepared.dvc data/features.dvc data/.gitignore
git commit -m "Data version <date>"
dvc push
git push
```

**Go back to an earlier version** (time travel):

```powershell
git log --oneline -- data/prepared.dvc     # find the commit
git checkout <commit> -- data/prepared.dvc
dvc checkout data/prepared.dvc             # restores that exact file
```

## 3. Lineage metadata

| Question | Where the answer is stored |
|---|---|
| Where did a raw record come from, and when? | `raw.weather_api` (`source`, `endpoint`, `location`, `fetched_at`, `lake_path`); `data/raw/_manifest.csv` |
| Which raw record produced a clean row? | `clean.weather_hourly.raw_id` -> `raw.weather_api.id` |
| Which CSV file version loaded a location? | `raw.city_catalogue.file_md5`, `clean.locations.source`, `updated_at` |
| How was a feature computed? | `pipeline/feature_store/registry.yaml` / `features.feature_registry` (source + transformation) |
| Which data trained a model? | MLflow run params `data_md5`, `feature_version`, `data_snapshot` + model version tag `data_md5` |
| Which model made a forecast? | `ml.predictions.model_name`, `model_version` |
| What ran, when, with how many rows? | `ml.pipeline_runs` (pipeline, start, end, status, rows in/out, notes); `logs/*.log` |
| Exported dataset details | `data/prepared/_metadata.json`, `data/features/_metadata.json` (source table, transformation, rows, MD5) |

**Full chain:** forecast -> model version -> MLflow run -> training data MD5 -> Git commit of the `.dvc` file
-> features (version) -> clean rows (`raw_id`) -> raw API record (`lake_path`) -> Open-Meteo.
