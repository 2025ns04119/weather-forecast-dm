# Weather Forecast Prediction – End-to-End Data Management Pipeline

Predicts the weather for the **next 7 days for any city** from **live data** (Open-Meteo REST API + a curated city
catalogue CSV), with a pipeline that covers every stage of data management for ML: collection, ingestion,
raw storage, profiling and validation, preparation, feature engineering, feature store, versioning and lineage,
training, evaluation and orchestration.

| | Link |
|---|---|
| Live app | https://weather-forecast-dm.streamlit.app |
| Code | https://github.com/2025ns04119/weather-forecast-dm |
| Experiments, models, data versions | https://dagshub.com/2025ns04119/weather-forecast-dm |
| Pipeline runs (orchestration) | Prefect Cloud workspace of the project owner |

**Flow:** Open-Meteo API + `cities.csv` → data lake + `raw` (bronze) → validation → `clean` (silver) → features (gold)
→ feature store → 7-day models (MLflow) → `ml.predictions` → Streamlit app. Orchestrated by **Prefect**
(`pipeline/orchestration/flow.py`), run every hour by **GitHub Actions**, monitored in **Prefect Cloud**.

---

## Project structure

```
weather-forecast-dm/
├── pipeline/                         all pipeline code, one folder per stage
│   ├── data_collection/              1  data sources
│   ├── data_ingestion/               2  pull data into raw storage
│   ├── raw_data_storage/             3  partitioned data lake
│   ├── profiling_validation/         4  data quality checks and report
│   ├── data_preparation/             5  exploratory data analysis
│   ├── feature_engineering/          6  model features (gold layer)
│   ├── feature_store/                7  feature registry + versioned retrieval
│   ├── versioning_lineage/           8  dataset export for DVC + lineage
│   ├── model_training/               9  multi-horizon training with MLflow
│   ├── model_evaluation/            10  metrics, baselines, performance report
│   ├── model_deployment/                serves the model: hourly and on-demand forecasts
│   ├── orchestration/               11  Prefect flow (the DAG)
│   └── common/                          shared helpers
├── dashboard/                        Streamlit web app
├── infra/                            database schema (Neon PostgreSQL)
├── data/                             sources, raw data lake, DVC pointers
├── reports/                          generated reports and figures (created by the pipeline)
├── .github/workflows/                cloud automation (GitHub Actions)
├── requirements-full.txt             all packages (install this on a laptop)
└── requirements-pipeline.txt         pipeline packages (GitHub Actions)
```

### What each file does

| Stage | File | Purpose |
|---|---|---|
| **1 Data collection** | `pipeline/data_collection/open_meteo.py` | Open-Meteo REST API client: forecast, archive and geocoding endpoints, with retries |
| | `pipeline/data_collection/locations.py` | Default monitored city (Bengaluru) and fallback list of training cities |
| | `data/sources/cities.csv` | City catalogue (32 cities) – the second, structured data source |
| **2 Data ingestion** | `pipeline/data_ingestion/fetch_weather.py` | Ingests API data (live hourly / history backfill / all cities) into the data lake and `raw.weather_api` |
| | `pipeline/data_ingestion/ingest_csv.py` | Validates and loads the city catalogue CSV into `raw.city_catalogue` and `clean.locations` |
| **3 Raw data storage** | `pipeline/raw_data_storage/datalake.py` | Saves every raw file unchanged under `data/raw/source=…/type=…/date=…/` and records it in `_manifest.csv` |
| | `pipeline/raw_data_storage/README.md` | Storage layout documentation (lake folders, bronze tables, logs) |
| **4 Profiling & validation** | `pipeline/profiling_validation/clean_weather.py` | Range/required-field rules, Great Expectations suite, quarantine; writes the silver layer |
| | `pipeline/profiling_validation/quality_report.py` | Data quality report (PDF + JSON): missing values, duplicates, ranges, completeness, quarantine |
| | `pipeline/profiling_validation/quality_metrics.py` | Data quality metrics in SQL, shared by the report and the app's *Data quality* tab |
| **5 Data preparation** | `pipeline/data_preparation/eda.py` | EDA: 7 plots + summary of distributions, correlations, seasonality, sparsity, completeness |
| | `pipeline/data_preparation/eda_metrics.py` | The same EDA computed with SQL aggregates for the app's *Exploratory analysis* tab |
| **6 Feature engineering** | `pipeline/feature_engineering/build_features.py` | Builds 18 features (lags, rolling means, solar hour, rain totals …), version v3, gold layer |
| **7 Feature store** | `pipeline/feature_store/registry.yaml` | Feature metadata: name, source, transformation, versions |
| | `pipeline/feature_store/store.py` | Registry sync/check and versioned retrieval for training and inference |
| | `pipeline/feature_store/demo_retrieval.py` | Demonstrates training and online retrieval for different versions |
| | `pipeline/feature_store/README.md` | Feature store documentation |
| **8 Versioning & lineage** | `pipeline/versioning_lineage/export_datasets.py` | Exports the silver and gold layers to `data/prepared/` and `data/features/` for DVC |
| | `pipeline/versioning_lineage/README.md` | What is versioned with Git / DVC / MLflow, the versioning workflow and the lineage chain |
| **9 Model training** | `pipeline/model_training/train.py` | Trains 5 targets × 7 days, logs to MLflow, registers the champion `weather-7day` |
| | `pipeline/model_training/multi_horizon.py` | The combined 7-day multi-target model class and target definitions |
| **10 Model evaluation** | `pipeline/model_evaluation/metrics.py` | MAE, RMSE, R², Brier, AUC and persistence/climatology baselines |
| | `pipeline/model_evaluation/performance_report.py` | Model vs baseline report (PDF + CSV) from MLflow |
| | `pipeline/model_evaluation/results.py` | Reads the latest test metrics from MLflow, for the report and the app's *Model evaluation* tab |
| **Model deployment** | `pipeline/model_deployment/predict.py` | Hourly 7-day forecast for Bengaluru, saved to `ml.predictions` |
| | `pipeline/model_deployment/on_demand.py` | Forecast for any searched city, in memory (used by the app) |
| **11 Orchestration** | `pipeline/orchestration/flow.py` | Prefect flow: CSV → API → validate → [reports] → features → feature store → [train] → predict |
| | `.github/workflows/hourly_pipeline.yml` | Runs the flow every hour in the cloud; buttons to retrain and build reports |
| | `.github/workflows/data_versioning.yml` | Cloud job: reports + EDA, dataset export, DVC versioning, commits the results |
| **Shared** | `pipeline/common/db.py` | Neon connection and the pipeline run log (`ml.pipeline_runs`) |
| | `pipeline/common/logger.py` | Writes every step to the console and `logs/<step>.log` |
| | `pipeline/common/model_loader.py` | Loads the champion model from the MLflow registry |
| | `pipeline/common/derived.py` | Feels-like temperature, UV estimate, sunrise/sunset, weather condition (used by the app) |
| **App** | `dashboard/app.py`, `dashboard/weather_view.py` | Streamlit app: weather for any city + Insights page (accuracy, pipeline health) |
| **Database** | `infra/neon_init.sql`, `infra/migrate_v2…v5.sql` | Schema: `raw`, `clean`, `quarantine`, `features`, `ml` |
| | `infra/run_sql.py`, `infra/setup_neon.py` | Run a SQL file / one-time database setup |
| **Packages** | `requirements-full.txt` | **Everything** – what a team member installs on a laptop |
| | `requirements-pipeline.txt` | Pipeline packages only – installed by GitHub Actions |
| | `dashboard/requirements.txt` | Web app packages only – installed by Streamlit Cloud (this exact name is required) |
| **Config** | `.env.example` | Template for your private `.env` settings |
| | `.gitignore`, `.dvcignore`, `.dvc/`, `data/**/*.dvc` | Git/DVC rules and DVC data pointers |

---

## 1. Register (free accounts)

### 1.1 GitHub
1. Go to https://github.com/signup.
2. Enter your email, a password and a username, then verify your email.
3. Send your **GitHub username** to the project owner (2025ns04119).
4. Open the invite email from GitHub and click **Accept invitation**.

### 1.2 DagsHub (stores the models and the data versions)
1. Go to https://dagshub.com and click **Sign up**, then **Sign up with GitHub**.
2. Send your **DagsHub username** to the project owner.
3. Accept the invite (it arrives by email or under the bell icon on DagsHub).
4. Copy your token: click your photo → **Your settings** → **Tokens** → copy the default token. Keep it private.

### 1.3 Neon (the cloud database)
1. Send the **email address** you will use for Neon to the project owner.
2. Open the Neon invite email and click **Accept**. If you don't have an account yet, sign up at https://neon.tech with **that same email**.
3. Open the project **weather-forecast-dm** → click **Connect** → copy the connection string (it starts with `postgresql://`). Keep it private.

> **The owner adds members here:** GitHub repo → Settings → Collaborators · DagsHub repo → Settings → Collaborators (Write) · Neon → Organization settings → Members → Invite.

---

## 2. Install the software (one time, Windows)

Open **PowerShell** and run:

```powershell
winget install Python.Python.3.11
winget install Git.Git
winget install Microsoft.VisualStudioCode
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Close PowerShell. Open **VS Code** → **Extensions** (Ctrl+Shift+X) → search for **Python** → install the extension by Microsoft.

> Use Python **3.11**. Python 3.13 and 3.14 break some of the packages this project needs.

---

## 3. Clone the GitHub repo in VS Code

1. Open VS Code.
2. Press **Ctrl+Shift+P**, type **Git: Clone** and press Enter.
3. Paste `https://github.com/2025ns04119/weather-forecast-dm.git` and press Enter.
4. Choose a folder (for example, **Desktop**) and click **Select as Repository Destination**.
5. If you are asked to sign in to GitHub, click **Allow** and sign in through your browser.
6. Click **Open**, then click **Yes, I trust the authors**.

You will see the project folders in the left **Explorer** panel.

---

## 4. Set up the project in VS Code

### 4.1 Open a terminal
Menu **Terminal → New Terminal**. A PowerShell terminal opens at the bottom of the window, already inside the project folder.

### 4.2 Create the virtual environment and install the packages
```powershell
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-full.txt
```
- The terminal prompt now starts with `(venv)`.
- Installing the packages takes about 5–10 minutes.
- Press **Ctrl+Shift+P** → **Python: Select Interpreter** → choose the one that shows `venv`.

### 4.3 Create the `.env` file (your private settings)
1. In **Explorer**, right-click an empty area → **New File** → name it `.env`.
2. Paste the lines below, fill in your own values, and save with **Ctrl+S**:

```
NEON_DB_URL=<your Neon connection string>
MLFLOW_TRACKING_URI=https://dagshub.com/2025ns04119/weather-forecast-dm.mlflow
MLFLOW_TRACKING_USERNAME=<your DagsHub username>
MLFLOW_TRACKING_PASSWORD=<your DagsHub token>
LAT=12.97
LON=77.59
LOCATION=Bengaluru
```
Don't put spaces around `=`, and don't use quotes. Git never uploads `.env`, so your secrets stay on your computer.

### 4.4 Download the versioned training data (DVC)
```powershell
python -m dvc remote modify origin --local auth basic
python -m dvc remote modify origin --local user <your DagsHub username>
python -m dvc remote modify origin --local password <your DagsHub token>
python -m dvc pull
```

---

## 5. Run the pipeline

Make sure the prompt shows `(venv)`.

**All stages at once (Prefect flow):**
```powershell
python -m pipeline.orchestration.flow             # hourly run: data + forecast
python -m pipeline.orchestration.flow --report    # + data quality report and EDA
python -m pipeline.orchestration.flow --train     # + retrain the model (only if the team agrees)
```

**Stage by stage:**

| Stage | Command | Output |
|---|---|---|
| 2 Ingestion (CSV) | `python -m pipeline.data_ingestion.ingest_csv` | `clean.locations` (skipped if the CSV is unchanged) |
| 2 Ingestion (API) | `python -m pipeline.data_ingestion.fetch_weather --mode live` | `data/raw/…json` + `raw.weather_api` |
| 4 Validation | `python -m pipeline.profiling_validation.clean_weather` | `clean.weather_hourly`, `quarantine.weather_rejects` |
| 4 Quality report | `python -m pipeline.profiling_validation.quality_report` | `reports/data_quality_report.pdf` |
| 5 EDA | `python -m pipeline.data_preparation.eda` | `reports/figures/*.png`, `reports/eda_summary.md` |
| 6 Features | `python -m pipeline.feature_engineering.build_features` | `features.weather_features` |
| 7 Feature store | `python -m pipeline.feature_store.store --sync --check` | `features.feature_registry` |
| 7 Feature demo | `python -m pipeline.feature_store.demo_retrieval` | Retrieval demo in the terminal |
| 8 Export for DVC | `python -m pipeline.versioning_lineage.export_datasets` | `data/prepared/`, `data/features/` |
| 9 Training | `python -m pipeline.model_training.train` | New model version in MLflow (alias `champion`) |
| 10 Evaluation | `python -m pipeline.model_evaluation.performance_report` | `reports/model_performance.pdf` |
| Model deployment | `python -m pipeline.model_deployment.predict` | 7-day forecast in `ml.predictions` |

> **Do not run** `infra/setup_neon.py`, the `migrate_*.sql` files or training unless the team agrees. The database is
> already set up, and training replaces the live model.

### In the cloud (no laptop needed)
- **Hourly pipeline:** GitHub → Actions → *Hourly weather pipeline* runs every hour. **Run workflow** → tick *train* /
  *report* to retrain or build reports. Results are in each run's **Artifacts** and in **Prefect Cloud → Runs**.
- **Data versioning and EDA:** GitHub → Actions → *Data versioning and EDA* → **Run workflow**. Commits the DVC
  data versions and reports back to the repository.

---

## 6. See it in the Streamlit app

### 6.1 On your laptop
```powershell
python -m streamlit run dashboard/app.py
```
- Your browser opens http://localhost:8501.
- **Weather page:** Bengaluru is shown by default. Type any city in the search bar to see its forecast.
- Switch between the **Hourly** and **7 days** views.
- **Insights page:** forecast accuracy, pipeline runs and data-quality counts.
- Press **Ctrl+C** in the terminal to stop the app.

### 6.2 Online (no setup needed)
Open https://weather-forecast-dm.streamlit.app. If it says *"This app has gone to sleep"*, click
**Yes, get this app back up** and wait about 1 minute.

---

## 7. Next time you open the project
```powershell
venv\Scripts\activate
git pull
python -m dvc pull
```

## 8. Common problems

| Problem | Fix |
|---|---|
| `activate` is blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `No module named pipeline` | Run commands from the project root folder (where `README.md` is) |
| `No module named …` (a package) | Make sure `(venv)` shows, then `python -m pip install -r requirements-full.txt` |
| `Fatal error in launcher` for `pip`, `dvc`, `prefect`, `streamlit` | Use `python -m pip …`, `python -m dvc …`, `python -m prefect …`, `python -m streamlit …` |
| `password authentication failed` | Copy the Neon connection string into `.env` again |
| `401` / `Unauthorized` (DagsHub or DVC) | Use your own DagsHub username and token; accept the DagsHub invite |
| `No registered versions of 'weather-7day'` | Check `MLFLOW_TRACKING_URI` in `.env` |
| Forecast says the features are too old | Run the API ingestion, validation and features steps, then predict |

**Team rule:** never share `.env`, your token or your connection string in chat or in screenshots.
