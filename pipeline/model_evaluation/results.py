"""
Collects the evaluation results of the latest training from MLflow (DagsHub) - shared by the
model performance report and the web app.
"""
import os
import warnings

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore")

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()
EXPERIMENT = "weather-forecast"
TARGETS = {"temp": "Temperature (°C)", "humidity": "Humidity (%)", "wind": "Wind speed (km/h)",
           "cloud": "Cloud cover (%)", "rain": "Rain yes/no"}
DAYS = range(1, 8)


def latest_run(client, exp_id, name: str):
    runs = client.search_runs([exp_id], f"attributes.run_name = '{name}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    return runs[0] if runs else None


def collect() -> tuple[pd.DataFrame, dict]:
    client = MlflowClient()
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise RuntimeError(f"MLflow experiment '{EXPERIMENT}' not found - run training first")
    rows = []
    for target in TARGETS:
        run = latest_run(client, exp.experiment_id, f"target_{target}")
        if run is None:
            continue
        m = run.data.metrics
        for d in DAYS:
            row = {"target": target, "day": d, "run_id": run.info.run_id}
            for k in ("mae", "rmse", "r2", "brier", "auc", "precision", "recall", "f1"):
                if f"best_{k}_day{d}" in m:
                    row[f"model_{k}"] = m[f"best_{k}_day{d}"]
                if f"baseline_{k}_day{d}" in m:
                    row[f"baseline_{k}"] = m[f"baseline_{k}_day{d}"]
            rows.append(row)
    champ = latest_run(client, exp.experiment_id, "global_champion")
    meta = {}
    if champ:
        p = champ.data.params
        meta = {"run_id": champ.info.run_id, "data_md5": p.get("data_md5"),
                "feature_version": p.get("feature_version"), "locations": p.get("locations"),
                "test_from": p.get("test_from"), "rows_total": p.get("rows_total"),
                "model_size_mb": round(champ.data.metrics.get("model_size_mb", float("nan")), 1),
                "models": {t: p.get(f"model_{t}_day1") for t in TARGETS}}
    return pd.DataFrame(rows), meta
