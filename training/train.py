"""
7-day temperature forecast training with experiment tracking (GOLD -> MODEL).

Strategy: DIRECT MULTI-HORIZON
    One model per forecast day d = 1..7 predicts the temperature at t + 24*d hours.
    Applied to the latest 24 hourly rows, the 7 models give an hourly forecast
    for the whole next week (168 hours).

Steps
1. Load features (v1) + build 7 labels from clean.weather_hourly
2. Save a training-data snapshot (versioned with DVC) + MD5 for lineage
3. Time-based split with an embargo gap (no label may fall in the test period)
4. Baseline + Linear Regression + Random Forest + XGBoost for every day
5. Pick the best model type per day, package all 7 as one model, log to MLflow
6. Register it in the MLflow Model Registry as weather-temp-7day (alias champion)

Usage (from the project root):
    python -m training.train
"""
import hashlib
import logging
import os
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore", category=UserWarning)

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import text
from xgboost import XGBRegressor

from common.db import finish_run, get_engine, start_run
from features.build_features import FEATURE_COLS, FEATURE_VERSION
from training.multi_horizon import HORIZON_DAYS, MultiHorizonModel

logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)

EXPERIMENT = "weather-forecast"
MODEL_NAME = "weather-temp-7day"
TEST_FRACTION = 0.2
SNAPSHOT = Path("data/training/train_7day_v1.parquet")
# Our own trained models -> cloudpickle is safe (MLflow's default "skops"
# format rejects tree models such as RandomForest / XGBoost)
SAVE_FORMAT = mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE


def label(d: int) -> str:
    return f"y_day{d}"


# ---------------------------------------------------------------- data
def load_data(engine, location: str) -> pd.DataFrame:
    with engine.connect() as conn:
        feats = pd.read_sql(text(f"""
            SELECT ts, {", ".join(FEATURE_COLS)} FROM features.weather_features
            WHERE location = :loc AND feature_version = :v ORDER BY ts"""),
            conn, params={"loc": location, "v": FEATURE_VERSION})
        temps = pd.read_sql(text("""
            SELECT ts, temperature_2m FROM clean.weather_hourly
            WHERE location = :loc ORDER BY ts"""), conn, params={"loc": location})
    feats["ts"] = pd.to_datetime(feats["ts"], utc=True)
    temps["ts"] = pd.to_datetime(temps["ts"], utc=True)
    return add_labels(feats, temps)


def add_labels(feats: pd.DataFrame, temps: pd.DataFrame) -> pd.DataFrame:
    """Label for day d = observed temperature exactly 24*d hours later."""
    series = temps.set_index("ts")["temperature_2m"]
    out = feats.copy()
    for d in HORIZON_DAYS:
        out[label(d)] = series.reindex(out["ts"] + pd.Timedelta(hours=24 * d)).values
    return out


def save_snapshot(df: pd.DataFrame) -> str:
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SNAPSHOT, index=False)
    return hashlib.md5(SNAPSHOT.read_bytes()).hexdigest()


def split_for_day(df: pd.DataFrame, cutoff: pd.Timestamp, d: int):
    """Train: label time < cutoff (embargo). Test: feature time >= cutoff."""
    y = label(d)
    lab = df[df[y].notna()]
    train = lab[lab["ts"] + pd.Timedelta(hours=24 * d) < cutoff]
    test = lab[lab["ts"] >= cutoff]
    return (train[FEATURE_COLS].astype(float), train[y],
            test[FEATURE_COLS].astype(float), test[y])


def metrics(y_true, y_pred) -> dict:
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred))}


def make_model(name: str):
    return {
        "linear_regression": lambda: LinearRegression(),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42),
        "xgboost": lambda: XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1),
    }[name]()


CANDIDATES = ["linear_regression", "random_forest", "xgboost"]


# ---------------------------------------------------------------- MLflow
def log_model(model, X_example):
    """Save in MLflow format and upload as plain run artifacts (works on DagsHub)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model"
        mlflow.sklearn.save_model(model, path, input_example=X_example,
                                  serialization_format=SAVE_FORMAT)
        mlflow.log_artifacts(str(path), artifact_path="model")


def register_champion(run_id: str, data_md5: str) -> str:
    """Register via create_model_version (DagsHub lacks MLflow 3's logged-model lookup)."""
    client = MlflowClient()
    try:
        try:
            client.create_registered_model(MODEL_NAME)
        except Exception:
            pass
        source = f"{client.get_run(run_id).info.artifact_uri}/model"
        mv = client.create_model_version(name=MODEL_NAME, source=source, run_id=run_id,
                                         tags={"data_md5": data_md5})
        label_txt = "alias champion"
        try:
            client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
        except Exception:
            try:
                client.transition_model_version_stage(
                    MODEL_NAME, mv.version, "Production", archive_existing_versions=True)
                label_txt = "stage Production"
            except Exception:
                label_txt = "no alias/stage"
        registered = f"{MODEL_NAME} v{mv.version}"
        print(f"Registered: {registered} ({label_txt})")
        return registered
    except Exception as e:
        print(f"Model registry step skipped: {e}")
        return "not registered"


# ---------------------------------------------------------------- main
def main():
    location = os.getenv("LOCATION", "Bengaluru")
    engine = get_engine()
    run_log = start_run(engine, "train_7day")
    try:
        df = load_data(engine, location)
        labelled = df[df[label(1)].notna()]
        if len(labelled) < 1000:
            raise RuntimeError(f"Only {len(labelled)} labelled rows - run build_features --all first")
        data_md5 = save_snapshot(df)
        cutoff = labelled["ts"].iloc[int(len(labelled) * (1 - TEST_FRACTION))]

        data_info = {"location": location, "feature_version": FEATURE_VERSION,
                     "data_md5": data_md5, "data_snapshot": str(SNAPSHOT),
                     "strategy": "direct_multi_horizon", "horizon_days": len(HORIZON_DAYS),
                     "test_from": str(cutoff), "rows_total": len(df)}
        print(f"Rows: {len(df)} | test period from {cutoff:%Y-%m-%d} | data md5: {data_md5[:10]}\n")

        mlflow.set_experiment(EXPERIMENT)
        scores = {}   # (model_name, day) -> metrics
        fitted = {}   # (model_name, day) -> fitted model

        # Baseline: "same hour on day d = temperature now"
        with mlflow.start_run(run_name="7day_baseline_persistence"):
            mlflow.log_params(data_info)
            for d in HORIZON_DAYS:
                _, _, X_te, y_te = split_for_day(df, cutoff, d)
                m = metrics(y_te, X_te["temp_lag_1h"])
                scores[("baseline", d)] = m
                mlflow.log_metrics({f"{k}_day{d}": v for k, v in m.items()})

        for name in CANDIDATES:
            with mlflow.start_run(run_name=f"7day_{name}"):
                mlflow.log_params(data_info)
                mlflow.set_tag("model_type", name)
                for d in HORIZON_DAYS:
                    X_tr, y_tr, X_te, y_te = split_for_day(df, cutoff, d)
                    model = make_model(name).fit(X_tr, y_tr)
                    m = metrics(y_te, model.predict(X_te))
                    scores[(name, d)] = m
                    fitted[(name, d)] = model
                    mlflow.log_metrics({f"{k}_day{d}": v for k, v in m.items()})
                mlflow.log_metric("rmse_avg", float(np.mean([scores[(name, d)]["rmse"]
                                                             for d in HORIZON_DAYS])))
            print(f"  trained {name} for days 1-7")

        # Best model type per day -> one combined 7-day model
        best = {d: min(CANDIDATES, key=lambda n: scores[(n, d)]["rmse"]) for d in HORIZON_DAYS}
        champion = MultiHorizonModel(models={d: fitted[(best[d], d)] for d in HORIZON_DAYS},
                                     feature_cols=FEATURE_COLS, model_types=best)

        print("\nDay | best model         | MAE (C) | RMSE  | baseline RMSE")
        with mlflow.start_run(run_name="7day_champion") as run:
            mlflow.log_params(data_info)
            mlflow.log_params({f"model_day{d}": best[d] for d in HORIZON_DAYS})
            for d in HORIZON_DAYS:
                m, b = scores[(best[d], d)], scores[("baseline", d)]
                mlflow.log_metrics({f"{k}_day{d}": v for k, v in m.items()})
                print(f" {d}  | {best[d]:<18} |  {m['mae']:.2f}   | {m['rmse']:.2f}  | {b['rmse']:.2f}")
            avg_rmse = float(np.mean([scores[(best[d], d)]["rmse"] for d in HORIZON_DAYS]))
            mlflow.log_metric("rmse_avg", avg_rmse)
            log_model(champion, df[FEATURE_COLS].astype(float).head(5))
            champion_run_id = run.info.run_id

        print()
        registered = register_champion(champion_run_id, data_md5)
        finish_run(engine, run_log, "success", len(df), len(HORIZON_DAYS),
                   f"model={MODEL_NAME}; champion_run_id={champion_run_id}; "
                   f"rmse_avg={avg_rmse:.3f}; registered={registered}; data_md5={data_md5}")
    except Exception as e:
        finish_run(engine, run_log, "failed", notes=str(e)[:500])
        raise


if __name__ == "__main__":
    main()
