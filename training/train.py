"""
GLOBAL 7-day temperature forecast training (GOLD -> MODEL), feature version v2.

One model for every place on Earth: trained on a corpus of cities with very
different climates, using location-agnostic features (solar hour, latitude, lags).

Strategy: DIRECT MULTI-HORIZON - one model per forecast day d = 1..7 predicts the
temperature at t + 24*d hours. Applied to the latest 24 hourly rows the 7 models
give an hourly forecast for the whole next week.

Steps
1. Load v2 features for ALL locations + build 7 labels per location
2. Save a training-data snapshot (versioned with DVC) + MD5 for lineage
3. Time-based split with an embargo gap (no label may fall in the test period)
4. Baseline + Linear Regression + Random Forest + XGBoost for every day
5. Best model type per day -> one combined model -> MLflow
6. Register as weather-temp-7day (alias champion)

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
SNAPSHOT = Path(f"data/training/train_7day_{FEATURE_VERSION}.parquet")
MONITORED = os.getenv("LOCATION", "Bengaluru")
# Our own trained models -> cloudpickle is safe (MLflow's default "skops"
# format rejects tree models such as RandomForest / XGBoost)
SAVE_FORMAT = mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
CANDIDATES = ["linear_regression", "random_forest", "xgboost"]


def label(d: int) -> str:
    return f"y_day{d}"


# ---------------------------------------------------------------- data
def load_data(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        feats = pd.read_sql(text(f"""
            SELECT location, ts, {", ".join(FEATURE_COLS)} FROM features.weather_features
            WHERE feature_version = :v ORDER BY location, ts"""),
            conn, params={"v": FEATURE_VERSION})
        temps = pd.read_sql(text("""
            SELECT location, ts, temperature_2m FROM clean.weather_hourly
            ORDER BY location, ts"""), conn)
    feats["ts"] = pd.to_datetime(feats["ts"], utc=True)
    temps["ts"] = pd.to_datetime(temps["ts"], utc=True)
    return add_labels(feats, temps)


def add_labels(feats: pd.DataFrame, temps: pd.DataFrame) -> pd.DataFrame:
    """Label for day d = observed temperature at the SAME location exactly 24*d hours later."""
    series = temps.set_index(["location", "ts"])["temperature_2m"]
    out = feats.copy()
    for d in HORIZON_DAYS:
        key = pd.MultiIndex.from_arrays([out["location"], out["ts"] + pd.Timedelta(hours=24 * d)])
        out[label(d)] = series.reindex(key).values
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
    return train, test


def metrics(y_true, y_pred) -> dict:
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred))}


def make_model(name: str):
    return {
        "linear_regression": lambda: LinearRegression(),
        # max_samples keeps training fast on the large multi-city corpus
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=120, max_depth=16, min_samples_leaf=10, max_samples=0.3,
            n_jobs=-1, random_state=42),
        "xgboost": lambda: XGBRegressor(
            n_estimators=500, max_depth=7, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=-1),
    }[name]()


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
        mv = client.create_model_version(
            name=MODEL_NAME, source=source, run_id=run_id,
            tags={"data_md5": data_md5, "feature_version": FEATURE_VERSION, "scope": "global"})
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
    engine = get_engine()
    run_log = start_run(engine, "train_7day")
    try:
        df = load_data(engine)
        labelled = df[df[label(1)].notna()].sort_values("ts")
        if len(labelled) < 1000:
            raise RuntimeError(f"Only {len(labelled)} labelled rows - run build_features --all-locations first")
        n_loc = df["location"].nunique()
        data_md5 = save_snapshot(df)
        cutoff = labelled["ts"].iloc[int(len(labelled) * (1 - TEST_FRACTION))]

        data_info = {"scope": "global", "locations": n_loc, "feature_version": FEATURE_VERSION,
                     "features": ",".join(FEATURE_COLS), "data_md5": data_md5,
                     "data_snapshot": str(SNAPSHOT), "strategy": "direct_multi_horizon",
                     "horizon_days": len(HORIZON_DAYS), "test_from": str(cutoff),
                     "rows_total": len(df)}
        print(f"Locations: {n_loc} | rows: {len(df):,} | test period from {cutoff:%Y-%m-%d} | "
              f"data md5: {data_md5[:10]}\n")

        mlflow.set_experiment(EXPERIMENT)
        scores, local_scores, fitted = {}, {}, {}

        with mlflow.start_run(run_name="global_baseline_persistence"):
            mlflow.log_params(data_info)
            for d in HORIZON_DAYS:
                _, te = split_for_day(df, cutoff, d)
                m = metrics(te[label(d)], te["temp_lag_1h"])
                scores[("baseline", d)] = m
                mlflow.log_metrics({f"{k}_day{d}": v for k, v in m.items()})

        for name in CANDIDATES:
            with mlflow.start_run(run_name=f"global_{name}"):
                mlflow.log_params(data_info)
                mlflow.set_tag("model_type", name)
                for d in HORIZON_DAYS:
                    tr, te = split_for_day(df, cutoff, d)
                    model = make_model(name).fit(tr[FEATURE_COLS].astype(float), tr[label(d)])
                    pred = model.predict(te[FEATURE_COLS].astype(float))
                    m = metrics(te[label(d)], pred)
                    scores[(name, d)] = m
                    fitted[(name, d)] = model
                    mon = (te["location"] == MONITORED).to_numpy()
                    if mon.any():
                        local_scores[(name, d)] = metrics(te[label(d)][mon], pred[mon])
                    mlflow.log_metrics({f"{k}_day{d}": v for k, v in m.items()})
                mlflow.log_metric("rmse_avg", float(np.mean([scores[(name, d)]["rmse"]
                                                             for d in HORIZON_DAYS])))
            print(f"  trained {name} for days 1-7")

        best = {d: min(CANDIDATES, key=lambda n: scores[(n, d)]["rmse"]) for d in HORIZON_DAYS}
        champion = MultiHorizonModel(models={d: fitted[(best[d], d)] for d in HORIZON_DAYS},
                                     feature_cols=FEATURE_COLS, model_types=best)

        print(f"\nDay | best model         | all cities MAE | RMSE | baseline RMSE | {MONITORED} MAE")
        with mlflow.start_run(run_name="global_champion") as run:
            mlflow.log_params(data_info)
            mlflow.log_params({f"model_day{d}": best[d] for d in HORIZON_DAYS})
            for d in HORIZON_DAYS:
                m, b = scores[(best[d], d)], scores[("baseline", d)]
                mlflow.log_metrics({f"{k}_day{d}": v for k, v in m.items()})
                loc = local_scores.get((best[d], d))
                if loc:
                    mlflow.log_metric(f"mae_day{d}_{MONITORED.lower()}", loc["mae"])
                loc_txt = f"{loc['mae']:.2f}" if loc else "-"
                print(f" {d}  | {best[d]:<18} |     {m['mae']:.2f}       | {m['rmse']:.2f} |"
                      f"     {b['rmse']:.2f}      |   {loc_txt}")
            avg_rmse = float(np.mean([scores[(best[d], d)]["rmse"] for d in HORIZON_DAYS]))
            mlflow.log_metric("rmse_avg", avg_rmse)
            log_model(champion, df[FEATURE_COLS].astype(float).head(5))
            champion_run_id = run.info.run_id

        print()
        registered = register_champion(champion_run_id, data_md5)
        finish_run(engine, run_log, "success", len(df), len(HORIZON_DAYS),
                   f"model={MODEL_NAME}; scope=global; locations={n_loc}; "
                   f"champion_run_id={champion_run_id}; rmse_avg={avg_rmse:.3f}; "
                   f"registered={registered}; data_md5={data_md5}")
    except Exception as e:
        finish_run(engine, run_log, "failed", notes=str(e)[:500])
        raise


if __name__ == "__main__":
    main()
