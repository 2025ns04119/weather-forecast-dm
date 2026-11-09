"""
GLOBAL multi-target 7-day weather forecast training (GOLD -> MODEL), feature version v3.

Targets (each for day 1..7 ahead):
    temp, humidity, wind, cloud   regression      -> metrics MAE / RMSE / R2
    rain                          classification  -> probability of >= 0.1 mm/h; Brier score, AUC

Strategy: DIRECT MULTI-HORIZON - one model per (target, day). Applied to the latest
24 hourly rows they give an hourly forecast of every variable for the next week.

Steps
1. Load v3 features for ALL locations + build labels from clean.weather_hourly
2. Save a training-data snapshot (versioned with DVC) + MD5 for lineage
3. Time-based split with an embargo gap (no label may fall in the test period)
4. Per target: baseline + candidate models for every day, logged to MLflow
5. Best model per (target, day) -> one combined model -> MLflow
6. Register as weather-7day (alias champion)

Usage (from the project root):
    python -m training.train
"""
import hashlib
import logging
import os
import pickle
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (brier_score_loss, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from xgboost import XGBClassifier, XGBRegressor

from common.db import finish_run, get_engine, start_run
from features.build_features import FEATURE_COLS, FEATURE_VERSION
from training.multi_horizon import (HORIZON_DAYS, RAIN_THRESHOLD_MM, TARGETS,
                                    MultiHorizonModel)

logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)

EXPERIMENT = "weather-forecast"
MODEL_NAME = "weather-7day"
TEST_FRACTION = 0.2
SNAPSHOT = Path(f"data/training/train_7day_{FEATURE_VERSION}.parquet")
MONITORED = os.getenv("LOCATION", "Bengaluru")
# Our own trained models -> cloudpickle is safe (MLflow's default "skops" format rejects trees)
SAVE_FORMAT = mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE


def label(target: str, d: int) -> str:
    return f"y_{target}_d{d}"


# ---------------------------------------------------------------- data
def load_data(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        feats = pd.read_sql(text(f"""
            SELECT location, ts, {", ".join(FEATURE_COLS)} FROM features.weather_features
            WHERE feature_version = :v ORDER BY location, ts"""),
            conn, params={"v": FEATURE_VERSION})
        clean = pd.read_sql(text("""
            SELECT location, ts, temperature_2m, relative_humidity_2m, wind_speed_10m,
                   cloud_cover, precipitation
            FROM clean.weather_hourly ORDER BY location, ts"""), conn)
    feats["ts"] = pd.to_datetime(feats["ts"], utc=True)
    clean["ts"] = pd.to_datetime(clean["ts"], utc=True)
    return add_labels(feats, clean)


def add_labels(feats: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Label (target, d) = observed value at the SAME location exactly 24*d hours later."""
    obs = clean.set_index(["location", "ts"])
    out = feats.copy()
    for d in HORIZON_DAYS:
        key = pd.MultiIndex.from_arrays([out["location"], out["ts"] + pd.Timedelta(hours=24 * d)])
        future = obs.reindex(key)
        for target, (col, kind) in TARGETS.items():
            v = future[col].to_numpy(dtype=float)
            if kind == "clf":
                v = np.where(np.isnan(v), np.nan, (v >= RAIN_THRESHOLD_MM).astype(float))
            out[label(target, d)] = v
    return out


def save_snapshot(df: pd.DataFrame) -> str:
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SNAPSHOT, index=False)
    return hashlib.md5(SNAPSHOT.read_bytes()).hexdigest()


def split_for(df, cutoff, target, d):
    """Train: label time < cutoff (embargo). Test: feature time >= cutoff."""
    y = label(target, d)
    lab = df[df[y].notna()]
    return lab[lab["ts"] + pd.Timedelta(hours=24 * d) < cutoff], lab[lab["ts"] >= cutoff]


# ---------------------------------------------------------------- models & metrics
def candidates(target: str, kind: str) -> dict:
    big = target == "temp"     # temperature gets the larger XGBoost; others stay small & fast
    if kind == "clf":
        return {
            "logistic_regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
            "xgboost": lambda: XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.08,
                                             subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                             eval_metric="logloss", random_state=42, n_jobs=-1),
        }
    return {
        "linear_regression": lambda: LinearRegression(),
        "xgboost": lambda: XGBRegressor(n_estimators=300 if big else 150, max_depth=6 if big else 5,
                                        learning_rate=0.06 if big else 0.08, subsample=0.8,
                                        colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=-1),
    }
# Random Forest is excluded: on the multi-city corpus one RF day-model is ~80 MB (too slow to serve).


def score(kind, y_true, pred) -> dict:
    if kind == "clf":
        out = {"brier": float(brier_score_loss(y_true, pred))}
        try:
            out["auc"] = float(roc_auc_score(y_true, pred))
        except ValueError:          # only one class present in the test period
            out["auc"] = float("nan")
        return out
    return {"mae": float(mean_absolute_error(y_true, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
            "r2": float(r2_score(y_true, pred))}


def selection_metric(kind) -> str:
    return "brier" if kind == "clf" else "rmse"


def baseline(target, kind, train, test):
    """Persistence for regression (value now); climatology (training rain rate) for rain."""
    if kind == "clf":
        return np.full(len(test), train[label(target, 1)].mean())
    now_col = {"temp": "temp_lag_1h", "humidity": "humidity", "wind": "wind_speed", "cloud": "cloud_cover"}[target]
    return test[now_col].to_numpy()


def predict(model, kind, X):
    return model.predict_proba(X)[:, 1] if kind == "clf" else model.predict(X)


# ---------------------------------------------------------------- MLflow helpers
def log_model(model, X_example):
    """Save in MLflow format and upload as plain run artifacts (works on DagsHub)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model"
        mlflow.sklearn.save_model(model, path, input_example=X_example, serialization_format=SAVE_FORMAT)
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
            tags={"data_md5": data_md5, "feature_version": FEATURE_VERSION, "targets": ",".join(TARGETS)})
        label_txt = "alias champion"
        try:
            client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
        except Exception:
            try:
                client.transition_model_version_stage(MODEL_NAME, mv.version, "Production",
                                                      archive_existing_versions=True)
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
        labelled = df[df[label("temp", 1)].notna()].sort_values("ts")
        if len(labelled) < 1000:
            raise RuntimeError(f"Only {len(labelled)} labelled rows - run build_features --all-locations first")
        n_loc = df["location"].nunique()
        data_md5 = save_snapshot(df)
        cutoff = labelled["ts"].iloc[int(len(labelled) * (1 - TEST_FRACTION))]
        data_info = {"scope": "global", "locations": n_loc, "feature_version": FEATURE_VERSION,
                     "targets": ",".join(TARGETS), "data_md5": data_md5, "data_snapshot": str(SNAPSHOT),
                     "strategy": "direct_multi_horizon", "horizon_days": len(HORIZON_DAYS),
                     "test_from": str(cutoff), "rows_total": len(df)}
        print(f"Locations: {n_loc} | rows: {len(df):,} | test from {cutoff:%Y-%m-%d} | md5 {data_md5[:10]}\n")

        mlflow.set_experiment(EXPERIMENT)
        models, kinds, types, summary = {}, {}, {}, []

        for target, (_, kind) in TARGETS.items():
            kinds[target], models[target], types[target] = kind, {}, {}
            sel = selection_metric(kind)
            with mlflow.start_run(run_name=f"target_{target}"):
                mlflow.log_params(data_info)
                mlflow.set_tags({"target": target, "kind": kind})
                for d in HORIZON_DAYS:
                    tr, te = split_for(df, cutoff, target, d)
                    X_tr, y_tr = tr[FEATURE_COLS].astype(float), tr[label(target, d)]
                    X_te, y_te = te[FEATURE_COLS].astype(float), te[label(target, d)]
                    base = score(kind, y_te, baseline(target, kind, tr, te))
                    best = None
                    for name, make in candidates(target, kind).items():
                        m = make().fit(X_tr, y_tr.astype(int) if kind == "clf" else y_tr)
                        s = score(kind, y_te, predict(m, kind, X_te))
                        mlflow.log_metrics({f"{name}_{k}_day{d}": v for k, v in s.items() if v == v})
                        if best is None or s[sel] < best[2][sel]:
                            best = (name, m, s)
                    models[target][d], types[target][d] = best[1], best[0]
                    mlflow.log_metrics({f"best_{k}_day{d}": v for k, v in best[2].items() if v == v})
                    mlflow.log_metrics({f"baseline_{k}_day{d}": v for k, v in base.items() if v == v})
                    summary.append((target, d, best[0], best[2], base))
            print(f"  trained {target:<9} ({'classification' if kind == 'clf' else 'regression'}) for days 1-7")

        print("\nTarget    Day  Best model            Score                 Baseline")
        for target, d, name, s, b in summary:
            if kinds[target] == "clf":
                txt, btxt = f"Brier {s['brier']:.3f}  AUC {s['auc']:.2f}", f"Brier {b['brier']:.3f}"
            else:
                txt, btxt = f"MAE {s['mae']:.2f}  RMSE {s['rmse']:.2f}", f"RMSE {b['rmse']:.2f}"
            if d in (1, 4, 7):
                print(f"{target:<9} {d:>3}  {name:<20}  {txt:<21} {btxt}")

        champion = MultiHorizonModel(models=models, kinds=kinds, feature_cols=FEATURE_COLS, model_types=types)
        size_mb = len(pickle.dumps(champion)) / 1e6
        with mlflow.start_run(run_name="global_champion") as run:
            mlflow.log_params(data_info)
            mlflow.log_params({f"model_{t}_day{d}": types[t][d] for t in types for d in HORIZON_DAYS})
            for target, d, _, s, _ in summary:
                mlflow.log_metrics({f"{target}_{k}_day{d}": v for k, v in s.items() if v == v})
            mlflow.log_metric("model_size_mb", size_mb)
            log_model(champion, df[FEATURE_COLS].astype(float).head(5))
            champion_run_id = run.info.run_id
        print(f"\nModel size: {size_mb:.1f} MB")

        registered = register_champion(champion_run_id, data_md5)
        temp_rmse = float(np.mean([s["rmse"] for t, d, _, s, _ in summary if t == "temp"]))
        finish_run(engine, run_log, "success", len(df), len(summary),
                   f"model={MODEL_NAME}; targets={','.join(TARGETS)}; locations={n_loc}; "
                   f"champion_run_id={champion_run_id}; temp_rmse_avg={temp_rmse:.3f}; "
                   f"size_mb={size_mb:.1f}; registered={registered}; data_md5={data_md5}")
    except Exception as e:
        finish_run(engine, run_log, "failed", notes=str(e)[:500])
        raise


if __name__ == "__main__":
    main()
