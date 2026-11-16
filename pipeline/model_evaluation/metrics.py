"""
Model evaluation metrics and baselines (used by training to compare and select models).
Regression: MAE, RMSE, R2. Rain (classification): Brier, AUC, precision, recall, F1. Baselines: persistence / climatology.
"""
import numpy as np
from sklearn.metrics import (brier_score_loss, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)

from pipeline.model_training.multi_horizon import label

RAIN_DECISION = 0.5     # predicted probability above which we say "rain" for precision / recall / F1


def score(kind, y_true, pred) -> dict:
    """Evaluation metrics for one target and forecast day."""
    if kind == "clf":
        out = {"brier": float(brier_score_loss(y_true, pred))}
        try:
            out["auc"] = float(roc_auc_score(y_true, pred))
        except ValueError:          # only one class present in the test period
            out["auc"] = float("nan")
        said_rain = np.asarray(pred) >= RAIN_DECISION
        y = np.asarray(y_true).astype(int)
        out["precision"] = float(precision_score(y, said_rain, zero_division=0))
        out["recall"] = float(recall_score(y, said_rain, zero_division=0))
        out["f1"] = float(f1_score(y, said_rain, zero_division=0))
        return out
    return {"mae": float(mean_absolute_error(y_true, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
            "r2": float(r2_score(y_true, pred))}


def selection_metric(kind) -> str:
    """Metric used to pick the best candidate (lower is better)."""
    return "brier" if kind == "clf" else "rmse"


def baseline(target, kind, train, test):
    """Persistence for regression (value now); climatology (training rain rate) for rain."""
    if kind == "clf":
        return np.full(len(test), train[label(target, 1)].mean())
    now_col = {"temp": "temp_lag_1h", "humidity": "humidity", "wind": "wind_speed", "cloud": "cloud_cover"}[target]
    return test[now_col].to_numpy()
