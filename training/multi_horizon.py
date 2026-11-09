"""
One model object that forecasts EVERY weather variable for the next 7 days.

    models[target][day]  -> fitted model predicting `target` at t + 24*day hours
    kinds[target]        -> "reg" (regression) or "clf" (classification -> probability)

Targets: temp (°C), humidity (%), wind (km/h), cloud (%), rain (probability of >= 0.1 mm/h)
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

HORIZON_DAYS = list(range(1, 8))          # day 1 ... day 7
TARGETS = {                               # target -> (clean column, kind)
    "temp": ("temperature_2m", "reg"),
    "humidity": ("relative_humidity_2m", "reg"),
    "wind": ("wind_speed_10m", "reg"),
    "cloud": ("cloud_cover", "reg"),
    "rain": ("precipitation", "clf"),
}
RAIN_THRESHOLD_MM = 0.1
CLIP = {"humidity": (0, 100), "cloud": (0, 100), "wind": (0, None)}


class MultiHorizonModel(BaseEstimator, RegressorMixin):
    def __init__(self, models=None, kinds=None, feature_cols=None, model_types=None):
        self.models = models
        self.kinds = kinds
        self.feature_cols = feature_cols
        self.model_types = model_types

    def fit(self, X, y=None):          # the parts are fitted individually during training
        return self

    def _predict_target(self, target: str, X: pd.DataFrame) -> np.ndarray:
        """(n_rows, 7): column d-1 = forecast for day d."""
        cols = []
        for d in sorted(self.models[target]):
            m = self.models[target][d]
            p = m.predict_proba(X)[:, 1] if self.kinds[target] == "clf" else m.predict(X)
            lo, hi = CLIP.get(target, (None, None))
            cols.append(np.clip(p, lo, hi) if (lo is not None or hi is not None) else p)
        return np.column_stack(cols)

    def predict(self, X) -> np.ndarray:
        """Temperature forecast (n_rows, 7) - kept for MLflow's sklearn interface."""
        X = pd.DataFrame(X)[self.feature_cols].astype(float)
        return self._predict_target("temp", X)

    def forecast(self, feats: pd.DataFrame) -> pd.DataFrame:
        """
        Hourly forecast for the next 7 days for every variable.
        feats: the latest 24 hourly feature rows (column 'ts' + feature columns).
        Row at time t with the day-d models -> forecast for t + 24*d hours.
        """
        feats = feats.sort_values("ts").tail(24)
        X = feats[self.feature_cols].astype(float)
        preds = {t: self._predict_target(t, X) for t in self.models}
        rows = []
        for i, ts in enumerate(feats["ts"]):
            for j, d in enumerate(HORIZON_DAYS):
                row = {"target_ts": ts + pd.Timedelta(hours=24 * d), "day_ahead": d}
                for t in preds:
                    row[t] = float(preds[t][i, j])
                rows.append(row)
        out = pd.DataFrame(rows).sort_values("target_ts").reset_index(drop=True)
        return out.rename(columns={"temp": "predicted_temp", "rain": "rain_prob"})
