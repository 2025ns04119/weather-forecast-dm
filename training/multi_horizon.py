"""A single model object that holds one fitted regressor per forecast day."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

HORIZON_DAYS = list(range(1, 8))   # day 1 ... day 7


class MultiHorizonModel(BaseEstimator, RegressorMixin):
    """
    models       : {day: fitted regressor}  - predicts temperature at t + 24*day hours
    feature_cols : list of input feature names
    model_types  : {day: "random_forest" / "xgboost" / ...}  (for information)
    """

    def __init__(self, models=None, feature_cols=None, model_types=None):
        self.models = models
        self.feature_cols = feature_cols
        self.model_types = model_types

    def fit(self, X, y=None):   # models are fitted individually during training
        return self

    def predict(self, X) -> np.ndarray:
        """Return an array of shape (n_rows, 7): column d-1 = prediction for day d."""
        X = pd.DataFrame(X)[self.feature_cols].astype(float)
        return np.column_stack([self.models[d].predict(X) for d in sorted(self.models)])

    def forecast(self, feats: pd.DataFrame) -> pd.DataFrame:
        """
        Hourly forecast for the next 7 days.
        feats: the latest 24 hourly feature rows (column 'ts' + feature columns).
        Row at time t with model 'day d' -> forecast for t + 24*d hours.
        """
        feats = feats.sort_values("ts").tail(24)
        preds = self.predict(feats)
        rows = []
        for i, ts in enumerate(feats["ts"]):
            for j, d in enumerate(sorted(self.models)):
                rows.append({"target_ts": ts + pd.Timedelta(hours=24 * d),
                             "day_ahead": d,
                             "predicted_temp": float(preds[i, j])})
        return pd.DataFrame(rows).sort_values("target_ts").reset_index(drop=True)
