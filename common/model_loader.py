"""Load the champion model from the MLflow Model Registry (DagsHub)."""
import os
import warnings

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore", category=UserWarning)

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

MODEL_NAME = "weather-temp-7day"


def load_champion(model_name: str = MODEL_NAME):
    """
    Return (model, version). Looks up the 'champion' alias; falls back to the
    newest version. Downloads the files directly from the version's source
    (works on DagsHub).
    """
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
    except Exception:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise RuntimeError(f"No registered versions of '{model_name}' - run training first")
        mv = max(versions, key=lambda v: int(v.version))
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=mv.source)
    model = mlflow.sklearn.load_model(local_path)
    return model, str(mv.version)
