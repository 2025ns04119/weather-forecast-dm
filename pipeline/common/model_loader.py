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

# Models saved before the code moved into pipeline/ refer to the old module path
# "training.multi_horizon". This alias lets those saved models still load.
import sys
import types

from pipeline.model_training import multi_horizon as _multi_horizon

sys.modules.setdefault("training", types.ModuleType("training"))
sys.modules.setdefault("training.multi_horizon", _multi_horizon)

MODEL_NAME = "weather-7day"   # multi-target: temp, humidity, wind, cloud, rain


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
