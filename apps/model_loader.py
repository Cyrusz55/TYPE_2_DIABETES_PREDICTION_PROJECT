import os
import logging
from pathlib import Path

import joblib
from huggingface_hub import hf_hub_download


logger = logging.getLogger(__name__)

MODEL_FILENAME = "best_pipeline_XGBoost.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", "model/best_pipeline_XGBoost.joblib"))
HF_REPO_ID = "cyrusnx/type2_diabetes_xgboost"
HF_REVISION = "v1"
HF_CACHE_DIR = Path("model") / "huggingface"

_model = None
_load_error = None


def load_model():
    global _model, _load_error
    try:
        if MODEL_PATH.is_file():
            path = MODEL_PATH
        else:
            path = Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=MODEL_FILENAME,
                    revision=HF_REVISION,
                    cache_dir=str(HF_CACHE_DIR),
                    token=os.getenv("HF_TOKEN") or None,
                )
            )
        _model = joblib.load(path)
        _load_error = None
        logger.info("Model loaded from %s", path)
    except Exception as error:
        _model = None
        _load_error = error
        logger.error("Could not load the diabetes model: %s", error)
    return _model


def get_model():
    if _model is None:
        load_model()
    return _model


def model_loaded():
    return _model is not None


def model_load_error():
    return _load_error
