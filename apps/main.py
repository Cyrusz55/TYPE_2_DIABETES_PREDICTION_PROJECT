"""
main.py
=======
FastAPI application entrypoint for the Type 2 Diabetes Prediction API.

Mirrors the smart-grid / heart-disease project's main.py:
  - Create the FastAPI app
  - Enable CORS
  - Mount the routes under /api/v1
  - Ensure the model is available on startup
  - Serve a simple root landing page (if a frontend exists)

Note:
  - The model is ALREADY trained and saved locally
    (model/best_pipeline_XGBoost.joblib). Startup verifies it is present and
    loads it into cache; it does NOT retrain.
  - The frontend is a single static file (frontend/index.html) served by this
    same FastAPI app at "/". No second process, no npm, no CORS preflight.
"""
import os

import pandas as pd  # noqa: F401  (left available, mirroring the smart-grid main.py)
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from apps.routes import router

from scheduler import start_scheduler

from machine_learning.machine_learning import train_model  # noqa: F401  (scheduler uses it)

load_dotenv()

app = FastAPI(
    title="Type 2 Diabetes Prediction API",
    description=(
        "Predict type 2 diabetes from clinical measurements using an "
        "already-trained XGBoost pipeline."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS - allow specific origins only (never '*' without a reason).
# ---------------------------------------------------------------------------
origins = [
    "http://localhost",
    # The API itself runs on 8001 (see README). The frontend is served from
    # the same origin, so these only matter for direct browser calls.
    "http://localhost:8001",
    "http://127.0.0.1:8001",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount the endpoint router under the /api/v1 prefix.
app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
def startup_event():
    """
    Ensure the model file exists when the API starts.

    The model is trained and stored locally, so we just verify it is present.
    If it is missing, we load it now (so the first /predict request does not
    trigger a slow, unexpected load).
    """
    from apps.model_loader import load_model

    os.makedirs("model", exist_ok=True)

    try:
        load_model()  # loads model/best_pipeline_XGBoost.joblib into the cache
        print("[startup] Model loaded and cached.")
    except FileNotFoundError as e:
        print(f"[startup] WARNING: model not available: {e}")
        print("[startup] Run machine_learning/machine_learning.py to produce "
              "model/best_pipeline_XGBoost.joblib, then restart the API.")
    except Exception as e:
        print(f"[startup] Could not load model: {e}")

    start_scheduler()


@app.get("/")
def root():
    """Root landing page (optional frontend) or a simple JSON message."""
    html_path = "frontend/index.html"

    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Type 2 diabetes prediction API is running"}
