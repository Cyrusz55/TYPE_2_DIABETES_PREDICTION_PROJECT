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
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from apps.routes import router
from apps.model_loader import load_model

load_dotenv()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Type 2 Diabetes Prediction API",
    description=(
        "Predict type 2 diabetes from clinical measurements using an "
        "already-trained XGBoost pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
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


@app.get("/")
def root():
    """Root landing page (optional frontend) or a simple JSON message."""
    html_path = "frontend/index.html"

    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Type 2 diabetes prediction API is running"}
