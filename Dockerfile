FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/huggingface \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY apps ./apps
COPY frontend ./frontend
COPY machine_learning ./machine_learning

RUN mkdir -p /app/model /opt/huggingface && \
    python -c "from pathlib import Path; from huggingface_hub import hf_hub_download; source = Path(hf_hub_download(repo_id='cyrusnx/type2_diabetes_xgboost', filename='best_pipeline_XGBoost.joblib', revision='v1', cache_dir='/app/model/huggingface', token=None)); Path('/app/model/best_pipeline_XGBoost.joblib').write_bytes(source.read_bytes())" && \
    python -c "from apps.model_loader import load_model; assert load_model() is not None"

RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app /opt/huggingface
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD-SHELL python -c "import json, os, urllib.request; port = os.environ.get('PORT', '8000'); data = json.load(urllib.request.urlopen('http://127.0.0.1:' + port + '/api/v1/health')); raise SystemExit(0 if data.get('model_loaded') is True else 1)"

CMD ["sh", "-c", "exec uvicorn apps.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
