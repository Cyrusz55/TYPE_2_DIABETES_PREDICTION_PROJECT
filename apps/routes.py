from fastapi import APIRouter, HTTPException
import os
import joblib
import pandas as pd

from apps.schemas import (
    DiabetesInput,
    PredictionResponse,
    BatchDiabetesInput,
    BatchPredictionResponse,
)

from machine_learning.machine_learning import (
    load_model,
    predict_diabetes,
    NUM_COLS,
)

router = APIRouter()

# Maps the API's snake_case fields onto the CSV's original column names.
# The pipeline was fitted on the original names, so this mapping is required.
FIELD_TO_COLUMN = {
    "no_of_pregnancy": "No. of Pregnancy",
    "age": "Age",
    "bmi": "BMI",
    "bp_systolic": "BP(Systolic)",
    "bp_diastolic": "BP(Diastolic)",
    "diabetes_pedigree_function": "DiabetesPedigreeFunction",
    "insulin": "Insulin",
    "skin_thickness_mm": "Skin Thickness(mm)",
    "glucose": "Glucose",
}


def _to_frame(payload: dict) -> pd.DataFrame:
    """Map API snake_case fields onto the original CSV column names.

    The pipeline was fitted on the original names and selects columns BY NAME,
    so this mapping is required. NUM_COLS ordering is applied for clarity.
    """
    row = {FIELD_TO_COLUMN[k]: v for k, v in payload.items()}
    return pd.DataFrame([row])[NUM_COLS]


@router.get("/health")
def health_check():
    return {"status": "ok", "message": "API is healthy"}


@router.post("/predict", response_model=PredictionResponse)
def predict(patient: DiabetesInput):
    try:
        model = load_model()
        new_data = _to_frame(patient.model_dump())

        prediction = int(predict_diabetes(model, new_data)[0])
        confidence = float(model.predict_proba(new_data)[0][prediction])

        return PredictionResponse(prediction=prediction, confidence=confidence)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not yet trained. Run training first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchDiabetesInput):
    """Score many patients in one call (used by the CSV-upload path)."""
    try:
        model = load_model()
        frame = pd.DataFrame(
            [_to_frame(p.model_dump()).iloc[0] for p in payload.patients]
        )[NUM_COLS]

        preds = model.predict(frame)
        probas = model.predict_proba(frame)

        results = [
            PredictionResponse(
                prediction=int(preds[i]),
                confidence=float(probas[i][int(preds[i])]),
            )
            for i in range(len(preds))
        ]
        return BatchPredictionResponse(predictions=results, count=len(results))
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not yet trained. Run training first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
def model_info():
    path = os.getenv("MODEL_PATH", "model/best_pipeline_XGBoost.joblib")

    if not os.path.exists(path):
        return {"status": "no model found"}

    model = joblib.load(path)
    return {
        "model_type": type(model).__name__,
        "model_path": path,
        "features": NUM_COLS,
        "target": "Type-2 Diabetic",
    }
