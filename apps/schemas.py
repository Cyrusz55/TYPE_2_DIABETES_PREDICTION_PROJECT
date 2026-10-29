from pydantic import BaseModel, Field
from typing import Literal


class DiabetesInput(BaseModel):
    """
    All 9 features are numeric - the dataset has no categorical columns.
    Defaults may look odd but they are copied from real rows in
    data/Diabetes_type.csv so the Swagger UI "Try it out" works immediately.
    """

    no_of_pregnancy: int = Field(
        ..., ge=0, le=20, description="Number of pregnancies"
    )
    age: int = Field(..., ge=1, le=120, description="Age in years")
    bmi: float = Field(
        ..., ge=10.0, le=70.0, description="Body mass index"
    )
    bp_systolic: int = Field(
        ..., ge=50, le=260, description="Systolic blood pressure (mm Hg)"
    )
    bp_diastolic: int = Field(
        ..., ge=30, le=160, description="Diastolic blood pressure (mm Hg)"
    )
    diabetes_pedigree_function: float = Field(
        ..., ge=0.0, le=3.0, description="Diabetes pedigree function (family history score)"
    )
    insulin: float = Field(
        ..., ge=0.0, le=2000.0, description="Serum insulin (mu U/ml)"
    )
    skin_thickness_mm: float = Field(
        ..., ge=0.0, le=600.0, description="Triceps skin fold thickness (mm)"
    )
    glucose: float = Field(
        ..., ge=0, le=400, description="Plasma glucose concentration (mg/dl)"
    )


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1] = Field(
        ..., description="0=not diabetic, 1=type 2 diabetic"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model probability for the predicted class"
    )
    status: str = "success"


class BatchDiabetesInput(BaseModel):
    """Batch scoring: a list of patients scored in one call."""

    patients: list[DiabetesInput] = Field(
        ..., min_length=1, max_length=500, description="Patients to score"
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse] = Field(
        ..., description="One prediction per input patient, in the same order"
    )
    count: int = Field(..., description="Number of patients scored")
    status: str = "success"
