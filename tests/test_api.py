from fastapi.testclient import TestClient

from apps.main import app
from apps.model_loader import get_model


VALID_PAYLOAD = {
    "no_of_pregnancy": 3,
    "age": 50,
    "bmi": 28.5,
    "bp_systolic": 130,
    "bp_diastolic": 85,
    "diabetes_pedigree_function": 0.35,
    "insulin": 95,
    "skin_thickness_mm": 23,
    "glucose": 146,
}


def test_health_check_returns_ok():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_root_serves_html_interface():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "<html" in response.text.lower()


def test_valid_prediction_returns_real_response_fields():
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert {"prediction", "confidence", "status"} <= body.keys()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["confidence"] <= 1


def test_invalid_prediction_returns_validation_error():
    payload = {**VALID_PAYLOAD, "no_of_pregnancy": -1}

    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422


def test_each_missing_field_returns_validation_error():
    for field in VALID_PAYLOAD:
        payload = {key: value for key, value in VALID_PAYLOAD.items() if key != field}

        with TestClient(app) as client:
            response = client.post("/api/v1/predict", json=payload)

        assert response.status_code == 422


def test_model_loader_returns_working_model():
    model = get_model()

    assert model is not None
    assert hasattr(model, "predict")
