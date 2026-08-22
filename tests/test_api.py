from pathlib import Path

import pytest

from churn_intelligence.api.app import create_app


def test_health_endpoint():
    client = create_app({"TESTING": True}).test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predict_rejects_invalid_body():
    if not (Path("models/churn_pipeline.joblib").exists()):
        pytest.skip("Full model is created by the local training pipeline, not by unit CI.")
    client = create_app({"TESTING": True}).test_client()
    response = client.post("/predict", json={"tenure": -1})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_dashboard_and_result_assets():
    client = create_app({"TESTING": True}).test_client()
    page = client.get("/")
    assert page.status_code == 200
    assert b"CUSTOMER RISK EXPLORER" in page.data
    assert b"MODEL PERFORMANCE" in page.data
    assert b"RETENTION QUEUE" in page.data
    assert client.get("/assets/figures/calibration_curve.png").status_code == 200
    assert client.get("/api/customer/7590-VHVEG").status_code == 200
