"""HTTP-level tests using FastAPI TestClient."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_with_mock_registry(mock_registry):
    """Patch the singleton registry referenced across the app with our mock."""
    from app import main as main_module
    from app.api.v1 import health as health_module
    from app.api.v1 import predict as predict_module
    from app.models import registry as registry_module

    # Replace fields on the existing singleton instance so every imported
    # reference (`from … import registry`) sees the mock data.
    orig_models  = registry_module.registry.models
    orig_medians = registry_module.registry.medians
    registry_module.registry.models  = mock_registry.models
    registry_module.registry.medians = mock_registry.medians

    # Also re-bind any modules that imported the symbol by name
    main_module.registry    = registry_module.registry
    predict_module.registry = registry_module.registry
    health_module.registry  = registry_module.registry

    yield main_module.app

    registry_module.registry.models  = orig_models
    registry_module.registry.medians = orig_medians


@pytest.fixture
def client(app_with_mock_registry):
    return TestClient(app_with_mock_registry)


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_health_response_shape(self, client):
        r = client.get("/api/v1/health").json()
        assert "status" in r
        assert "modelsLoaded" in r
        assert "ollamaReady" in r
        assert "uptimeSeconds" in r

    def test_health_lists_loaded_models(self, client):
        r = client.get("/api/v1/health").json()
        assert "catboost" in r["modelsLoaded"]
        assert "lightgbm" in r["modelsLoaded"]


class TestPredictEndpoint:

    def test_valid_request_returns_200(self, client):
        r = client.post("/api/v1/predict", json={
            "vitals": {"age": 67, "temp": 38.2, "hr": 118, "sbp": 95, "dbp": 62, "rr": 22, "o2": 94},
            "clinicalNote": "Pacjent z bólem w klatce",
        })
        assert r.status_code == 200, r.text

    def test_response_matches_frontend_contract(self, client):
        r = client.post("/api/v1/predict", json={
            "vitals": {"age": 40, "temp": 36.7, "hr": 80, "sbp": 120, "dbp": 80, "rr": 16, "o2": 99},
            "clinicalNote": "test",
        })
        body = r.json()
        # camelCase keys (matches frontend types.ts)
        for key in ("finalCategory", "confidence", "modelPredictions", "medgemma",
                    "shapTop5", "conflict", "processingTimeMs"):
            assert key in body, f"missing key {key}"

        # Nested
        assert "alertDoctor" in body["conflict"]
        assert "riskFlags" in body["medgemma"]
        assert "keyFindings" in body["medgemma"]

    def test_validation_rejects_invalid_vitals(self, client):
        r = client.post("/api/v1/predict", json={
            "vitals": {"age": 999, "temp": 36.7, "hr": 80, "sbp": 120, "dbp": 80, "rr": 16, "o2": 99},
            "clinicalNote": "",
        })
        assert r.status_code == 422

    def test_validation_rejects_missing_vitals(self, client):
        r = client.post("/api/v1/predict", json={"clinicalNote": "test"})
        assert r.status_code == 422

    def test_long_note_rejected(self, client):
        r = client.post("/api/v1/predict", json={
            "vitals": {"age": 40, "temp": 36.7, "hr": 80, "sbp": 120, "dbp": 80, "rr": 16, "o2": 99},
            "clinicalNote": "x" * 3000,
        })
        assert r.status_code == 422

    def test_root_endpoint(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "SOR-AI" in r.json()["service"]
