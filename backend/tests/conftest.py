"""Shared pytest fixtures for the SOR-AI backend test suite."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.models.registry import ModelRegistry
from app.models.schemas import PredictRequest, Vitals


# ── Event loop (session-scoped for asyncio mode=auto) ─────────────────

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Sample requests ───────────────────────────────────────────────────

@pytest.fixture
def critical_request() -> PredictRequest:
    return PredictRequest(
        vitals=Vitals(age=67, temp=38.2, hr=118, sbp=95, dbp=62, rr=22, o2=94),
        clinicalNote="Pacjent blady, spocony, ból w klatce piersiowej, duszność",
    )


@pytest.fixture
def stable_request() -> PredictRequest:
    return PredictRequest(
        vitals=Vitals(age=30, temp=36.7, hr=72, sbp=120, dbp=80, rr=14, o2=99),
        clinicalNote="Wizyta kontrolna",
    )


@pytest.fixture
def empty_note_request() -> PredictRequest:
    return PredictRequest(
        vitals=Vitals(age=45, temp=36.6, hr=80, sbp=120, dbp=80, rr=16, o2=98),
        clinicalNote="",
    )


# ── Mock model registry (no joblib I/O) ───────────────────────────────

def _make_mock_model(predicted_class: int = 0, confidence: float = 0.85):
    """Returns a mock that mimics sklearn predict_proba."""
    model = MagicMock()
    probs = np.zeros(5)
    probs[predicted_class] = confidence
    remaining = 1.0 - confidence
    for i in range(5):
        if i != predicted_class:
            probs[i] = remaining / 4
    model.predict_proba.return_value = np.array([probs])
    return model


@pytest.fixture
def mock_registry() -> ModelRegistry:
    """Registry with two mock models pre-loaded. Loads the real medians json so
    that build_feature_row + engineer_features have all the columns they need.
    """
    backend_root = Path(__file__).resolve().parents[1]
    medians_path = backend_root / "models" / "imputation_medians.json"

    if medians_path.exists():
        payload = json.loads(medians_path.read_text())
        medians = payload["medians"]
    else:
        # Minimal fallback (no real medians on disk)
        medians = {
            "triage_vital_hr": 84.0, "triage_vital_sbp": 131.0,
            "triage_vital_dbp": 80.0, "triage_vital_rr": 18.0,
            "triage_vital_o2": 98.0, "triage_vital_temp": 98.0,
            "age": 49.0,
        }

    feature_names = list(medians.keys())

    reg = ModelRegistry()
    reg.medians = medians
    reg.feature_set = "triage_only"
    reg.models = {
        "catboost": {
            "model": _make_mock_model(0, 0.85),
            "feature_names": feature_names,
            "path": "(mock)",
        },
        "lightgbm": {
            "model": _make_mock_model(0, 0.80),
            "feature_names": feature_names,
            "path": "(mock)",
        },
    }
    return reg


@pytest.fixture
def real_registry() -> ModelRegistry:
    """Load actual joblib models from disk — used for integration tests."""
    reg = ModelRegistry()
    backend_root = Path(__file__).resolve().parents[1]
    reg.load_from_dir(backend_root / "models")
    if not reg.models:
        pytest.skip("No real joblib models on disk — skipping integration test")
    return reg
