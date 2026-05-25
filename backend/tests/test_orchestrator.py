"""End-to-end orchestrator tests using mocked models."""
from __future__ import annotations

import numpy as np
import pytest

from app.models.schemas import PredictRequest, Vitals
from app.pipeline.orchestrator import predict


pytestmark = pytest.mark.asyncio


class TestOrchestratorContract:
    """Validate that the final response always satisfies the frontend contract."""

    async def test_response_has_all_required_fields(self, mock_registry, critical_request):
        resp = await predict(critical_request, mock_registry)

        # Top-level
        assert 0 <= resp.final_category <= 4
        assert 0.0 <= resp.confidence <= 1.0
        assert isinstance(resp.processing_time_ms, int)
        assert resp.processing_time_ms >= 0

        # Models
        assert len(resp.model_predictions) >= 1
        for mp in resp.model_predictions:
            assert mp.model_name
            assert 0 <= mp.category <= 4
            assert len(mp.probabilities) == 5
            assert abs(sum(mp.probabilities) - 1.0) < 0.01

        # MedGemma
        assert 0 <= resp.medgemma.category <= 4
        assert 0.0 <= resp.medgemma.confidence <= 1.0
        assert resp.medgemma.reasoning

        # Conflict info
        assert resp.conflict.severity in ("low", "high")
        assert isinstance(resp.conflict.alert_doctor, bool)

    async def test_critical_case_alerts_doctor(self, mock_registry, critical_request):
        resp = await predict(critical_request, mock_registry)
        # Critical vitals + ML predicting Red → alert
        assert resp.conflict.alert_doctor is True

    async def test_stable_case_basic(self, mock_registry, stable_request):
        resp = await predict(stable_request, mock_registry)
        # Just sanity — shape is correct, no exception
        assert resp.final_category is not None

    async def test_empty_note_does_not_crash(self, mock_registry, empty_note_request):
        resp = await predict(empty_note_request, mock_registry)
        assert resp is not None


class TestSafetyEnforcement:
    async def test_final_never_exceeds_min_of_models(self, mock_registry, critical_request):
        resp = await predict(critical_request, mock_registry)
        min_model = min(mp.category for mp in resp.model_predictions)
        assert resp.final_category <= min_model


class TestProcessingTime:
    async def test_processing_time_is_reasonable(self, mock_registry, critical_request):
        resp = await predict(critical_request, mock_registry)
        assert resp.processing_time_ms < 10_000   # < 10s in mock mode


@pytest.mark.integration
class TestRealModels:
    """Integration tests — use the actual joblib models on disk."""

    async def test_real_models_predict_critical_case(self, real_registry, critical_request):
        resp = await predict(critical_request, real_registry)
        # Chest pain + tachycardia + hypotension → should not be Blue/Green
        assert resp.final_category <= 2

    async def test_real_models_return_shap_values(self, real_registry, critical_request):
        resp = await predict(critical_request, real_registry)
        # SHAP top-5 should be populated for the best loaded model
        assert len(resp.shap_top5) > 0
        for s in resp.shap_top5:
            assert s.feature
            assert s.direction in ("positive", "negative")
