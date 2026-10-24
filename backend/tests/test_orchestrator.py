"""End-to-end orchestrator tests using mocked models."""
from __future__ import annotations

import numpy as np
import pandas as pd
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
        # Critical vitals + ML predicting ESI 1 → alert
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


class TestPatientContext:
    async def test_optional_fields_reach_feature_row(self, mock_registry, monkeypatch):
        from app.pipeline import orchestrator

        captured: dict = {}
        real = orchestrator.build_feature_row

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(orchestrator, "build_feature_row", spy)
        request = PredictRequest(
            vitals=Vitals(age=50, temp=36.8, hr=90, sbp=130, dbp=80, rr=16, o2=97),
            clinicalNote="ból brzucha", sex="F", arrivalMode="ZRM", o2Device=True,
        )
        await predict(request, mock_registry)
        assert captured["is_male"] == "F"
        assert captured["arrival_mode"] == "ZRM"
        assert captured["o2_device"] is True

    async def test_feature_row_encodes_context(self, mock_registry, monkeypatch):
        from app.pipeline import orchestrator

        rows: list = []
        real = orchestrator.build_feature_row
        monkeypatch.setattr(orchestrator, "build_feature_row",
                            lambda *a, **k: rows.append(real(*a, **k)) or rows[-1])
        request = PredictRequest(
            vitals=Vitals(age=50, temp=36.8, hr=90, sbp=130, dbp=80, rr=16, o2=97),
            sex="M", arrivalMode="ambulance", o2Device=False,
        )
        await predict(request, mock_registry)
        row = rows[0].iloc[0]
        assert row["is_male"] == 1.0
        assert row["arrival_ambulance"] == 1.0 and row["arrival_self"] == 0.0
        assert row["triage_vital_o2_device"] == 0.0


class TestEsiSafetyInPipeline:
    async def test_point_a_vitals_force_esi1(self, mock_registry):
        request = PredictRequest(
            vitals=Vitals(age=60, temp=36.8, hr=160, sbp=130, dbp=80, rr=16, o2=97),
            clinicalNote="",
        )
        # mock models predict ESI 1 already → make them predict ESI 3
        for bundle in mock_registry.models.values():
            bundle["model"].predict_proba.return_value = np.array([[0.02, 0.03, 0.85, 0.05, 0.05]])
        resp = await predict(request, mock_registry)
        assert resp.final_category == 0
        assert resp.conflict.alert_doctor is True
        assert "ESI 1 (czerwony)" in resp.conflict.message

    async def test_escalation_kept_next_to_synthesis_summary(self, mock_registry):
        probas = {"catboost": [0.02, 0.03, 0.85, 0.05, 0.05], "lightgbm": [0.02, 0.03, 0.05, 0.05, 0.85]}
        for name, proba in probas.items():
            mock_registry.models[name]["model"].predict_proba.return_value = np.array([proba])
        request = PredictRequest(
            vitals=Vitals(age=60, temp=36.8, hr=105, sbp=130, dbp=80, rr=16, o2=97),
            clinicalNote="",
        )
        resp = await predict(request, mock_registry)   # spread ESI 3 vs ESI 5 → synthesis runs
        assert resp.final_category == 1                # danger zone HR > 100 → minimum ESI 2
        assert "minimum ESI 2 (pomarańczowy)" in resp.conflict.message
        assert "Rekomenduj" in resp.conflict.message or "fallback" in resp.conflict.message


class TestShapSelection:
    async def test_shap_falls_back_to_next_tree_model(self, critical_request):
        """Highest-weight tree model without a usable explainer → next tree model explains."""
        from unittest.mock import MagicMock

        from lightgbm import LGBMClassifier

        from app.models.ensemble import EnsembleConfig
        from app.models.registry import ModelRegistry

        features = ["age", "triage_vital_hr", "triage_vital_sbp", "triage_vital_o2"]
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(200, 4)) * [15, 20, 20, 3] + [50, 90, 125, 96], columns=features)
        y = np.arange(200) % 5
        lgbm = LGBMClassifier(n_estimators=20, num_leaves=4, n_jobs=1, verbose=-1).fit(X, y)

        broken = MagicMock()                       # predicts fine, TreeExplainer cannot handle it
        broken.predict_proba.return_value = np.array([[0.1, 0.6, 0.2, 0.05, 0.05]])

        reg = ModelRegistry()
        reg.models = {
            "lightgbm": {"model": lgbm, "feature_names": features, "classes": [0, 1, 2, 3, 4], "path": "(t)"},
            "xgboost": {"model": broken, "feature_names": features, "classes": [0, 1, 2, 3, 4], "path": "(t)"},
        }
        reg.ensemble = EnsembleConfig(weights={"xgboost": 0.7, "lightgbm": 0.3}, source="test")
        resp = await predict(critical_request, reg)
        assert 1 <= len(resp.shap_top5) <= 4
        assert {s.feature for s in resp.shap_top5} <= set(features)


class TestProcessingTime:
    async def test_processing_time_is_reasonable(self, mock_registry, critical_request):
        resp = await predict(critical_request, mock_registry)
        assert resp.processing_time_ms < 10_000   # < 10s in mock mode


@pytest.mark.integration
class TestRealModels:
    """Integration tests — use the actual joblib models on disk."""

    async def test_real_models_predict_critical_case(self, real_registry, critical_request):
        resp = await predict(critical_request, real_registry)
        # Chest pain + tachycardia + borderline SBP → should not be ESI 4–5
        assert resp.final_category <= 2

    async def test_real_models_return_shap_values(self, real_registry, critical_request):
        resp = await predict(critical_request, real_registry)
        # SHAP top-5 should be populated for the best loaded model
        assert len(resp.shap_top5) > 0
        for s in resp.shap_top5:
            assert s.feature
            assert s.direction in ("positive", "negative")
