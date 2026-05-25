"""Tests for the hardcoded safety rules — these are critical clinical logic."""
from __future__ import annotations

import pytest

from app.models.schemas import MedGemmaAssessment, ModelPrediction, Vitals
from app.pipeline.safety_rules import apply_safety_rules


def _mp(model: str, category: int, confidence: float = 0.9) -> ModelPrediction:
    probs = [0.05] * 5
    probs[category] = confidence
    return ModelPrediction(
        modelName=model, category=category, probabilities=probs, confidence=confidence,
    )


def _mg(category: int, confidence: float = 0.9) -> MedGemmaAssessment:
    return MedGemmaAssessment(
        category=category, confidence=confidence,
        reasoning="test", riskFlags=[], keyFindings=[],
    )


@pytest.fixture
def stable_vitals() -> Vitals:
    return Vitals(age=40, temp=36.7, hr=80, sbp=120, dbp=80, rr=16, o2=99)


@pytest.fixture
def critical_vitals() -> Vitals:
    return Vitals(age=72, temp=39.0, hr=140, sbp=85, dbp=55, rr=30, o2=87)


class TestRule1NeverDowngrade:
    def test_final_is_lowered_to_minimum_of_models(self, stable_vitals):
        models = [_mp("a", 1), _mp("b", 2), _mp("c", 3)]
        result = apply_safety_rules(
            final_category=2, confidence=0.85,
            model_predictions=models, medgemma=_mg(2),
            vitals=stable_vitals,
        )
        assert result.final_category == 1  # the most urgent across models

    def test_already_minimum_unchanged(self, stable_vitals):
        models = [_mp("a", 2), _mp("b", 2)]
        result = apply_safety_rules(
            final_category=2, confidence=0.9,
            model_predictions=models, medgemma=_mg(2),
            vitals=stable_vitals,
        )
        assert result.final_category == 2


class TestRule2ModelConflict:
    def test_spread_of_2_triggers_alert(self, stable_vitals):
        models = [_mp("a", 0), _mp("b", 2)]  # Red vs Yellow → spread 2
        result = apply_safety_rules(
            final_category=1, confidence=0.7,
            model_predictions=models, medgemma=_mg(1),
            vitals=stable_vitals,
            conflict_threshold=2,
        )
        assert result.detected is True
        assert result.severity == "high"
        assert result.alert_doctor is True

    def test_spread_of_1_low_severity(self, stable_vitals):
        models = [_mp("a", 1), _mp("b", 2)]
        result = apply_safety_rules(
            final_category=1, confidence=0.85,
            model_predictions=models, medgemma=_mg(1),
            vitals=stable_vitals,
            conflict_threshold=2,
        )
        assert result.detected is True
        assert result.severity == "low"

    def test_full_agreement_no_alert(self, stable_vitals):
        models = [_mp("a", 2), _mp("b", 2), _mp("c", 2)]
        result = apply_safety_rules(
            final_category=2, confidence=0.95,
            model_predictions=models, medgemma=_mg(2),
            vitals=stable_vitals,
        )
        assert result.detected is False
        assert result.alert_doctor is False


class TestRule3LowConfidence:
    def test_medgemma_below_threshold_alerts(self, stable_vitals):
        result = apply_safety_rules(
            final_category=3, confidence=0.8,
            model_predictions=[_mp("a", 3)],
            medgemma=_mg(3, confidence=0.3),
            vitals=stable_vitals,
            confidence_threshold=0.6,
        )
        assert result.alert_doctor is True
        assert "pewnoś" in result.message.lower()


class TestRule4CriticalVitals:
    def test_low_sbp_forces_orange_or_better(self, critical_vitals):
        # ML says Yellow, but vitals are critical
        result = apply_safety_rules(
            final_category=3, confidence=0.85,
            model_predictions=[_mp("a", 3)],
            medgemma=_mg(3),
            vitals=critical_vitals,
        )
        assert result.final_category <= 1
        assert result.alert_doctor is True

    def test_severe_hypoxia_forces_escalation(self, stable_vitals):
        v = stable_vitals.model_copy(update={"o2": 85.0})
        result = apply_safety_rules(
            final_category=2, confidence=0.85,
            model_predictions=[_mp("a", 2)],
            medgemma=_mg(2),
            vitals=v,
        )
        assert result.final_category <= 1

    def test_normal_vitals_no_forced_escalation(self, stable_vitals):
        result = apply_safety_rules(
            final_category=3, confidence=0.85,
            model_predictions=[_mp("a", 3)],
            medgemma=_mg(3),
            vitals=stable_vitals,
        )
        assert result.final_category == 3


class TestRule5RedAlwaysAlerts:
    def test_any_model_red_triggers_alert(self, stable_vitals):
        models = [_mp("a", 0), _mp("b", 1)]
        result = apply_safety_rules(
            final_category=0, confidence=0.95,
            model_predictions=models, medgemma=_mg(0),
            vitals=stable_vitals,
        )
        assert result.alert_doctor is True


class TestRule6MLvsLLMConflict:
    def test_ml_red_llm_green_alerts(self, stable_vitals):
        models = [_mp("a", 0)]                    # ML says Red
        result = apply_safety_rules(
            final_category=0, confidence=0.9,
            model_predictions=models,
            medgemma=_mg(3, confidence=0.9),      # MedGemma says Green
            vitals=stable_vitals,
        )
        assert result.detected is True
        assert result.alert_doctor is True
