"""Tests for the hardcoded safety rules — these are critical clinical logic.

Class index 0–4 = ESI 1–5 (TOPSOR: red, orange, yellow, green, blue).
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from types import ModuleType

import pytest

from app.config import get_settings
from app.models.schemas import MedGemmaAssessment, ModelPrediction, Vitals
from app.pipeline.safety_rules import (
    COMPLAINT_RULES,
    ESI_DANGER_ZONE_ADULT,
    apply_safety_rules,
    esi_label,
    esi_rule_floor,
    esi_rule_triggers,
)


def _mp(model: str, category: int, confidence: float = 0.9) -> ModelPrediction:
    probs = [0.025] * 5
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


class TestEsiLabels:
    def test_labels_use_esi_levels_and_topsor_colours(self):
        assert esi_label(0) == "ESI 1 (czerwony)"
        assert esi_label(1) == "ESI 2 (pomarańczowy)"
        assert esi_label(4) == "ESI 5 (niebieski)"


class TestRule1NeverDowngrade:
    def test_final_is_lowered_to_minimum_of_models(self, stable_vitals):
        models = [_mp("a", 1), _mp("b", 2), _mp("c", 3)]
        result = apply_safety_rules(
            final_category=2, confidence=0.85,
            model_predictions=models, medgemma=_mg(2),
            vitals=stable_vitals,
        )
        assert result.final_category == 1  # the most urgent across models
        assert "ESI 2 (pomarańczowy)" in result.message

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
        models = [_mp("a", 0), _mp("b", 2)]  # ESI 1 vs ESI 3 → spread 2
        result = apply_safety_rules(
            final_category=1, confidence=0.7,
            model_predictions=models, medgemma=_mg(1),
            vitals=stable_vitals,
            conflict_threshold=2,
        )
        assert result.detected is True
        assert result.severity == "high"
        assert result.alert_doctor is True
        assert "poziomów ESI" in result.message

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
        assert result.message == "Wszystkie modele zgodne"
        assert result.escalations == []


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


class TestRule4EsiPointA:
    """Life-saving intervention needed now → ESI 1."""

    @pytest.mark.parametrize("update", [
        {"o2": 84.0}, {"sbp": 79, "dbp": 50}, {"hr": 151}, {"hr": 39},
        {"rr": 36}, {"rr": 7}, {"temp": 41.0}, {"temp": 31.9},
    ])
    def test_extreme_vitals_force_esi1(self, stable_vitals, update):
        v = stable_vitals.model_copy(update=update)
        result = apply_safety_rules(
            final_category=3, confidence=0.85,
            model_predictions=[_mp("a", 3)], medgemma=_mg(3), vitals=v,
        )
        assert result.final_category == 0
        assert result.alert_doctor is True
        assert "ESI 1 (czerwony)" in result.message
        assert result.escalations and "Punkt A" in result.escalations[0]

    @pytest.mark.parametrize("cc", ["cc_cardiacarrest", "cc_unresponsive"])
    def test_life_threatening_complaint_forces_esi1(self, stable_vitals, cc):
        result = apply_safety_rules(
            final_category=2, confidence=0.85,
            model_predictions=[_mp("a", 2)], medgemma=_mg(2),
            vitals=stable_vitals, cc_flags={cc: 1},
        )
        assert result.final_category == 0
        assert result.alert_doctor is True

    def test_esi1_already_predicted_still_alerts(self):
        v = Vitals(age=60, temp=36.6, hr=80, sbp=120, dbp=80, rr=16, o2=80)
        result = apply_safety_rules(
            final_category=0, confidence=0.9,
            model_predictions=[_mp("a", 0)], medgemma=_mg(0), vitals=v,
        )
        assert result.final_category == 0
        assert result.alert_doctor is True
        assert result.escalations == []          # nothing to escalate


class TestRule4EsiPointsBD:
    """High-risk situation / danger-zone vitals → at least ESI 2."""

    def test_critical_vitals_force_esi2(self, critical_vitals):
        # SBP 85 (point B) + HR 140 / RR 30 / SpO2 87 (danger zone) — no point A
        result = apply_safety_rules(
            final_category=3, confidence=0.85,
            model_predictions=[_mp("a", 3)], medgemma=_mg(3), vitals=critical_vitals,
        )
        assert result.final_category == 1
        assert result.alert_doctor is True
        assert "minimum ESI 2 (pomarańczowy)" in result.message

    @pytest.mark.parametrize("update", [{"hr": 101}, {"rr": 21}, {"o2": 91.0}, {"sbp": 89, "dbp": 60}])
    def test_danger_zone_or_hypotension_forces_esi2(self, stable_vitals, update):
        v = stable_vitals.model_copy(update=update)
        result = apply_safety_rules(
            final_category=2, confidence=0.85,
            model_predictions=[_mp("a", 2)], medgemma=_mg(2), vitals=v,
        )
        assert result.final_category == 1
        assert result.escalations

    @pytest.mark.parametrize("update", [{"hr": 100}, {"rr": 20}, {"o2": 92.0}, {"sbp": 90, "dbp": 60}])
    def test_thresholds_are_strict(self, stable_vitals, update):
        v = stable_vitals.model_copy(update=update)
        result = apply_safety_rules(
            final_category=3, confidence=0.85,
            model_predictions=[_mp("a", 3)], medgemma=_mg(3), vitals=v,
        )
        assert result.final_category == 3

    @pytest.mark.parametrize("cc", [
        "cc_respiratorydistress", "cc_strokealert", "cc_trauma_activation",
        "cc_suicidal", "cc_homicidal", "cc_overdose_intentional",
    ])
    def test_high_risk_complaint_forces_esi2(self, stable_vitals, cc):
        result = apply_safety_rules(
            final_category=3, confidence=0.85,
            model_predictions=[_mp("a", 3)], medgemma=_mg(3),
            vitals=stable_vitals, cc_flags={cc: 1},
        )
        assert result.final_category == 1

    def test_severe_pain_forces_esi2(self, stable_vitals):
        result = apply_safety_rules(
            final_category=3, confidence=0.85,
            model_predictions=[_mp("a", 3)], medgemma=_mg(3),
            vitals=stable_vitals, pain=8.0,
        )
        assert result.final_category == 1
        assert "silny ból" in result.message

    def test_danger_zone_not_applied_to_children(self):
        child = Vitals(age=4, temp=36.8, hr=120, sbp=100, dbp=60, rr=26, o2=98)
        assert esi_rule_triggers(child) == []

    def test_more_urgent_prediction_is_kept(self, critical_vitals):
        result = apply_safety_rules(
            final_category=0, confidence=0.85,
            model_predictions=[_mp("a", 0)], medgemma=_mg(0), vitals=critical_vitals,
        )
        assert result.final_category == 0
        assert result.escalations == []

    def test_normal_vitals_no_forced_escalation(self, stable_vitals):
        result = apply_safety_rules(
            final_category=3, confidence=0.85,
            model_predictions=[_mp("a", 3)],
            medgemma=_mg(3),
            vitals=stable_vitals,
        )
        assert result.final_category == 3
        assert esi_rule_floor(esi_rule_triggers(stable_vitals)) is None


class TestRule5Esi1AlwaysAlerts:
    def test_any_model_esi1_triggers_alert(self, stable_vitals):
        models = [_mp("a", 0), _mp("b", 1)]
        result = apply_safety_rules(
            final_category=0, confidence=0.95,
            model_predictions=models, medgemma=_mg(0),
            vitals=stable_vitals,
        )
        assert result.alert_doctor is True


class TestRule6MLvsLLMConflict:
    def test_ml_esi1_llm_esi4_alerts(self, stable_vitals):
        models = [_mp("a", 0)]                    # ML says ESI 1
        result = apply_safety_rules(
            final_category=0, confidence=0.9,
            model_predictions=models,
            medgemma=_mg(3, confidence=0.9),      # MedGemma says ESI 4
            vitals=stable_vitals,
        )
        assert result.detected is True
        assert result.alert_doctor is True


# ── Consistency with training/src/explain/esi_rules.py ───────────────

@pytest.fixture(scope="module")
def training_esi_rules() -> ModuleType:
    """training/src/explain/esi_rules.py loaded by path (without src.explain's heavy imports)."""
    training_root = get_settings().project_root / "training"
    path = training_root / "src" / "explain" / "esi_rules.py"
    if not path.exists():
        pytest.skip(f"{path} not found")
    if str(training_root) not in sys.path:
        sys.path.insert(0, str(training_root))
    spec = importlib.util.spec_from_file_location("_training_esi_rules", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses look the module up while executing it
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except ImportError as exc:
        sys.modules.pop(spec.name, None)
        pytest.skip(f"training esi_rules not importable: {exc}")
    return module


class TestConsistencyWithTraining:
    def test_danger_zone_thresholds_match(self, training_esi_rules):
        assert ESI_DANGER_ZONE_ADULT == training_esi_rules.ESI_DANGER_ZONE_ADULT

    def test_complaint_levels_match(self, training_esi_rules):
        training = {k: v[0] for k, v in training_esi_rules.COMPLAINT_DISCRIMINATORS.items()}
        assert {k: v[0] for k, v in COMPLAINT_RULES.items()} == training

    def test_vital_rules_match_on_threshold_grid(self, training_esi_rules):
        """Same ESI level (1, 2 or none) as training rule_based_triage() for adult vitals."""
        grid = {
            "hr": [39, 40, 100, 101, 150, 151],
            "sbp": [79, 80, 89, 90],
            "rr": [7, 8, 20, 21, 35, 36],
            "o2": [84.0, 85.0, 91.0, 92.0],
            "temp": [31.9, 32.0, 40.9, 41.0],
        }
        mismatches = []
        for hr, sbp, rr, o2, temp in itertools.product(*grid.values()):
            v = Vitals(age=50, temp=temp, hr=hr, sbp=sbp, dbp=45, rr=rr, o2=o2)
            ours = esi_rule_floor(esi_rule_triggers(v, pain=None))
            theirs = training_esi_rules.rule_based_triage({
                "age": 50, "triage_vital_hr": hr, "triage_vital_sbp": sbp, "triage_vital_dbp": 45,
                "triage_vital_rr": rr, "triage_vital_o2": o2, "triage_vital_temp": temp,
            })["suggested_class_idx"]
            if ours != theirs:
                mismatches.append((v, ours, theirs))
        assert not mismatches, mismatches[:5]

    @pytest.mark.parametrize("pain,expected", [(6.0, None), (7.0, 1)])
    def test_pain_rule_matches(self, training_esi_rules, stable_vitals, pain, expected):
        ours = esi_rule_floor(esi_rule_triggers(stable_vitals, pain=pain))
        theirs = training_esi_rules.rule_based_triage({
            "triage_vital_hr": 80, "triage_vital_sbp": 120, "triage_vital_rr": 16,
            "triage_vital_o2": 99, "triage_vital_temp": 36.7, "triage_pain": pain,
        })["suggested_class_idx"]
        assert ours == theirs == expected
