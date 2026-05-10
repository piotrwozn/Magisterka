"""Testy MTS rule engine."""

from __future__ import annotations

from src.explain.mts_rules import (
    MTS_VITAL_THRESHOLDS,
    check_consistency,
    explain_rule_decision,
    rule_based_triage,
)


class TestRuleBasedTriage:
    def test_normal_vitals_returns_green(self):
        vitals = {
            "triage_sbp": 130,
            "triage_o2sat": 99,
            "triage_pulse": 70,
            "triage_resp": 16,
            "triage_temp": 36.6,
            "triage_pain": 0,
        }
        result = rule_based_triage(vitals)
        # Dla całkowicie zdrowego pacjenta (pain=0) reguła Blue zadziała
        # bo pain < 2; logika startuje od Red.
        assert result["suggested_category"] in ["Green", "Blue"]

    def test_red_via_low_o2sat(self):
        vitals = {
            "triage_sbp": 110,
            "triage_o2sat": 80,  # < 85 → Red
            "triage_pulse": 100,
            "triage_resp": 20,
            "triage_temp": 37.0,
            "triage_pain": 5,
        }
        result = rule_based_triage(vitals)
        assert result["suggested_category"] == "Red"
        assert result["max_wait_minutes"] == 0
        assert len(result["triggered_rules"]) > 0

    def test_red_via_shock(self):
        vitals = {
            "triage_sbp": 80,  # < 90 → wstrząs
            "triage_o2sat": 95,
            "triage_pulse": 110,
            "triage_resp": 22,
            "triage_temp": 37.0,
            "triage_pain": 7,
        }
        result = rule_based_triage(vitals)
        assert result["suggested_category"] == "Red"

    def test_orange_via_o2sat(self):
        vitals = {
            "triage_sbp": 110,
            "triage_o2sat": 90,  # < 92 → Orange
            "triage_pulse": 95,
            "triage_resp": 24,
            "triage_temp": 38.0,
            "triage_pain": 6,
        }
        result = rule_based_triage(vitals)
        assert result["suggested_category"] == "Orange"
        assert result["max_wait_minutes"] == 10

    def test_yellow_via_pain(self):
        vitals = {
            "triage_sbp": 130,
            "triage_o2sat": 98,
            "triage_pulse": 80,
            "triage_resp": 16,
            "triage_temp": 36.6,
            "triage_pain": 6,  # 5-7 → Yellow
        }
        result = rule_based_triage(vitals)
        assert result["suggested_category"] == "Yellow"

    def test_handles_missing_vitals(self):
        """Brak vital signs = brak reguł = fallback Green."""
        vitals = {}
        result = rule_based_triage(vitals)
        assert result["suggested_category"] == "Green"
        assert result["triggered_rules"] == []

    def test_red_discriminator(self):
        vitals = {"triage_sbp": 130, "triage_o2sat": 99}
        result = rule_based_triage(vitals, discriminators={"unresponsive": True})
        assert result["suggested_category"] == "Red"


class TestConsistency:
    def test_agree(self):
        result = check_consistency(ml_prediction=2, rule_prediction=2)
        assert result["agree"] is True
        assert result["distance"] == 0

    def test_ml_more_urgent(self):
        result = check_consistency(ml_prediction=1, rule_prediction=3)
        assert result["agree"] is False
        assert result["safer_side"] == "ml"

    def test_potential_undertriage(self):
        """ML mówi Yellow, reguły Orange — niebezpieczne."""
        result = check_consistency(ml_prediction=2, rule_prediction=1)
        assert result["agree"] is False
        assert result["safer_side"] == "rule"


class TestExplainRule:
    def test_returns_string(self):
        result = rule_based_triage({"triage_sbp": 120, "triage_o2sat": 99})
        text = explain_rule_decision(result)
        assert isinstance(text, str)
        assert len(text) > 0


class TestThresholdsStructure:
    def test_all_categories_present(self):
        for cat in ["Red", "Orange", "Yellow", "Green", "Blue"]:
            assert cat in MTS_VITAL_THRESHOLDS

    def test_max_wait_increases_with_severity(self):
        """Max wait time powinien rosnąć Red→Blue."""
        order = ["Red", "Orange", "Yellow", "Green", "Blue"]
        prev_wait = -1
        for cat in order:
            wait = MTS_VITAL_THRESHOLDS[cat]["max_wait_minutes"]
            assert wait >= prev_wait, f"{cat} ma niespójny max_wait"
            prev_wait = wait
