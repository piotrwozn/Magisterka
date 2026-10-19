"""Testy regułowej weryfikacji parametrów życiowych (ESI, punkty A / B / D)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.explain.mts_rules import (
    MTS_VITAL_THRESHOLDS,
    check_consistency,
    explain_rule_decision,
    rule_based_triage,
)
from src.utils.config import CLASS_NAMES, TOPSOR_COLORS, TOPSOR_MAX_WAIT_MINUTES

NORMAL_VITALS = {
    "triage_vital_hr": 72,
    "triage_vital_sbp": 130,
    "triage_vital_dbp": 80,
    "triage_vital_rr": 16,
    "triage_vital_o2": 99,
    "triage_vital_temp": 36.6,
}


def _with(**changes) -> dict:
    return {**NORMAL_VITALS, **changes}


class TestRuleBasedTriage:
    def test_normal_vitals_do_not_force_level(self):
        """Parametry w normie → reguły nie wskazują ESI 1–2 (ESI 3–5 zależy od zasobów)."""
        result = rule_based_triage(NORMAL_VITALS)
        assert result["suggested_class_idx"] is None
        assert result["suggested_esi"] is None
        assert result["triggered_rules"] == []
        assert result["missing_vitals"] == []

    def test_esi1_via_critical_hypoxia(self):
        result = rule_based_triage(_with(triage_vital_o2=80))
        assert result["suggested_esi"] == 1
        assert result["suggested_class_idx"] == 0
        assert result["suggested_category"] == TOPSOR_COLORS[0]
        assert result["max_wait_minutes"] == 0
        assert result["decision_point"] == "A"
        assert len(result["triggered_rules"]) > 0

    def test_esi1_via_shock(self):
        result = rule_based_triage(_with(triage_vital_sbp=75, triage_vital_hr=125))
        assert result["suggested_esi"] == 1
        # Wszystkie spełnione reguły zachowane (również strefa zagrożenia HR > 100)
        assert {v["esi"] for v in result["all_violations"]} == {1, 2}

    def test_esi1_via_extreme_temperature_celsius(self):
        assert rule_based_triage(_with(triage_vital_temp=41.2))["suggested_esi"] == 1
        assert rule_based_triage(_with(triage_vital_temp=31.5))["suggested_esi"] == 1
        # 38,5 °C to gorączka, ale nie zagrożenie życia ani strefa zagrożenia
        assert rule_based_triage(_with(triage_vital_temp=38.5))["suggested_class_idx"] is None

    def test_fahrenheit_temperature_is_converted(self):
        """98,6 °F (= 37 °C) nie może zostać uznane za hipertermię."""
        result = rule_based_triage(_with(triage_vital_temp=98.6))
        assert result["suggested_class_idx"] is None

    def test_esi2_via_hypotension(self):
        result = rule_based_triage(_with(triage_vital_sbp=85))
        assert result["suggested_esi"] == 2
        assert result["decision_point"] == "B"
        assert result["advisory"] is False
        assert result["max_wait_minutes"] == TOPSOR_MAX_WAIT_MINUTES[TOPSOR_COLORS[1]]

    def test_danger_zone_hr(self):
        result = rule_based_triage(_with(triage_vital_hr=110))
        assert result["suggested_esi"] == 2
        assert result["decision_point"] == "D"
        assert result["advisory"] is True

    def test_danger_zone_rr_and_spo2(self):
        assert rule_based_triage(_with(triage_vital_rr=24))["suggested_esi"] == 2
        assert rule_based_triage(_with(triage_vital_o2=90))["suggested_esi"] == 2

    def test_danger_zone_boundaries(self):
        """Progi ostre: HR > 100, RR > 20, SpO2 < 92."""
        result = rule_based_triage(_with(triage_vital_hr=100, triage_vital_rr=20, triage_vital_o2=92))
        assert result["suggested_class_idx"] is None

    def test_legacy_aliases(self):
        legacy = {"triage_pulse": 120, "triage_sbp": 130, "triage_o2sat": 98, "triage_resp": 16}
        assert rule_based_triage(legacy)["suggested_esi"] == 2
        assert rule_based_triage({"triage_o2sat": 80})["suggested_esi"] == 1

    def test_pain_is_optional_high_risk(self):
        assert rule_based_triage(_with(triage_pain=8))["suggested_esi"] == 2
        assert rule_based_triage(_with(triage_pain=3))["suggested_class_idx"] is None

    def test_handles_missing_vitals(self):
        """Brak parametrów = brak reguł; informacja o brakach w notes."""
        result = rule_based_triage({})
        assert result["suggested_class_idx"] is None
        assert result["triggered_rules"] == []
        assert len(result["missing_vitals"]) == 5
        assert result["notes"]

    def test_nan_values_are_ignored(self):
        result = rule_based_triage(_with(triage_vital_o2=np.nan, triage_vital_hr=None))
        assert result["suggested_class_idx"] is None
        assert "triage_vital_o2" in result["missing_vitals"]
        assert "triage_vital_hr" in result["missing_vitals"]

    def test_explicit_discriminator(self):
        result = rule_based_triage(NORMAL_VITALS, discriminators={"unresponsive": True})
        assert result["suggested_esi"] == 1
        assert result["triggered_rules"][0]["type"] == "discriminator"

    def test_chief_complaint_discriminators(self):
        assert rule_based_triage(_with(cc_cardiacarrest=1))["suggested_esi"] == 1
        assert rule_based_triage(_with(cc_suicidal=1))["suggested_esi"] == 2
        assert rule_based_triage(_with(cc_cardiacarrest=0))["suggested_class_idx"] is None

    def test_accepts_series_and_frame(self):
        row = pd.Series(_with(triage_vital_o2=80))
        assert rule_based_triage(row)["suggested_esi"] == 1
        assert rule_based_triage(row.to_frame().T)["suggested_esi"] == 1


class TestConsistency:
    def test_agree(self):
        result = check_consistency(ml_prediction=1, rule_prediction=1)
        assert result["agree"] is True
        assert result["distance"] == 0
        assert result["undertriage"] is False

    def test_ml_more_urgent(self):
        result = check_consistency(ml_prediction=0, rule_prediction=1)
        assert result["agree"] is False
        assert result["safer_side"] == "ml"

    def test_potential_undertriage(self):
        """Model ESI 3, reguły ESI 2 — niebezpieczne."""
        result = check_consistency(ml_prediction=2, rule_prediction=1)
        assert result["agree"] is False
        assert result["safer_side"] == "rule"
        assert result["undertriage"] is True
        assert result["ml_class"] == CLASS_NAMES[2]
        assert result["rule_class"] == CLASS_NAMES[1]

    def test_no_rule_level_accepts_esi_3_to_5(self):
        for ml in (2, 3, 4):
            result = check_consistency(ml_prediction=ml, rule_prediction=None)
            assert result["agree"] is True
            assert result["undertriage"] is False

    def test_no_rule_level_ml_high_acuity_is_safe(self):
        result = check_consistency(ml_prediction=1, rule_prediction=None)
        assert result["safer_side"] == "ml"
        assert result["undertriage"] is False


class TestExplainRule:
    def test_returns_string_without_rules(self):
        text = explain_rule_decision(rule_based_triage(NORMAL_VITALS))
        assert isinstance(text, str)
        assert "ESI 3–5" in text

    def test_returns_string_with_rules(self):
        text = explain_rule_decision(rule_based_triage(_with(triage_vital_o2=80)))
        assert "ESI 1" in text
        assert "SpO2" in text
        assert "MTS" not in text


class TestThresholdsStructure:
    def test_all_esi_levels_present(self):
        assert list(MTS_VITAL_THRESHOLDS) == CLASS_NAMES
        for idx, name in enumerate(CLASS_NAMES):
            level = MTS_VITAL_THRESHOLDS[name]
            assert level["esi"] == idx + 1
            assert level["class_idx"] == idx
            assert level["color"] == TOPSOR_COLORS[idx]

    def test_max_wait_increases_with_level(self):
        """Maks. czas oczekiwania TOPSOR rośnie od ESI 1 do ESI 5."""
        waits = [MTS_VITAL_THRESHOLDS[name]["max_wait_minutes"] for name in CLASS_NAMES]
        assert waits == sorted(waits)
        assert waits[0] == 0

    def test_only_esi1_and_esi2_have_vital_rules(self):
        """ESI 3–5 zależą od zasobów — brak reguł progowych."""
        for idx, name in enumerate(CLASS_NAMES):
            has_rules = bool(MTS_VITAL_THRESHOLDS[name]["vital_rules"])
            assert has_rules == (idx < 2)
