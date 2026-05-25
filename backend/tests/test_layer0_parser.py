"""Tests for the Layer 0 keyword parser."""
from __future__ import annotations

import pytest

from app.pipeline.layer0_parser import KNOWN_CC_CODES, parse_with_keywords


class TestKeywordParser:
    def test_empty_note_returns_empty_cc(self):
        result = parse_with_keywords("")
        assert result["cc"] == {}
        assert result["extra"] == {}
        assert result["keywords"] == []
        assert result["source"] == "keyword"

    def test_whitespace_only_returns_empty(self):
        result = parse_with_keywords("   \n\t  ")
        assert result["cc"] == {}

    @pytest.mark.parametrize("note, expected_cc", [
        ("Pacjent ma ból w klatce piersiowej", ["cc_chestpain", "cc_chestpressure"]),
        ("chest pain radiating to jaw", ["cc_chestpain", "cc_chestpressure"]),
        ("duszność, brak tchu", ["cc_dyspnea", "cc_shortnessofbreath"]),
        ("udar mózgu, niedowład", ["cc_stroke", "cc_strokealert", "cc_weakness"]),
        ("drgawki uogólnione", ["cc_seizure"]),
        ("ból głowy migrenowy", ["cc_headache"]),
        ("ból brzucha + wymioty", ["cc_abdominalpain", "cc_vomiting", "cc_nausea"]),
        ("upadek z wysokości", ["cc_trauma", "cc_fall"]),
        ("złamanie nogi", ["cc_fracture"]),
        ("wysoka gorączka 40°C", ["cc_fever"]),
        ("próba samobójcza", ["cc_suicidalideation", "cc_suicidal"]),
        ("intoksykacja alkoholem", ["cc_intoxication", "cc_alcoholintoxication", "cc_overdose"]),
        ("reakcja alergiczna z obrzękiem", ["cc_allergicreaction", "cc_anaphylaxis", "cc_swelling"]),
    ])
    def test_keyword_matches(self, note: str, expected_cc: list[str]):
        result = parse_with_keywords(note)
        for cc in expected_cc:
            assert cc in result["cc"], f"Expected {cc} from '{note}' got {list(result['cc'])}"

    def test_pain_score_extracted(self):
        result = parse_with_keywords("ból 8/10, ból w klatce piersiowej")
        assert result["extra"].get("triage_pain") == 8.0
        assert "cc_chestpain" in result["cc"]

    def test_pain_score_clamped(self):
        result = parse_with_keywords("ból 15")
        # We never extract > 10
        if "triage_pain" in result["extra"]:
            assert result["extra"]["triage_pain"] <= 10.0

    def test_keywords_field_populated(self):
        result = parse_with_keywords("ból w klatce + duszność + spocony")
        assert len(result["keywords"]) >= 2

    def test_all_returned_cc_are_known(self):
        """Every cc_* code emitted must exist in the trained model vocabulary."""
        text = "ból klatka duszność udar gorączka samobójcza"
        result = parse_with_keywords(text)
        for cc in result["cc"]:
            code = cc.replace("cc_", "")
            assert code in KNOWN_CC_CODES, f"Unknown cc code emitted: {cc}"


class TestParserContract:
    """Schema-level guarantees the orchestrator depends on."""

    def test_output_has_required_keys(self):
        result = parse_with_keywords("test")
        for key in ("cc", "extra", "keywords", "source"):
            assert key in result

    def test_cc_values_are_int_1(self):
        result = parse_with_keywords("ból w klatce")
        assert all(v == 1 for v in result["cc"].values())

    def test_extra_values_are_numeric(self):
        result = parse_with_keywords("ból 7/10")
        for v in result["extra"].values():
            assert isinstance(v, (int, float))
