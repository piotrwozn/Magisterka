"""Tests for the Layer 0 parser (keyword matcher + LLM output filtering)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pipeline.layer0_parser import (
    KEYWORD_MAP,
    KNOWN_CC_CODES,
    normalize_cc_code,
    parse_with_keywords,
    parse_with_llm,
)

FEATURE_SETS_JSON = (
    Path(__file__).resolve().parents[2] / "training" / "data" / "processed" / "esi" / "feature_sets.json"
)

# Notes exercising most of the keyword map (PL + EN)
SAMPLE_NOTES = [
    "Pacjent blady, spocony, ból w klatce piersiowej, duszność",
    "ucisk w klatce, kołatanie serca, migotanie przedsionków, tachykardia",
    "NZK, RKO w karetce, politrauma, pacjent nieprzytomny",
    "udar mózgu, niedowład, splątany, drgawki, omdlenie, przedomdlenie",
    "ból głowy migrenowy, najgorszy ból głowy w życiu, zawroty głowy, drętwienie",
    "ból brzucha + wymioty + nudności, biegunka, zaparcie, zgaga, kolka nerkowa",
    "krwawienie z przewodu pokarmowego, krwawienie z odbytu, krwawienie z nosa",
    "upadek, złamanie nogi, uraz kolana, uraz głowy, wypadek komunikacyjny, motocykl",
    "pobity, uraz twarzy, uraz oka, złamanie żeber, skręcenie kostki, uraz barku",
    "rana cięta dłoni, oparzenie, pogryzienie przez psa, użądlenie, kleszcz",
    "wysoka gorączka 40°C, sepsa, chemioterapia, dreszcze, odwodnienie, grypa",
    "próba samobójcza, przedawkowanie leków, halucynacje, psychoza, pobudzony",
    "intoksykacja alkoholem, zespół abstynencyjny, narkotyki, zatrucie czadem",
    "reakcja alergiczna z obrzękiem twarzy, wysypka, obrzęk nóg, obrzęk języka",
    "ciąża 20 tydz., ból brzucha, krwawienie z dróg rodnych",
    "hipoglikemia, infekcja dróg moczowych, krwiomocz, zatrzymanie moczu",
    "ból pleców, ból karku, ból gardła, ból ucha, ból zęba, ropień",
    "zdjęcie szwów, kontrola rany, lęk, atak paniki, depresja, myśli samobójcze",
    "chest pain, shortness of breath, seizure, stroke, fever, overdose, rash",
]


@pytest.fixture(scope="module")
def trained_cc_columns() -> set[str]:
    """Chief-complaint columns of the trained models (skip if data not prepared)."""
    if not FEATURE_SETS_JSON.exists():
        pytest.skip("No training/data/processed/esi/feature_sets.json — run 02_preprocess.py")
    return set(json.loads(FEATURE_SETS_JSON.read_text())["groups"]["chief_complaints"])


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
        ("Pacjent ma ból w klatce piersiowej", ["cc_chestpain"]),
        ("chest pain radiating to jaw", ["cc_chestpain"]),
        ("ucisk w klatce piersiowej", ["cc_chesttightness"]),
        ("duszność, brak tchu", ["cc_dyspnea", "cc_shortnessofbreath"]),
        ("udar mózgu, niedowład", ["cc_strokealert", "cc_extremityweakness"]),
        ("drgawki uogólnione", ["cc_seizures"]),
        ("ból głowy migrenowy", ["cc_headache", "cc_migraine"]),
        ("ból brzucha + wymioty", ["cc_abdominalpain", "cc_emesis"]),
        ("nudności", ["cc_nausea"]),
        ("upadek z wysokości", ["cc_fall"]),
        ("złamanie nogi", ["cc_trauma", "cc_leginjury"]),
        ("wysoka gorączka 40°C", ["cc_fever"]),
        ("podejrzenie sepsy", ["cc_fever"]),
        ("próba samobójcza", ["cc_suicidal"]),
        ("intoksykacja alkoholem", ["cc_alcoholintoxication"]),
        ("reakcja alergiczna z obrzękiem", ["cc_allergicreaction", "cc_edema"]),
        ("krwawienie z rany", ["cc_bleeding_bruising", "cc_laceration"]),
        ("halucynacje, psychoza", ["cc_hallucinations", "cc_psychoticsymptoms"]),
        ("infekcja dróg moczowych", ["cc_urinarytractinfection"]),
        ("NZK, RKO", ["cc_cardiacarrest"]),
    ])
    def test_keyword_matches(self, note: str, expected_cc: list[str]):
        result = parse_with_keywords(note)
        for cc in expected_cc:
            assert cc in result["cc"], f"Expected {cc} from '{note}' got {list(result['cc'])}"

    @pytest.mark.parametrize("note, absent", [
        ("Ból brzucha od rana", "cc_laceration"),           # "od rana" = since the morning
        ("krwawienie z nosa", "cc_bleeding_bruising"),      # specific → cc_epistaxis only
        ("krwawienie z przewodu pokarmowego", "cc_bleeding_bruising"),
        ("ból brzucha", "cc_abdominalpainpregnant"),        # no pregnancy mentioned
        ("Wizyta kontrolna", "cc_woundcheck"),
    ])
    def test_no_false_positive(self, note: str, absent: str):
        assert absent not in parse_with_keywords(note)["cc"]

    def test_pregnancy_with_abdominal_pain(self):
        result = parse_with_keywords("Ciąża 20 tydz. Ból brzucha od rana.")
        assert "cc_abdominalpainpregnant" in result["cc"]

    def test_overdose_intent(self):
        intentional = parse_with_keywords("próba samobójcza, przedawkowanie leków")["cc"]
        accidental = parse_with_keywords("przypadkowe przedawkowanie leków")["cc"]
        assert "cc_overdose_intentional" in intentional
        assert "cc_overdose_accidental" not in intentional
        assert "cc_overdose_accidental" in accidental
        assert "cc_overdose_intentional" not in accidental

    def test_fever_in_immunocompromised(self):
        result = parse_with_keywords("gorączka 39, w trakcie chemioterapii")
        assert {"cc_fever", "cc_feverimmunocompromised"} <= set(result["cc"])

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
        assert all(isinstance(k, str) and k for k in result["keywords"])

    def test_all_returned_cc_are_known(self):
        """Every cc_* code emitted must be in the parser vocabulary."""
        for note in SAMPLE_NOTES:
            for cc in parse_with_keywords(note)["cc"]:
                assert cc.startswith("cc_")
                assert cc[3:] in KNOWN_CC_CODES, f"Unknown cc code emitted: {cc}"


class TestTrainedVocabulary:
    """The parser must only target real chief-complaint columns of the trained models."""

    def test_known_codes_unique(self):
        assert len(KNOWN_CC_CODES) == len(set(KNOWN_CC_CODES))

    def test_known_codes_match_trained_columns(self, trained_cc_columns):
        assert {f"cc_{c}" for c in KNOWN_CC_CODES} == trained_cc_columns

    def test_keyword_map_targets_exist(self, trained_cc_columns):
        targets = {cc for cc_list in KEYWORD_MAP.values() for cc in cc_list}
        assert targets - trained_cc_columns == set()

    def test_every_parsed_code_exists_in_feature_list(self, trained_cc_columns):
        emitted: set[str] = set()
        for note in SAMPLE_NOTES:
            emitted |= set(parse_with_keywords(note)["cc"])
        assert len(emitted) > 50                       # the notes cover most of the map
        assert emitted - trained_cc_columns == set()


class _FakeOllama:
    def __init__(self, payload: dict):
        self.payload = payload

    async def chat_json(self, **_kwargs) -> dict:
        return self.payload


class TestLlmParser:
    def test_normalize_cc_code(self):
        assert normalize_cc_code("chestpain") == "cc_chestpain"
        assert normalize_cc_code("cc_chestpain") == "cc_chestpain"
        assert normalize_cc_code("Chest Pain") == "cc_chestpain"
        assert normalize_cc_code("seizure") is None          # real column: cc_seizures
        assert normalize_cc_code("pregnancy") is None

    async def test_llm_output_filtered_to_trained_codes(self):
        client = _FakeOllama({
            "chief_complaints": ["chestpain", "cc_dyspnea", "seizure", "sepsis"],
            "altered_mental_status": True,
            "pain_score": 7,
            "urgency_signals": ["tachykardia"],
        })
        result = await parse_with_llm("notatka", client, "llama3")
        assert set(result["cc"]) == {"cc_chestpain", "cc_dyspnea", "cc_alteredmentalstatus"}
        assert result["extra"] == {"triage_pain": 7.0}
        assert result["source"] == "llm"


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
