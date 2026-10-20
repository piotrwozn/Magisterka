"""Tests for the Layer 0 parser (keyword matcher + LLM output filtering)."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from app.pipeline import feature_engineering  # noqa: F401  (puts training/ on sys.path)
from app.pipeline.layer0_parser import (
    KEYWORD_MAP,
    KNOWN_CC_CODES,
    LEGACY_CC_CODES,
    REDUNDANT_GENERIC,
    drop_redundant_generic,
    normalize_cc_code,
    parse_with_keywords,
    parse_with_llm,
)
from src.data import yale_schema as ys  # noqa: E402
from src.data.prepare import plan_features  # noqa: E402
from src.utils.config import DATA_VERSION  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "training" / "data" / "processed" / "esi"
FEATURE_SETS_JSON = PROCESSED_DIR / "feature_sets.json"
TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
# Columns training's prepare always reads besides the complaints (names only — no data)
_RAW_BASE_COLUMNS = (
    ["esi", "dep_name", "disposition", "age", "gender", "arrivalmode", "arrivalmonth",
     "arrivalday", "arrivalhour_bin", "previousdispo"]
    + ys.TRIAGE_VITAL_COLS + ys.ED_USAGE_COLS + ys.IMAGING_COLS
)

# Notes exercising most of the keyword map (PL + EN)
SAMPLE_NOTES = [
    "Pacjent blady, spocony, ból w klatce piersiowej, duszność",
    "ucisk w klatce, kołatanie serca, migotanie przedsionków, tachykardia",
    "NZK, RKO w karetce, politrauma, aktywacja zespołu urazowego, pacjent nieprzytomny",
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
    """Chief-complaint columns of the trained models (skip if data missing or stale)."""
    if not FEATURE_SETS_JSON.exists():
        pytest.skip("No training/data/processed/esi/feature_sets.json — run 02_preprocess.py")
    payload = json.loads(FEATURE_SETS_JSON.read_text(encoding="utf-8"))
    version = payload.get("data_version")
    if version != DATA_VERSION:
        pytest.skip(f"feature_sets.json is from data_version {version!r}, the code expects "
                    f"{DATA_VERSION!r} — re-run training/scripts/02_preprocess.py")
    return set(payload["groups"]["chief_complaints"])


@pytest.fixture(scope="module")
def train_cc_flags(trained_cc_columns):
    """Boolean frame of the cc_* columns of the real train split (skip if absent)."""
    if not TRAIN_PARQUET.exists():
        pytest.skip("No training/data/processed/esi/train.parquet — run 02_preprocess.py")
    pd = pytest.importorskip("pandas")
    return pd.read_parquet(TRAIN_PARQUET, columns=sorted(trained_cc_columns)) > 0


def _co_occurrence(flags, a: str, b: str) -> float:
    """Share of the visits with the rarer of the two complaints that also have the other one."""
    both = int((flags[a] & flags[b]).sum())
    return both / max(1, min(int(flags[a].sum()), int(flags[b].sum())))


# Yale synonyms: the keyword rules emit only the code that is more frequent in train
SYNONYMS = [("cc_dyspnea", "cc_shortnessofbreath"), ("cc_confusion", "cc_alteredmentalstatus")]

# Short notes describing ONE clinical problem each (several rules may match them)
SINGLE_PROBLEM_NOTES = [
    "duszność, brak tchu", "shortness of breath", "splątany, zdezorientowany",
    "złamanie nogi", "uraz głowy", "upadek, uraz głowy", "uraz kolana po upadku",
    "wielokrotne upadki", "wypadek komunikacyjny", "motocyklista po wypadku",
    "politrauma po wypadku", "pobity, uraz twarzy", "skręcenie kostki", "złamanie żeber",
    "obrzęk nóg", "obrzęk twarzy", "reakcja alergiczna z obrzękiem twarzy",
    "gorączka, chemioterapia", "ciąża, ból brzucha", "ból głowy migrenowy",
    "najgorszy ból głowy w życiu", "krwawienie z nosa", "krwawienie z rany",
    "zatrucie lekami", "przedawkowanie leków, próba samobójcza", "drgawki",
]


@pytest.fixture(scope="module")
def prepared_cc_columns() -> set[str]:
    """Complaint columns training's prepare builds from the raw Yale vocabulary
    (the model vocabulary before the trauma merge) — computed from names only."""
    raw_cc = [f"cc_{c}" for c in KNOWN_CC_CODES if c != "trauma_activation"]
    raw_cc += list(ys.TRAUMA_ACTIVATION_SOURCES)
    return set(plan_features(_RAW_BASE_COLUMNS + raw_cc).groups["chief_complaints"])


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
        ("duszność, brak tchu", ["cc_shortnessofbreath"]),
        ("udar mózgu, niedowład", ["cc_strokealert", "cc_extremityweakness"]),
        ("drgawki uogólnione", ["cc_seizures"]),
        ("ból głowy migrenowy", ["cc_migraine"]),
        ("ból brzucha + wymioty", ["cc_abdominalpain", "cc_emesis"]),
        ("nudności", ["cc_nausea"]),
        ("upadek z wysokości", ["cc_fall"]),
        ("złamanie nogi", ["cc_leginjury"]),
        ("złamanie, uraz", ["cc_trauma"]),
        ("wysoka gorączka 40°C", ["cc_fever"]),
        ("podejrzenie sepsy", ["cc_fever"]),
        ("próba samobójcza", ["cc_suicidal"]),
        ("intoksykacja alkoholem", ["cc_alcoholintoxication"]),
        ("reakcja alergiczna z obrzękiem", ["cc_allergicreaction", "cc_edema"]),
        ("krwawienie z rany", ["cc_bleeding_bruising", "cc_laceration"]),
        ("halucynacje, psychoza", ["cc_hallucinations", "cc_psychoticsymptoms"]),
        ("infekcja dróg moczowych", ["cc_urinarytractinfection"]),
        ("NZK, RKO", ["cc_cardiacarrest"]),
        ("politrauma po wypadku", ["cc_trauma_activation"]),
        ("powiadomienie ZRM: aktywacja zespołu urazowego", ["cc_trauma_activation"]),
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
        assert "cc_abdominalpain" not in result["cc"]           # generic dropped

    @pytest.mark.parametrize("note, expected", [
        ("duszność, brak tchu", {"cc_shortnessofbreath"}),       # not + cc_dyspnea
        ("pacjent splątany", {"cc_alteredmentalstatus"}),        # not + cc_confusion
        ("złamanie nogi", {"cc_leginjury"}),                     # not + cc_trauma
        ("wypadek komunikacyjny", {"cc_motorvehiclecrash"}),
        ("wielokrotne upadki", {"cc_multiplefalls"}),            # not + cc_fall
        ("obrzęk nóg", {"cc_legswelling"}),                      # not + cc_edema
        ("ból głowy migrenowy", {"cc_migraine"}),                # not + cc_headache
        ("upadek, uraz głowy", {"cc_fall", "cc_headinjury"}),    # both real complaints
    ])
    def test_one_code_per_problem(self, note: str, expected: set[str]):
        assert set(parse_with_keywords(note)["cc"]) == expected

    def test_generic_trauma_kept_without_specific_code(self):
        assert set(parse_with_keywords("uraz w pracy")["cc"]) == {"cc_trauma"}

    def test_overdose_intent(self):
        intentional = parse_with_keywords("próba samobójcza, przedawkowanie leków")["cc"]
        accidental = parse_with_keywords("przypadkowe przedawkowanie leków")["cc"]
        assert "cc_overdose_intentional" in intentional
        assert "cc_overdose_accidental" not in intentional
        assert "cc_overdose_accidental" in accidental
        assert "cc_overdose_intentional" not in accidental

    def test_fever_in_immunocompromised(self):
        result = parse_with_keywords("gorączka 39, w trakcie chemioterapii")
        assert set(result["cc"]) == {"cc_feverimmunocompromised"}       # generic fever dropped

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

    def test_trauma_activation_replaces_yale_protocol_codes(self):
        assert len(KNOWN_CC_CODES) == 199
        assert "trauma_activation" in KNOWN_CC_CODES
        assert not {"fulltrauma", "modifiedtrauma"} & set(KNOWN_CC_CODES)
        assert set(LEGACY_CC_CODES.values()) <= set(KNOWN_CC_CODES)

    def test_known_codes_match_prepare(self, prepared_cc_columns):
        assert {f"cc_{c}" for c in KNOWN_CC_CODES} == prepared_cc_columns
        assert "cc_trauma_activation" == ys.TRAUMA_ACTIVATION_COL

    def test_keyword_map_targets_exist_in_prepare_output(self, prepared_cc_columns):
        targets = set(KEYWORD_MAP.values())
        assert targets - prepared_cc_columns == set()

    def test_each_rule_emits_one_code(self):
        assert all(isinstance(cc, str) and cc[3:] in KNOWN_CC_CODES for cc in KEYWORD_MAP.values())

    def test_known_codes_match_trained_columns(self, trained_cc_columns):
        assert {f"cc_{c}" for c in KNOWN_CC_CODES} == trained_cc_columns

    def test_keyword_map_targets_exist(self, trained_cc_columns):
        targets = set(KEYWORD_MAP.values())
        assert targets - trained_cc_columns == set()

    def test_redundancy_table_uses_real_codes(self, trained_cc_columns):
        codes = set(REDUNDANT_GENERIC) | {c for spec in REDUNDANT_GENERIC.values() for c in spec}
        assert codes - trained_cc_columns == set()

    def test_every_parsed_code_exists_in_feature_list(self, trained_cc_columns):
        emitted: set[str] = set()
        for note in SAMPLE_NOTES:
            emitted |= set(parse_with_keywords(note)["cc"])
        assert len(emitted) > 50                       # the notes cover most of the map
        assert emitted - trained_cc_columns == set()


class TestTrainCoOccurrence:
    """Code combinations the parser emits must exist in the training data
    (computed from the real train.parquet; skipped when it is absent)."""

    @pytest.mark.parametrize("note", SINGLE_PROBLEM_NOTES)
    def test_no_rare_pair_emitted(self, train_cc_flags, note):
        codes = sorted(parse_with_keywords(note)["cc"])
        rare = [(a, b, round(_co_occurrence(train_cc_flags, a, b), 4))
                for a, b in itertools.combinations(codes, 2)
                if _co_occurrence(train_cc_flags, a, b) < 0.01]
        assert rare == [], f"{note!r} → {codes}: pairs rare in train {rare}"

    def test_synonym_rules_emit_most_frequent_code(self, train_cc_flags):
        for rare, frequent in SYNONYMS:
            assert train_cc_flags[frequent].sum() > train_cc_flags[rare].sum()
            assert rare in REDUNDANT_GENERIC and frequent in REDUNDANT_GENERIC[rare]
            assert rare not in KEYWORD_MAP.values()

    def test_dropped_generic_codes_rarely_co_occur(self, train_cc_flags):
        synonyms = {rare for rare, _ in SYNONYMS}
        for generic, specific in REDUNDANT_GENERIC.items():
            if generic in synonyms:
                continue
            for code in specific:
                rate = _co_occurrence(train_cc_flags, generic, code)
                assert rate < 0.01, f"{generic} + {code} co-occur in {rate:.1%} — not redundant"


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
        # Yale's trauma activation codes → merged training column
        assert normalize_cc_code("fulltrauma") == "cc_trauma_activation"
        assert normalize_cc_code("cc_modifiedtrauma") == "cc_trauma_activation"

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

    async def test_llm_redundant_codes_dropped(self):
        client = _FakeOllama({
            "chief_complaints": ["dyspnea", "shortnessofbreath", "trauma", "leginjury", "confusion"],
            "altered_mental_status": True,
        })
        result = await parse_with_llm("notatka", client, "llama3")
        assert set(result["cc"]) == {"cc_shortnessofbreath", "cc_leginjury", "cc_alteredmentalstatus"}

    def test_drop_redundant_generic(self):
        assert drop_redundant_generic({"cc_trauma", "cc_fall"}) == {"cc_fall"}
        assert drop_redundant_generic({"cc_trauma"}) == {"cc_trauma"}
        assert drop_redundant_generic(["cc_fever", "cc_fever_75yearsorolder"]) == {"cc_fever_75yearsorolder"}


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
