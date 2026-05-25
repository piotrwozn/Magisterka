"""Layer 0 — parse a clinical note into chief-complaint flags.

Two-mode implementation:
  1. Ollama / Llama 3 with structured JSON output (when Ollama is reachable)
  2. Deterministic keyword matcher (always available; used as fallback)

The keyword matcher covers ~30 common PL+EN triage signals; the LLM picks up
anything else (Polish abbreviations, paraphrases, embedded vitals etc.).
"""
from __future__ import annotations

import logging
import re

from app.inference.ollama_client import OllamaClient, OllamaError

log = logging.getLogger(__name__)


# ── Known cc_* codes the model was trained on (canonical lowercase) ──
# This is a subset of the 200+ chief-complaint columns in the parquet.
# The LLM is told to pick only from this list.
KNOWN_CC_CODES: list[str] = [
    "chestpain", "chestpressure", "palpitations",
    "dyspnea", "shortnessofbreath", "cough", "asthma", "respiratorydistress",
    "stroke", "strokealert", "weakness", "syncope", "alteredmentalstatus", "confusion",
    "seizure", "headache",
    "abdominalpain", "abdominalcramping", "vomiting", "nausea", "diarrhea", "gibleeding",
    "trauma", "fall", "fracture", "laceration", "bleeding", "burn",
    "fever", "sepsis",
    "suicidalideation", "suicidal", "intoxication", "alcoholintoxication", "overdose",
    "psychiatricproblem", "psychiatricevaluation", "anxiety", "agitation",
    "dizziness", "diaphoresis", "swelling",
    "rash", "allergicreaction", "anaphylaxis",
    "pregnancy",
    "backpain", "uti",
    "fulltrauma", "cardiacarrest", "motorvehiclecrash",
]


# Keyword → cc_* mapping (PL + EN). Patterns use leading \b but no trailing \b
# because Polish stems (drgawk-, gorączk-, samobójcz-) need to match across
# different word endings (drgawki, gorączki, samobójcza).
KEYWORD_MAP: dict[str, list[str]] = {
    r"\b(ból w klatce|chest pain|ucisk w klatce|zawał|heart attack)":
        ["cc_chestpain", "cc_chestpressure"],
    r"\b(palpitation|kołatanie|arytmi|tachycardia)":
        ["cc_palpitations"],
    r"\b(duszność|dyspnea|brak tchu|krótki oddech|shortness of breath)":
        ["cc_dyspnea", "cc_shortnessofbreath"],
    r"\b(kaszel|cough|krwioplucie|hemoptysis)":
        ["cc_cough"],
    r"\b(astma|asthma|wheezing|świszcz)":
        ["cc_asthma"],
    r"\b(udar|stroke|niedowład|paraliż|porażen)":
        ["cc_stroke", "cc_strokealert", "cc_weakness"],
    r"\b(utrata przytomności|unconscious|omdlenie|syncope|zasłab)":
        ["cc_syncope", "cc_alteredmentalstatus"],
    r"\b(splątani|confused|dezorientac|altered mental)":
        ["cc_alteredmentalstatus", "cc_confusion"],
    r"\b(drgawk|seizure|napad|konwulsj)":
        ["cc_seizure"],
    r"\b(ból głowy|headache|migrena|migraine)":
        ["cc_headache"],
    r"\b(ból brzucha|abdominal pain|brzucha)":
        ["cc_abdominalpain"],
    r"\b(wymioty|vomit|nausea|nudnoś)":
        ["cc_vomiting", "cc_nausea"],
    r"\b(biegunka|diarrhea)":
        ["cc_diarrhea"],
    r"\b(krwawienie z przewodu|gi bleed|krew w stolcu|hematemesis|smolisty stolec)":
        ["cc_gibleeding"],
    r"\b(uraz|trauma|wypadek|accident|upadek|fall)":
        ["cc_trauma", "cc_fall"],
    r"\b(złamanie|fracture|otwarte złamanie)":
        ["cc_fracture"],
    r"\b(rana|laceration|krwawienie|bleeding)":
        ["cc_laceration", "cc_bleeding"],
    r"\b(oparzenie|burn)":
        ["cc_burn"],
    r"\b(gorączk|fever|wysoka temperatura)":
        ["cc_fever"],
    r"\b(sepsa|sepsis|posocznica)":
        ["cc_sepsis"],
    r"\b(samobójcz|suicide|suicidal|próba s)":
        ["cc_suicidalideation", "cc_suicidal"],
    r"\b(intoksykacja|alkohol|alcohol|narkotyk|drug overdose|przedawkowani)":
        ["cc_intoxication", "cc_alcoholintoxication", "cc_overdose"],
    r"\b(psychotyczn|psychosis|halucynacj|hallucination)":
        ["cc_psychiatricproblem"],
    r"\b(zawroty głowy|dizziness|vertigo)":
        ["cc_dizziness"],
    r"\b(osłabieni|weakness|fatigue|zmęczeni)":
        ["cc_weakness"],
    r"\b(blad|pale|spocon|sweaty|diaphoretic)":
        ["cc_diaphoresis"],
    r"\b(obrzęk|swelling|edema)":
        ["cc_swelling"],
    r"\b(wysypka|rash|pokrzywka|hives)":
        ["cc_rash"],
    r"\b(reakcja alergiczna|allergic|anafilaksja|anaphylaxis)":
        ["cc_allergicreaction", "cc_anaphylaxis"],
    r"\b(ciąż|pregnancy|położnicz)":
        ["cc_pregnancy"],
    r"\b(ból plec|back pain)":
        ["cc_backpain"],
    r"\b(infekcja dróg moczowych|uti|urinary)":
        ["cc_uti"],
}


# Pain score: extract first 0-10 number near "ból" or "pain"
_PAIN_RE = re.compile(r"(?:ból|pain)\s+(\d{1,2})\s*/?\s*10?", re.IGNORECASE)


# ── LLM prompt + schema ───────────────────────────────────────────────

_PARSER_SYSTEM = """Jesteś precyzyjnym parserem notatek triażowych SOR.
Czytasz polski (lub angielski) tekst pielęgniarki i wyodrębniasz objawy zgodne ze schematem MTS.

Reguły:
- chief_complaints: lista kodów ANGIELSKICH z dostarczonej listy (np. "chestpain", "dyspnea")
- pain_score: 0-10 jeśli wspomniana, w przeciwnym razie null
- altered_mental_status: true/false
- urgency_signals: krótkie hasła (np. "hipotensja", "tachykardia")

Zwróć WYŁĄCZNIE poprawny JSON zgodny z schematem. Bez komentarzy.
"""

_PARSER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "chief_complaints": {
            "type": "array",
            "items": {"type": "string", "enum": KNOWN_CC_CODES},
        },
        "pain_score":           {"type": ["integer", "null"], "minimum": 0, "maximum": 10},
        "altered_mental_status":{"type": "boolean"},
        "urgency_signals":      {"type": "array", "items": {"type": "string"}},
    },
    "required": ["chief_complaints", "altered_mental_status"],
}


# ── Public API ────────────────────────────────────────────────────────

def parse_with_keywords(note: str) -> dict:
    """Deterministic fallback parser — no LLM required."""
    cc_features: dict[str, int] = {}
    keywords: list[str] = []
    extra: dict[str, float] = {}

    if note and note.strip():
        note_lc = note.lower()
        matched: set[str] = set()
        for pattern, cc_list in KEYWORD_MAP.items():
            if re.search(pattern, note_lc, re.IGNORECASE | re.UNICODE):
                matched.update(cc_list)
                keywords.append(pattern[:40])
        cc_features = {cc: 1 for cc in matched}

        pain_match = _PAIN_RE.search(note_lc)
        if pain_match:
            try:
                extra["triage_pain"] = float(min(10, int(pain_match.group(1))))
            except ValueError:
                pass

    log.debug("[keyword] parsed %d CC flags", len(cc_features))
    return {"cc": cc_features, "extra": extra, "keywords": keywords, "source": "keyword"}


async def parse_with_llm(
    note: str,
    client: OllamaClient,
    model: str,
) -> dict:
    """Use Ollama / Llama to parse the note. Raises OllamaError on failure."""
    if not note or not note.strip():
        return {"cc": {}, "extra": {}, "keywords": [], "source": "llm-empty"}

    parsed = await client.chat_json(
        model=model,
        system=_PARSER_SYSTEM,
        user=f"Notatka triażowa: {note}",
        schema=_PARSER_SCHEMA,
    )

    complaints = parsed.get("chief_complaints") or []
    if parsed.get("altered_mental_status") and "alteredmentalstatus" not in complaints:
        complaints.append("alteredmentalstatus")

    cc_features = {f"cc_{c.lower().replace(' ', '')}": 1 for c in complaints}

    extra: dict[str, float] = {}
    pain = parsed.get("pain_score")
    if pain is not None:
        try:
            extra["triage_pain"] = float(max(0, min(10, int(pain))))
        except (TypeError, ValueError):
            pass

    log.info("[llm] parsed %d CC flags via %s", len(cc_features), model)
    return {
        "cc": cc_features,
        "extra": extra,
        "keywords": parsed.get("urgency_signals", []),
        "source": "llm",
    }


async def parse_clinical_note_async(
    note: str,
    client: OllamaClient | None = None,
    model: str | None = None,
) -> dict:
    """Async entry point — uses Ollama if available, else falls back to keywords."""
    if client is not None and model:
        try:
            if await client.health():
                return await parse_with_llm(note, client, model)
        except OllamaError as exc:
            log.warning("LLM parser failed (%s) — falling back to keyword matcher", exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM parser unexpected error: %s — fallback to keyword", exc)
    return parse_with_keywords(note)


# Backwards-compatible sync API used by older callers
def parse_clinical_note(note: str) -> dict:
    return parse_with_keywords(note)
