"""Layer 1B — clinical NLP via MedGemma 27B (or deterministic stub fallback)."""
from __future__ import annotations

import logging
from typing import Any

from app.inference.ollama_client import OllamaClient, OllamaError
from app.models.schemas import MedGemmaAssessment, Vitals

log = logging.getLogger(__name__)


_MTS_LABELS_PL = ["Red (Natychmiastowy)", "Orange (Pilny)", "Yellow (Mniej pilny)",
                  "Green (Niepilny)", "Blue (Nieostry)"]

# ── Prompts / schema ──────────────────────────────────────────────────

_MEDGEMMA_SYSTEM = """Jesteś asystentem klinicznym w SOR. Otrzymujesz parametry życiowe
pacjenta + notatkę pielęgniarki oraz sugestię modelu ML. Oceniasz pilność według
Manchester Triage System (0=Red, 1=Orange, 2=Yellow, 3=Green, 4=Blue).

Reguły:
- "category": liczba 0-4
- "confidence": liczba 0.0-1.0
- "reasoning": krótkie uzasadnienie po polsku (max 350 znaków)
- "risk_flags": lista krótkich etykiet (np. "hipotensja", "tachykardia")
- "key_findings": lista konkretnych wartości (np. "SBP=95", "HR=118")

Zwróć WYŁĄCZNIE poprawny JSON. Bez komentarzy.
"""

_MEDGEMMA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category":     {"type": "integer", "minimum": 0, "maximum": 4},
        "confidence":   {"type": "number",  "minimum": 0, "maximum": 1},
        "reasoning":    {"type": "string"},
        "risk_flags":   {"type": "array", "items": {"type": "string"}},
        "key_findings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["category", "confidence", "reasoning", "risk_flags", "key_findings"],
}


def _build_prompt(vitals: Vitals, note: str, ml_category: int) -> str:
    return f"""Parametry życiowe pacjenta:
- Wiek: {vitals.age} lat
- Temperatura: {vitals.temp:.1f}°C
- HR: {vitals.hr} bpm | SBP/DBP: {vitals.sbp}/{vitals.dbp} mmHg
- SpO2: {vitals.o2}% | RR: {vitals.rr}/min

Notatka pielęgniarska: {note or "(brak)"}

Sugestia ML: kategoria {ml_category} ({_MTS_LABELS_PL[ml_category]})

Oceń:
1. Czy zgadzasz się z ML?
2. Jakie czynniki ryzyka widzisz?
3. Jakie kluczowe wartości to potwierdzają?
"""


# ── Deterministic fallback stub ───────────────────────────────────────

def _collect_risk_flags(vitals: Vitals) -> list[str]:
    flags = []
    if vitals.sbp < 90: flags.append("hipotensja")
    elif vitals.sbp < 100: flags.append("graniczne niskie SBP")
    if vitals.sbp > 180: flags.append("nadciśnienie")
    if vitals.hr > 130: flags.append("ciężka tachykardia")
    elif vitals.hr > 110: flags.append("tachykardia")
    if vitals.hr < 50: flags.append("bradykardia")
    if vitals.o2 < 90: flags.append("ciężka hipoksja")
    elif vitals.o2 < 94: flags.append("hipoksja")
    if vitals.temp >= 39: flags.append("wysoka gorączka")
    elif vitals.temp < 35: flags.append("hipotermia")
    if vitals.rr > 28: flags.append("tachypnoe")
    if vitals.age >= 75: flags.append("wiek podeszły")
    return flags


def _key_findings(vitals: Vitals) -> list[str]:
    findings = []
    if vitals.sbp < 100: findings.append(f"SBP={vitals.sbp} mmHg")
    if vitals.hr > 110 or vitals.hr < 50: findings.append(f"HR={vitals.hr} bpm")
    if vitals.o2 < 94: findings.append(f"SpO₂={vitals.o2}%")
    if vitals.temp >= 38 or vitals.temp < 36: findings.append(f"Temp={vitals.temp}°C")
    if vitals.rr > 22: findings.append(f"RR={vitals.rr}/min")
    if not findings: findings.append(f"Parametry stabilne (wiek={vitals.age})")
    return findings


_REASONING_BY_CATEGORY = {
    0: "Stan zagrożenia życia — wymagana natychmiastowa interwencja resuscytacyjna.",
    1: "Bardzo wysoki priorytet. Krytyczne parametry wymagają oceny w ciągu 10 minut.",
    2: "Stabilne parametry z odchyleniami. Wymagana ocena lekarska w ciągu 60 minut.",
    3: "Stan nieostry, niski priorytet kliniczny. Możliwa obserwacja.",
    4: "Stan stabilny — przypadek nieostry, tryb ambulatoryjny.",
}


def stub_assessment(ml_category: int, vitals: Vitals) -> MedGemmaAssessment:
    flags = _collect_risk_flags(vitals)
    findings = _key_findings(vitals)
    confidence = 0.85
    high_risk = sum(1 for f in flags if any(k in f for k in ("hipoksja", "hipotensja", "bradykardia", "tachykardia")))
    if ml_category <= 1 and high_risk >= 2:
        confidence = min(0.95, confidence + 0.05 * high_risk)
    elif ml_category >= 3 and high_risk >= 2:
        confidence -= 0.15
    confidence = max(0.5, min(0.99, confidence))

    return MedGemmaAssessment(
        category=ml_category,
        confidence=round(confidence, 3),
        reasoning=_REASONING_BY_CATEGORY.get(ml_category, "Brak oceny."),
        risk_flags=flags,
        key_findings=findings,
    )


# ── Public API ────────────────────────────────────────────────────────

async def assess_async(
    ml_category: int,
    vitals: Vitals,
    note: str,
    client: OllamaClient | None = None,
    model: str | None = None,
) -> MedGemmaAssessment:
    """Run MedGemma; fall back to a deterministic clinical heuristic on any error."""
    if client is not None and model:
        try:
            if await client.health():
                parsed = await client.chat_json(
                    model=model,
                    system=_MEDGEMMA_SYSTEM,
                    user=_build_prompt(vitals, note, ml_category),
                    schema=_MEDGEMMA_SCHEMA,
                )
                return MedGemmaAssessment(
                    category=int(parsed.get("category", ml_category)),
                    confidence=float(parsed.get("confidence", 0.8)),
                    reasoning=str(parsed.get("reasoning", "")) or _REASONING_BY_CATEGORY.get(ml_category, ""),
                    risk_flags=list(parsed.get("risk_flags", [])) or _collect_risk_flags(vitals),
                    key_findings=list(parsed.get("key_findings", [])) or _key_findings(vitals),
                )
        except OllamaError as exc:
            log.warning("MedGemma call failed (%s) — using stub", exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("MedGemma unexpected error: %s — using stub", exc)
    return stub_assessment(ml_category, vitals)


# Backwards-compatible sync API
def assess(ml_category: int, vitals: Vitals, note: str = "", parsed_cc: dict | None = None) -> MedGemmaAssessment:
    return stub_assessment(ml_category, vitals)
