"""Layer 2 — synthesis with Qwen3.

Runs ONLY when a clinically meaningful conflict was detected between Layer 1A
(ML ensemble) and Layer 1B (MedGemma). Otherwise returns None to save latency.

Produces a clinician-facing summary explaining the disagreement, which
populates the `conflict.message` field shown in the frontend's demo.
The LLM answers with an ESI level 1–5; `recommended_category` = ESI − 1 (index 0–4).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.inference.ollama_client import OllamaClient, OllamaError
from app.models.schemas import MedGemmaAssessment, ModelPrediction, Vitals
from app.pipeline.layer1b_nlp import ESI_LABELS_PL, esi_to_category

log = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    summary: str
    recommended_category: int
    confidence: float


_SYNTHESIS_SYSTEM = """Jesteś nadrzędnym asystentem klinicznym SOR. Otrzymujesz:
- predykcje N modeli ML (poziom ESI 1-5 i pewność każdej),
- ocenę kliniczną MedGemma (poziom ESI + uzasadnienie + risk flags),
- parametry życiowe pacjenta.

Poziomy Emergency Severity Index: ESI 1 = natychmiastowa interwencja ratująca życie,
ESI 2 = sytuacja wysokiego ryzyka / strefa zagrożenia parametrów, ESI 3-5 = pacjent
stabilny z ≥ 2 / 1 / 0 przewidywanymi zasobami. Kolory TOPSOR: ESI 1 czerwony,
ESI 2 pomarańczowy, ESI 3 żółty, ESI 4 zielony, ESI 5 niebieski.

Twoim zadaniem jest wyjaśnić rozbieżność i ZAREKOMENDOWAĆ ostateczny poziom ESI.

Zwróć WYŁĄCZNIE JSON:
- summary: krótkie polskie uzasadnienie dla pielęgniarki (max 350 znaków)
- recommended_esi: 1-5 (zawsze wybieraj BARDZIEJ PILNY poziom przy niepewności)
- confidence: 0.0-1.0
"""

_SYNTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary":              {"type": "string"},
        "recommended_esi": {"type": "integer", "minimum": 1, "maximum": 5},
        "confidence":      {"type": "number",  "minimum": 0, "maximum": 1},
    },
    "required": ["summary", "recommended_esi", "confidence"],
}


def should_run(
    model_predictions: list[ModelPrediction],
    medgemma: MedGemmaAssessment,
    threshold: int = 2,
) -> bool:
    """Synthesis triggers when models disagree ≥ threshold, OR ML vs MedGemma differ."""
    if not model_predictions:
        return False
    cats = [p.category for p in model_predictions]
    spread = max(cats) - min(cats)
    if spread >= threshold:
        return True
    ml_min = min(cats)
    if abs(ml_min - medgemma.category) >= threshold:
        return True
    return False


def _build_prompt(
    model_predictions: list[ModelPrediction],
    medgemma: MedGemmaAssessment,
    vitals: Vitals,
) -> str:
    mp_lines = "\n".join(
        f"- {p.model_name}: {ESI_LABELS_PL[p.category]}, pewność {p.confidence:.2f}"
        for p in model_predictions
    )
    return f"""Pacjent ({vitals.age}l): SBP={vitals.sbp} HR={vitals.hr} SpO2={vitals.o2} Temp={vitals.temp:.1f}°C RR={vitals.rr}.

Predykcje ML:
{mp_lines}

Ocena MedGemma: {ESI_LABELS_PL[medgemma.category]}, pewność {medgemma.confidence:.2f}
Uzasadnienie: {medgemma.reasoning}
Risk flags: {', '.join(medgemma.risk_flags) or '(brak)'}

Wyjaśnij rozbieżność i zarekomenduj ostateczny poziom ESI (jeśli niepewność — pilniejszy).
"""


async def synthesize(
    model_predictions: list[ModelPrediction],
    medgemma: MedGemmaAssessment,
    vitals: Vitals,
    client: OllamaClient | None,
    model: str | None,
) -> SynthesisResult | None:
    """Run Qwen3 only when conflict triggers. Returns None otherwise."""
    if not should_run(model_predictions, medgemma):
        return None

    if client is None or not model:
        # Deterministic fallback when LLM is offline
        cats = [p.category for p in model_predictions] + [medgemma.category]
        rec = min(cats)  # safer = more urgent
        summary = (
            f"Rozbieżność między modelami (ESI {min(cats) + 1}–{max(cats) + 1}). "
            "Rekomenduję pilniejszy poziom ESI ze względu na bezpieczeństwo."
        )
        return SynthesisResult(summary=summary, recommended_category=rec, confidence=0.6)

    try:
        if not await client.health():
            raise OllamaError("Ollama not healthy")
        parsed = await client.chat_json(
            model=model,
            system=_SYNTHESIS_SYSTEM,
            user=_build_prompt(model_predictions, medgemma, vitals),
            schema=_SYNTHESIS_SCHEMA,
        )
        return SynthesisResult(
            summary=str(parsed.get("summary", ""))[:500],
            recommended_category=esi_to_category(parsed.get("recommended_esi"), medgemma.category),
            confidence=float(parsed.get("confidence", 0.7)),
        )
    except (OllamaError, Exception) as exc:
        log.warning("Synthesis (Qwen3) failed: %s — using deterministic fallback", exc)
        cats = [p.category for p in model_predictions] + [medgemma.category]
        return SynthesisResult(
            summary="Konflikt modeli — rekomenduję pilniejszy poziom ESI (fallback bez LLM).",
            recommended_category=min(cats),
            confidence=0.55,
        )
