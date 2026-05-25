"""Layer 2 — synthesis with Qwen3.

Runs ONLY when a clinically meaningful conflict was detected between Layer 1A
(ML ensemble) and Layer 1B (MedGemma). Otherwise returns None to save latency.

Produces a clinician-facing summary explaining the disagreement, which
populates the `conflict.message` field shown in the frontend's demo.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.inference.ollama_client import OllamaClient, OllamaError
from app.models.schemas import MedGemmaAssessment, ModelPrediction, Vitals

log = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    summary: str
    recommended_category: int
    confidence: float


_SYNTHESIS_SYSTEM = """Jesteś nadrzędnym asystentem klinicznym SOR. Otrzymujesz:
- predykcje N modeli ML (kategoria MTS i pewność każdej),
- ocenę kliniczną MedGemma (kategoria + uzasadnienie + risk flags),
- parametry życiowe pacjenta.

Twoim zadaniem jest wyjaśnić rozbieżność i ZAREKOMENDOWAĆ ostateczną kategorię.

Zwróć WYŁĄCZNIE JSON:
- summary: krótkie polskie uzasadnienie dla pielęgniarki (max 350 znaków)
- recommended_category: 0-4 (zawsze wybieraj BARDZIEJ PILNĄ przy niepewności)
- confidence: 0.0-1.0
"""

_SYNTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary":              {"type": "string"},
        "recommended_category": {"type": "integer", "minimum": 0, "maximum": 4},
        "confidence":           {"type": "number",  "minimum": 0, "maximum": 1},
    },
    "required": ["summary", "recommended_category", "confidence"],
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
        f"- {p.model_name}: kategoria {p.category}, pewność {p.confidence:.2f}"
        for p in model_predictions
    )
    return f"""Pacjent ({vitals.age}l): SBP={vitals.sbp} HR={vitals.hr} SpO2={vitals.o2} Temp={vitals.temp:.1f}°C RR={vitals.rr}.

Predykcje ML:
{mp_lines}

Ocena MedGemma: kategoria {medgemma.category}, pewność {medgemma.confidence:.2f}
Uzasadnienie: {medgemma.reasoning}
Risk flags: {', '.join(medgemma.risk_flags) or '(brak)'}

Wyjaśnij rozbieżność i zarekomenduj ostateczną kategorię (jeśli niepewność — pilniejszą).
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
            f"Rozbieżność między modelami (zakres {min(cats)}–{max(cats)}). "
            "Rekomenduję pilniejszą kategorię ze względu na bezpieczeństwo."
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
            recommended_category=int(parsed.get("recommended_category", medgemma.category)),
            confidence=float(parsed.get("confidence", 0.7)),
        )
    except (OllamaError, Exception) as exc:
        log.warning("Synthesis (Qwen3) failed: %s — using deterministic fallback", exc)
        cats = [p.category for p in model_predictions] + [medgemma.category]
        return SynthesisResult(
            summary="Konflikt modeli — rekomenduję pilniejszą kategorię (fallback bez LLM).",
            recommended_category=min(cats),
            confidence=0.55,
        )
