"""Main orchestrator — ties all layers together for a single /predict call."""
from __future__ import annotations

import logging
import time

from app.config import get_settings
from app.inference.ollama_client import OllamaClient, get_ollama_client
from app.inference.shap_explainer import top_n_shap
from app.models.registry import ModelRegistry
from app.models.schemas import (
    ConflictInfo,
    PredictRequest,
    PredictResponse,
    ShapValue,
)
from app.pipeline import layer0_parser, layer1a_ml, layer1b_nlp, layer2_synthesis
from app.pipeline.feature_engineering import align_features, build_feature_row
from app.pipeline.safety_rules import apply_safety_rules

log = logging.getLogger(__name__)


def _build_ollama() -> OllamaClient | None:
    s = get_settings()
    if not s.ollama_enabled:
        return None
    return get_ollama_client(s.ollama_base_url, s.ollama_timeout_s)


async def predict(request: PredictRequest, registry: ModelRegistry) -> PredictResponse:
    """End-to-end inference pipeline (async)."""
    settings = get_settings()
    t_start = time.perf_counter()

    ollama = _build_ollama()

    # ── Layer 0: parse clinical note (LLM if available, else keywords) ─
    parsed = await layer0_parser.parse_clinical_note_async(
        request.clinical_note,
        client=ollama,
        model=settings.ollama_parser_model,
    )
    cc_features: dict[str, int] = parsed["cc"]
    extra: dict = parsed["extra"]

    # ── Build feature row ───────────────────────────────────────────
    all_cc = [k for k in registry.medians.keys() if k.startswith("cc_")]
    df_row = build_feature_row(request.vitals, cc_features, extra, all_cc_columns=all_cc)

    # ── Layer 1A: ensemble ML inference ─────────────────────────────
    final_category, final_confidence, model_predictions = layer1a_ml.run_ensemble(df_row, registry)

    # ── Layer 1B: MedGemma assessment ───────────────────────────────
    medgemma = await layer1b_nlp.assess_async(
        ml_category=final_category,
        vitals=request.vitals,
        note=request.clinical_note,
        client=ollama,
        model=settings.ollama_nlp_model,
    )

    # ── Layer 2: Qwen3 synthesis (conditional on conflict) ──────────
    synthesis = await layer2_synthesis.synthesize(
        model_predictions=model_predictions,
        medgemma=medgemma,
        vitals=request.vitals,
        client=ollama,
        model=settings.ollama_synthesis_model,
    )
    # If synthesis suggests a more urgent category, honour it (safer)
    if synthesis is not None and synthesis.recommended_category < final_category:
        final_category = synthesis.recommended_category

    # ── Safety rules ────────────────────────────────────────────────
    safety = apply_safety_rules(
        final_category=final_category,
        confidence=final_confidence,
        model_predictions=model_predictions,
        medgemma=medgemma,
        vitals=request.vitals,
        confidence_threshold=settings.confidence_threshold,
        conflict_threshold=settings.conflict_threshold,
    )

    # Layer 2 message overrides safety message when present
    conflict_message = synthesis.summary if synthesis is not None else safety.message

    # ── SHAP (best loaded model) ────────────────────────────────────
    shap_top: list[ShapValue] = []
    if settings.shap_enabled:
        best_id = max(
            (p.model_name for p in model_predictions),
            key=lambda m: layer1a_ml.MODEL_WEIGHTS.get(m, 0.0),
            default=None,
        )
        if best_id and best_id in registry.models:
            bundle = registry.models[best_id]
            X_aligned = align_features(df_row.copy(), bundle["feature_names"], registry.medians)
            shap_dicts = top_n_shap(
                model_id=best_id,
                model=bundle["model"],
                X=X_aligned,
                predicted_class=safety.final_category,
                top_n=settings.shap_top_n,
            )
            shap_top = [ShapValue(**s) for s in shap_dicts]

    duration_ms = int((time.perf_counter() - t_start) * 1000)
    log.info(
        "predict | final=%d conf=%.2f models=%d alert=%s synthesis=%s dur=%dms src=%s",
        safety.final_category,
        final_confidence,
        len(model_predictions),
        safety.alert_doctor,
        synthesis is not None,
        duration_ms,
        parsed.get("source", "?"),
    )

    return PredictResponse(
        final_category=safety.final_category,
        confidence=round(final_confidence, 4),
        model_predictions=model_predictions,
        medgemma=medgemma,
        shap_top5=shap_top,
        conflict=ConflictInfo(
            detected=safety.detected,
            severity=safety.severity,  # type: ignore[arg-type]
            alert_doctor=safety.alert_doctor,
            message=conflict_message,
        ),
        processing_time_ms=duration_ms,
    )
