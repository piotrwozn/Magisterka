"""Layer 1A — ensemble inference across loaded ML models.

Weighted blend of the class probabilities of the loaded models (arithmetic or
geometric mean), optional temperature scaling of the blend, then the decision rule
(`registry.ensemble`: equal weights, mean, no temperature and argmax by default;
optional override from `backend/models/ensemble_weights.json` — format in
app/models/ensemble.py). Class index 0–4 = ESI 1–5.

Foundation models (TFM_MODELS) are served only on opt-in (app/models/registry.py).
A served TFM that fails during a request is dropped from the registry and — when
`ensemble_weights_no_tfm.json` was deployed — the request is recomputed with that
recipe (re-fitted without TFMs) instead of renormalising a calibrated full blend.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.models.ensemble import MODEL_WEIGHTS, TFM_MODELS, EnsembleConfig, decide
from app.models.registry import N_CLASSES, ModelRegistry
from app.models.schemas import ModelPrediction
from app.pipeline.feature_engineering import align_features

log = logging.getLogger(__name__)

__all__ = ["MODEL_WEIGHTS", "full_proba", "predict_with_model", "run_ensemble"]


def full_proba(model, X: pd.DataFrame, classes: list[int] | None = None) -> np.ndarray:
    """predict_proba of one row → 5 probabilities ordered ESI 1..5 (float64, sum 1).

    A class absent in training (raw estimator returns fewer columns) gets 0.
    """
    raw = np.asarray(model.predict_proba(X), dtype=float)[0]
    classes = list(range(N_CLASSES)) if classes is None else list(classes)
    if len(raw) != len(classes):
        raise ValueError(f"predict_proba returned {len(raw)} columns for classes {classes}")
    proba = np.zeros(N_CLASSES)
    for col, cls in enumerate(classes):
        proba[int(cls)] = raw[col]
    total = proba.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"invalid probabilities {raw.tolist()}")
    return proba / total


def predict_with_model(
    model_id: str,
    df_row: pd.DataFrame,
    registry: ModelRegistry,
    rule: str = "argmax",
) -> ModelPrediction:
    """Single-model prediction with feature alignment."""
    bundle = registry.models[model_id]
    X = align_features(df_row.copy(), bundle["feature_names"], registry.medians)
    proba = full_proba(bundle["model"], X, bundle.get("classes"))
    category = decide(proba, rule)
    return ModelPrediction(
        model_name=model_id,
        category=category,
        probabilities=[float(p) for p in proba],
        confidence=float(proba[category]),
    )


def run_ensemble(
    df_row: pd.DataFrame,
    registry: ModelRegistry,
    config: EnsembleConfig | None = None,
) -> tuple[int, float, list[ModelPrediction]]:
    """Run the loaded models with weight > 0 and combine their probabilities.

    Returns: (final_category, final_confidence, per_model_predictions)
    """
    if not registry.models:
        raise RuntimeError("No models loaded — cannot run ensemble")
    use_registry_config = config is None
    config = config or registry.ensemble
    active = config.active(registry.loaded_ids)
    if not active:
        raise RuntimeError(
            f"No loaded model has a weight > 0 ({config.source}); loaded: {registry.loaded_ids}"
        )

    per_model: list[ModelPrediction] = []
    probas: dict[str, np.ndarray] = {}

    for model_id in active:
        try:
            pred = predict_with_model(model_id, df_row, registry, config.rule)
        except Exception as exc:  # noqa: BLE001
            log.warning("Model %s failed: %s", model_id, exc)
            # A TFM failing at request time (OOM, CUDA) keeps failing — drop it once and, when the
            # registry switched to the no-TFM recipe, recompute this request with that recipe
            if (model_id in TFM_MODELS and use_registry_config and hasattr(registry, "drop_tfm")
                    and registry.drop_tfm(model_id, f"{type(exc).__name__}: {exc}")):
                return run_ensemble(df_row, registry)
            continue

        per_model.append(pred)
        probas[model_id] = np.array(pred.probabilities)

    if not probas:
        raise RuntimeError("All models failed during inference")

    # Weights renormalised over the models that answered; temperature on the blend
    final_proba = config.combine(probas)
    final_category = decide(final_proba, config.rule)
    final_confidence = float(final_proba[final_category])

    return final_category, final_confidence, per_model
