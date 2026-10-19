"""Layer 1A — ensemble inference across loaded ML models.

Weighted average of the class probabilities of the loaded models + decision rule
(`registry.ensemble`: equal weights and argmax by default, optional override from
`backend/models/ensemble_weights.json` — format in app/models/ensemble.py).
Class index 0–4 = ESI 1–5.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.models.ensemble import MODEL_WEIGHTS, EnsembleConfig, decide
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
    config = config or registry.ensemble
    active = config.active(registry.loaded_ids)
    if not active:
        raise RuntimeError(
            f"No loaded model has a weight > 0 ({config.source}); loaded: {registry.loaded_ids}"
        )

    per_model: list[ModelPrediction] = []
    weighted_proba = np.zeros(N_CLASSES)
    total_weight = 0.0

    for model_id in active:
        try:
            pred = predict_with_model(model_id, df_row, registry, config.rule)
        except Exception as exc:  # noqa: BLE001
            log.warning("Model %s failed: %s", model_id, exc)
            continue

        per_model.append(pred)
        weight = config.weight(model_id)
        weighted_proba += np.array(pred.probabilities) * weight
        total_weight += weight

    if total_weight == 0:
        raise RuntimeError("All models failed during inference")

    weighted_proba /= total_weight
    final_category = decide(weighted_proba, config.rule)
    final_confidence = float(weighted_proba[final_category])

    return final_category, final_confidence, per_model
