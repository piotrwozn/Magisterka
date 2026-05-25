"""Layer 1A — ensemble inference across loaded ML models.

Produces per-model predictions plus a stacked/averaged final category.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.models.registry import ModelRegistry
from app.models.schemas import ModelPrediction
from app.pipeline.feature_engineering import align_features

log = logging.getLogger(__name__)


# Real test-set QWK (used as ensemble weights when no stacking meta-learner)
MODEL_WEIGHTS: dict[str, float] = {
    "catboost":      0.8729,
    "lightgbm":      0.8705,
    "xgboost":       0.876,
    "random_forest": 0.84,
    "extra_trees":   0.83,
    "hist_gbt":      0.84,
    "ebm":           0.81,
}


def predict_with_model(
    model_id: str,
    df_row: pd.DataFrame,
    registry: ModelRegistry,
) -> ModelPrediction:
    """Single-model prediction with feature alignment."""
    bundle = registry.models[model_id]
    X = align_features(df_row.copy(), bundle["feature_names"], registry.medians)

    proba = bundle["model"].predict_proba(X)[0]
    category = int(np.argmax(proba))
    return ModelPrediction(
        model_name=model_id,
        category=category,
        probabilities=[float(p) for p in proba],
        confidence=float(proba[category]),
    )


def run_ensemble(
    df_row: pd.DataFrame,
    registry: ModelRegistry,
) -> tuple[int, float, list[ModelPrediction]]:
    """Run all loaded models and combine via weighted average of probabilities.

    Returns: (final_category, final_confidence, per_model_predictions)
    """
    if not registry.models:
        raise RuntimeError("No models loaded — cannot run ensemble")

    per_model: list[ModelPrediction] = []
    weighted_proba = np.zeros(5)
    total_weight = 0.0

    for model_id in registry.loaded_ids:
        try:
            pred = predict_with_model(model_id, df_row, registry)
        except Exception as exc:  # noqa: BLE001
            log.warning("Model %s failed: %s", model_id, exc)
            continue

        per_model.append(pred)
        weight = MODEL_WEIGHTS.get(model_id, 0.85)
        weighted_proba += np.array(pred.probabilities) * weight
        total_weight += weight

    if total_weight == 0:
        raise RuntimeError("All models failed during inference")

    weighted_proba /= total_weight
    final_category = int(np.argmax(weighted_proba))
    final_confidence = float(weighted_proba[final_category])

    return final_category, final_confidence, per_model
