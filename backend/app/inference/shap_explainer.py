"""SHAP top-N explainer for tree-based models."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except ImportError:
    shap = None  # type: ignore
    SHAP_AVAILABLE = False
    log.warning("shap not installed — SHAP explanations disabled")


# Per-process cache of TreeExplainer instances keyed by (model_id, id(model))
# — including id(model) prevents stale cache when the underlying model
# instance is swapped (e.g. between tests with mocks and real joblib loads).
_explainer_cache: dict[tuple[str, int], Any] = {}


def clear_cache() -> None:
    """Clear cached SHAP explainers (mostly for tests)."""
    _explainer_cache.clear()


def _build_explainer(model_id: str, model: Any):
    """Build and cache a TreeExplainer instance per (model_id, model_instance)."""
    if not SHAP_AVAILABLE:
        return None
    key = (model_id, id(model))
    if key in _explainer_cache:
        return _explainer_cache[key]
    try:
        explainer = shap.TreeExplainer(model)
        _explainer_cache[key] = explainer
        return explainer
    except Exception as exc:
        log.warning("Failed to build SHAP explainer for %s: %s", model_id, exc)
        _explainer_cache[key] = None
        return None


def top_n_shap(
    model_id: str,
    model: Any,
    X: pd.DataFrame,
    predicted_class: int,
    top_n: int = 5,
) -> list[dict]:
    """Return top-N features by absolute SHAP value for the predicted class."""
    if not SHAP_AVAILABLE:
        return []

    explainer = _build_explainer(model_id, model)
    if explainer is None:
        return []

    try:
        shap_values = explainer.shap_values(X)
    except Exception as exc:
        log.warning("SHAP shap_values failed for %s: %s", model_id, exc)
        return []

    # Normalise to shape (n_classes, 1, n_features) or (1, n_features, n_classes)
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # Common layouts: (classes, samples, features) or (samples, features, classes)
        if arr.shape[0] == 5:
            values = arr[predicted_class, 0]
        elif arr.shape[-1] == 5:
            values = arr[0, :, predicted_class]
        else:
            values = arr[0]
    elif isinstance(shap_values, list) and len(shap_values) > predicted_class:
        values = np.asarray(shap_values[predicted_class])[0]
    else:
        values = arr[0] if arr.ndim == 2 else arr

    values = np.asarray(values).flatten()
    feature_names = list(X.columns)

    if len(values) != len(feature_names):
        log.warning("SHAP length mismatch for %s: %d vs %d", model_id, len(values), len(feature_names))
        return []

    top_idx = np.argsort(np.abs(values))[-top_n:][::-1]
    return [
        {
            "feature":   feature_names[i],
            "value":     float(values[i]),
            "direction": "positive" if values[i] > 0 else "negative",
        }
        for i in top_idx
    ]
