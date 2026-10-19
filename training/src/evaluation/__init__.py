"""
Ewaluacja modeli ESI: metryki, reguły decyzyjne, CV, wizualizacje, UQ.

Metryki importowane od razu (lekkie, używa ich trening i src.ensemble).
CV, wizualizacje (matplotlib/seaborn) i UQ ładowane leniwie przy pierwszym
użyciu nazwy — `import src.evaluation.metrics` nie wciąga matplotlib, a
`python -m src.evaluation.uncertainty` nie importuje modułu dwukrotnie.
"""

from __future__ import annotations

import importlib
from typing import Any

from src.evaluation.metrics import (
    brier_score,
    cost_matrix,
    critical_miss_rate,
    decide,
    expected_cost,
    full_evaluation,
    linear_weighted_kappa,
    mistriage_rates,
    overtriage_rate,
    quadratic_weighted_kappa,
    undertriage_rate,
)

# nazwa → moduł (import leniwy)
_LAZY_EXPORTS: dict[str, str] = {
    "cross_validate_model": "src.evaluation.cross_validation",
    "save_cv_results": "src.evaluation.cross_validation",
    "summarize_cv": "src.evaluation.cross_validation",
    "compute_ensemble_uq": "src.evaluation.uncertainty",
    "compute_model_uq": "src.evaluation.uncertainty",
    "ensemble_report": "src.evaluation.uncertainty",
    "ESI_COLORS": "src.evaluation.visualizations",
    "plot_calibration": "src.evaluation.visualizations",
    "plot_confusion_matrix": "src.evaluation.visualizations",
    "plot_feature_importances": "src.evaluation.visualizations",
    "plot_roc_curves": "src.evaluation.visualizations",
    "plot_training_history": "src.evaluation.visualizations",
}


def __getattr__(name: str) -> Any:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "ESI_COLORS",
    "brier_score",
    "compute_ensemble_uq",
    "compute_model_uq",
    "cost_matrix",
    "critical_miss_rate",
    "cross_validate_model",
    "decide",
    "ensemble_report",
    "expected_cost",
    "full_evaluation",
    "linear_weighted_kappa",
    "mistriage_rates",
    "overtriage_rate",
    "plot_calibration",
    "plot_confusion_matrix",
    "plot_feature_importances",
    "plot_roc_curves",
    "plot_training_history",
    "quadratic_weighted_kappa",
    "save_cv_results",
    "summarize_cv",
    "undertriage_rate",
]
