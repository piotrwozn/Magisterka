"""Ewaluacja modeli triażowych: metryki, CV, wizualizacje, UQ."""

from src.evaluation.cross_validation import (
    cross_validate_model,
    save_cv_results,
)
from src.evaluation.metrics import (
    full_evaluation,
    overtriage_rate,
    quadratic_weighted_kappa,
    undertriage_rate,
)
from src.evaluation.uncertainty import (
    compute_ensemble_uq,
    compute_model_uq,
    ensemble_report,
)
from src.evaluation.visualizations import (
    plot_calibration,
    plot_confusion_matrix,
    plot_feature_importances,
    plot_roc_curves,
    plot_training_history,
)

__all__ = [
    "compute_ensemble_uq",
    "compute_model_uq",
    "cross_validate_model",
    "ensemble_report",
    "full_evaluation",
    "overtriage_rate",
    "plot_calibration",
    "plot_confusion_matrix",
    "plot_feature_importances",
    "plot_roc_curves",
    "plot_training_history",
    "quadratic_weighted_kappa",
    "save_cv_results",
    "undertriage_rate",
]
