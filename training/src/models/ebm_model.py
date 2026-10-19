"""
Explainable Boosting Machine — model "szklanej skrzynki".

EBM (Microsoft InterpretML) to algorytm GAM, który osiąga wyniki bliskie
XGBoost, ale jest INHERENTLY INTERPRETABLE.

UWAGA BADAWCZA:
    Porównanie EBM z czarno-skrzynkowym XGBoost+SHAP to świetny materiał
    badawczy: EBM ma wbudowaną interpretowalność (każdy feature ma
    wizualizowalną funkcję), co w medycynie jest bezcenne.

Braki (NaN) obsługiwane natywnie. Najwolniejszy model w zestawie (godziny na pełnym
zbiorze). EBM zrównolegla się tylko po workach (`outer_bags`): finalny model 14 worków,
do tuningu można podać mniej. Early stopping EBM robi na własnej wewnętrznej walidacji —
fit_estimator() ignoruje zbiór ES (w fit() służy tylko do zalogowania dokładności).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseTriageModel
from src.utils.config import EBM_DEFAULT_PARAMS
from src.utils.logger import get_logger

log = get_logger(__name__)


class EBMTriageModel(BaseTriageModel):
    """Explainable Boosting Machine dla 5-klasowej klasyfikacji ESI."""

    name = "ebm"
    display_name = "EBM"
    iteration_param = "max_rounds"

    def __init__(self, params: dict[str, Any] | None = None):
        merged = {**EBM_DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged)

    def _build_model(self):
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(**self.params)

    def _fit_impl(self, X, y, X_es, y_es, sample_weight) -> None:
        self.model = self._build_model()
        self.model.fit(X, y, sample_weight=sample_weight)

    def _after_fit(self, has_es: bool) -> None:
        super()._after_fit(has_es)
        # best_iteration_ (etapy × worki) liczy KROKI = rundy × liczba termów etapu;
        # rundy efektów głównych porównujemy z limitem max_rounds.
        self.iteration_cap = int(self.params.get("max_rounds") or 0) or None
        try:
            steps = np.asarray(self.model.best_iteration_)
            steps = steps.reshape(-1, steps.shape[-1])[0]          # etap 0 = efekty główne
            n_mains = sum(1 for t in self.model.term_features_ if len(t) == 1)
            self.best_iteration = int(math.ceil(float(steps.max()) / max(n_mains, 1)))
        except (AttributeError, IndexError, ValueError, TypeError):
            self.best_iteration = None

    def feature_importances(self) -> pd.DataFrame | None:
        """Ważność termów EBM (efekty główne; interakcje par cech, jeśli włączone)."""
        if self.model is None:
            return None
        try:
            importances = self.model.term_importances()
            names = list(self.model.term_names_)
        except AttributeError:
            return None
        df = pd.DataFrame({"feature": names, "importance": importances})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    # ──────── EBM-specific: wbudowana interpretowalność ────────
    def explain_global(self):
        """Zwraca obiekt explanation EBM dla globalnej interpretacji.

        Użycie:
            from interpret import show
            ebm = model.explain_global()
            show(ebm)  # interaktywna wizualizacja w przeglądarce
        """
        if not self.is_fitted:
            raise RuntimeError("Model nie wytrenowany.")
        return self.model.explain_global(name="EBM Global")

    def explain_local(self, X: pd.DataFrame, y: np.ndarray | pd.Series | None = None):
        """Lokalna interpretacja per-pacjent."""
        if not self.is_fitted:
            raise RuntimeError("Model nie wytrenowany.")
        return self.model.explain_local(self._align_features(X), y, name="EBM Local")
