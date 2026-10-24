"""
Random Forest — jedyny model baggingowy w zestawie.

Rola w projekcie:
    - inny mechanizm niż boosting (uśrednianie niezależnych drzew), więc błędy
      są słabiej skorelowane z XGBoost/LightGBM/CatBoost — dywersyfikacja ensemble,
    - stabilny punkt odniesienia, mało wrażliwy na hiperparametry.

Braki (NaN) obsługiwane natywnie przez scikit-learn >= 1.4 — bez imputacji.
Brak early stopping: fit_estimator() ignoruje zbiór ES (w fit() służy tylko do
zalogowania dokładności). Prawdopodobieństwa RF bywają słabo skalibrowane; przy
łączeniu modeli warto je skalibrować na zbiorze `val` (zob. src/ensemble).
"""

from __future__ import annotations

from typing import Any

import sklearn
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseTriageModel
from src.utils.config import RF_DEFAULT_PARAMS
from src.utils.logger import get_logger

log = get_logger(__name__)


def _check_sklearn_nan_support() -> None:
    major, minor = (int(p) for p in sklearn.__version__.split(".")[:2])
    if (major, minor) < (1, 4):
        raise RuntimeError(
            f"RandomForest z brakami (NaN) wymaga scikit-learn >= 1.4 "
            f"(zainstalowano {sklearn.__version__}). Zaktualizuj: pip install -U scikit-learn"
        )


class RandomForestTriageModel(BaseTriageModel):
    """Random Forest dla 5-klasowej klasyfikacji ESI."""

    name = "random_forest"
    display_name = "Random Forest"

    def __init__(self, params: dict[str, Any] | None = None):
        merged = {**RF_DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged)

    def _build_model(self):
        params = dict(self.params)
        if not params.get("bootstrap", True):
            params.pop("max_samples", None)
        return RandomForestClassifier(**params)

    def _fit_impl(self, X, y, X_es, y_es, sample_weight) -> None:
        _check_sklearn_nan_support()
        self.model = self._build_model()
        self.model.fit(X, y, sample_weight=sample_weight)

    def _log_extra_details(self) -> None:
        n_nodes = sum(int(est.tree_.node_count) for est in self.model.estimators_)
        self.train_logger.info(
            f"Drzewa: {len(self.model.estimators_)}, węzłów łącznie: {n_nodes:,} "
            f"(~{n_nodes * 104 / 1e6:.0f} MB w RAM)"
        )
        self.tracker.log_metrics({"n_nodes_total": n_nodes})
