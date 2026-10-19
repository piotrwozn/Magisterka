"""
Random Forest — jedyny model baggingowy w zestawie.

Rola w projekcie:
    - inny mechanizm niż boosting (uśrednianie niezależnych drzew), więc błędy
      są słabiej skorelowane z XGBoost/LightGBM/CatBoost — dywersyfikacja ensemble,
    - stabilny punkt odniesienia, mało wrażliwy na hiperparametry.

Braki (NaN) obsługiwane natywnie przez scikit-learn >= 1.4 — bez imputacji.
Prawdopodobieństwa RF bywają słabo skalibrowane; przy łączeniu modeli warto
je skalibrować na zbiorze `val` (zob. src/ensemble).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseTriageModel, compute_sample_weights
from src.utils.config import DEFAULT_SAMPLE_WEIGHT_STRATEGY, RF_DEFAULT_PARAMS
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

    def __init__(self, params: dict[str, Any] | None = None):
        merged = {**RF_DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged)

    def _build_model(self):
        params = dict(self.params)
        if not params.get("bootstrap", True):
            params.pop("max_samples", None)
        return RandomForestClassifier(**params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        sample_weight_strategy: str = DEFAULT_SAMPLE_WEIGHT_STRATEGY,
        run_id: str | None = None,
        use_mlflow: bool = False,
        feature_set: str = "triage",
        **kwargs,
    ) -> "RandomForestTriageModel":
        """
        Trenuje Random Forest.

        RF nie ma early stopping — zbiór X_val służy tylko do zalogowania dokładności.
        """
        _check_sklearn_nan_support()

        # 1. Setup trackingu
        self._setup_tracking(run_id=run_id, use_mlflow=use_mlflow)
        self._log_data_info(X_train, y_train, X_val, y_val, feature_set=feature_set)

        self.train_logger.info(f"Trenowanie Random Forest (n={len(X_train):,}, features={X_train.shape[1]})")
        self.feature_names = list(X_train.columns)

        # 2. Wagi — jedno źródło prawdy: sample_weight (Optuna > strategia)
        sample_weights = compute_sample_weights(
            y_train, strategy=sample_weight_strategy, class_weights=self.optuna_class_weights
        )
        self.train_logger.info(
            f"Wagi sampli: strategia='{sample_weight_strategy}', "
            f"min={sample_weights.min():.2f}, max={sample_weights.max():.2f}"
        )

        # 3. Trenuj
        self.model = self._build_model()
        t0 = self._start_timer()
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        duration = self._stop_timer(t0)

        self.classes_ = self.model.classes_
        self.is_fitted = True

        # 4. Dokładność na zbiorze early stopping (orientacyjnie)
        if X_val is not None and y_val is not None:
            val_acc = self.model.score(X_val, y_val)
            self.train_logger.info(f"Accuracy (zbiór ES): {val_acc:.4f}")
            self.tracker.log_metrics({"es_accuracy": val_acc})

        self.train_logger.info(f"Trening Random Forest zakończony w {duration:.1f}s")

        # 5. Finalizuj
        self._finalize_tracking()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model nie wytrenowany.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model nie wytrenowany.")
        return self.model.predict_proba(X)
