"""
Random Forest — interpretable baseline.

Random Forest:
    - Łatwy w treningu i tuningu
    - Naturalnie odporny na overfitting
    - Daje feature importances out-of-the-box
    - Dobry baseline do porównań
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseTriageModel, compute_sample_weights
from src.utils.config import RF_DEFAULT_PARAMS
from src.utils.logger import get_logger

log = get_logger(__name__)


class RandomForestTriageModel(BaseTriageModel):
    """Random Forest dla 5-klasowej klasyfikacji triażu MTS."""

    name = "random_forest"

    def __init__(self, params: dict[str, Any] | None = None):
        merged = {**RF_DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged)

    def _build_model(self):
        return RandomForestClassifier(**self.params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        sample_weight_strategy: str = "custom",
        run_id: str | None = None,
        use_mlflow: bool = False,
        feature_set: str = "triage_only",
        **kwargs,
    ) -> "RandomForestTriageModel":
        """
        Trenuje Random Forest.

        Uwaga: RF nie ma per-iteration metric (jak XGBoost) — zapisujemy tylko końcowe.
        """
        # 1. Setup trackingu
        self._setup_tracking(run_id=run_id, use_mlflow=use_mlflow)
        self._log_data_info(X_train, y_train, X_val, y_val, feature_set=feature_set)

        self.train_logger.info(f"Trenowanie Random Forest (n={len(X_train):,}, features={X_train.shape[1]})")
        self.feature_names = list(X_train.columns)

        # 2. Wagi — preferuj custom z Optuny nad sklearn balanced
        sample_weights = None
        if self.optuna_class_weights:
            self.params["class_weight"] = self.optuna_class_weights
            sample_weights = compute_sample_weights(y_train, strategy="none", class_weights={})
            self.train_logger.info(
                f"Wagi: Optuna custom class_weight={self.optuna_class_weights}"
            )
        elif self.params.get("class_weight") not in (None, "balanced", "balanced_subsample"):
            sample_weights = compute_sample_weights(y_train, strategy=sample_weight_strategy, class_weights=self.optuna_class_weights)
            self.train_logger.info(
                f"Wagi sampli: strategia='{sample_weight_strategy}', "
                f"min={sample_weights.min():.2f}, max={sample_weights.max():.2f}"
            )
        else:
            self.train_logger.info(f"Wagi: class_weight='{self.params.get('class_weight', 'none')}' (sklearn internal)")

        # 3. Trenuj
        self.model = self._build_model()
        t0 = self._start_timer()

        if sample_weights is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            self.model.fit(X_train, y_train)

        duration = self._stop_timer(t0)

        self.classes_ = self.model.classes_
        self.is_fitted = True

        # 4. Loguj OOB score jeśli dostępny
        oob_score = getattr(self.model, "oob_score_", None)
        if oob_score is not None:
            self.train_logger.info(f"OOB score: {oob_score:.4f}")
            self.tracker.log_metrics({"oob_score": oob_score})

        # 5. Loguj train accuracy + val accuracy (jeśli mamy val)
        train_acc = self.model.score(X_train, y_train)
        self.train_logger.info(f"Train accuracy: {train_acc:.4f}")
        self.tracker.log_metrics({"train_accuracy": train_acc})

        if X_val is not None and y_val is not None:
            val_acc = self.model.score(X_val, y_val)
            self.train_logger.info(f"Val accuracy:   {val_acc:.4f}")
            self.tracker.log_metrics({"val_accuracy": val_acc})

        self.train_logger.info(f"Trening Random Forest zakończony w {duration:.1f}s")

        # 6. Finalizuj
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
