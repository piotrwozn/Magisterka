from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from src.models.base import BaseTriageModel, compute_sample_weights
from src.utils.config import ET_DEFAULT_PARAMS
from src.utils.logger import get_logger

log = get_logger(__name__)


class ExtraTreesTriageModel(BaseTriageModel):
    """ExtraTrees dla 5-klasowej klasyfikacji triażu MTS."""

    name = "extra_trees"

    def __init__(self, params: dict[str, Any] | None = None):
        merged = {**ET_DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged)

    def _build_model(self):
        return ExtraTreesClassifier(**self.params)

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
    ) -> "ExtraTreesTriageModel":
        self._setup_tracking(run_id=run_id, use_mlflow=use_mlflow)
        self._log_data_info(X_train, y_train, X_val, y_val, feature_set=feature_set)

        self.train_logger.info(f"Trenowanie ExtraTrees (n={len(X_train):,}, features={X_train.shape[1]})")
        self.feature_names = list(X_train.columns)

        sample_weights = None
        if self.params.get("class_weight") not in ("balanced", "balanced_subsample"):
            sample_weights = compute_sample_weights(y_train, strategy=sample_weight_strategy, class_weights=self.optuna_class_weights)
            self.train_logger.info(
                f"Wagi sampli: strategia='{sample_weight_strategy}', "
                f"min={sample_weights.min():.2f}, max={sample_weights.max():.2f}"
            )
        else:
            self.train_logger.info(f"Wagi: class_weight='{self.params['class_weight']}' (sklearn internal)")

        self.model = self._build_model()
        t0 = self._start_timer()

        if sample_weights is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            self.model.fit(X_train, y_train)

        duration = self._stop_timer(t0)

        self.classes_ = self.model.classes_
        self.is_fitted = True

        oob_score = getattr(self.model, "oob_score_", None)
        if oob_score is not None:
            self.train_logger.info(f"OOB score: {oob_score:.4f}")
            self.tracker.log_metrics({"oob_score": oob_score})

        train_acc = self.model.score(X_train, y_train)
        self.train_logger.info(f"Train accuracy: {train_acc:.4f}")
        self.tracker.log_metrics({"train_accuracy": train_acc})

        if X_val is not None and y_val is not None:
            val_acc = self.model.score(X_val, y_val)
            self.train_logger.info(f"Val accuracy:   {val_acc:.4f}")
            self.tracker.log_metrics({"val_accuracy": val_acc})

        self.train_logger.info(f"Trening ExtraTrees zakończony w {duration:.1f}s")

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
