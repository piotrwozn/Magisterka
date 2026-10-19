"""
Regresja logistyczna (multinomialna) — liniowy punkt odniesienia.

Po co w pracy: pokazuje, ile zyskują modele nieliniowe względem prostego,
w pełni przejrzystego modelu (standard w artykułach o ML w medycynie ratunkowej).

Pipeline (dopasowany wyłącznie na zbiorze treningowym):
    imputacja medianą → standaryzacja → LogisticRegression.
Informacja o brakach trafia do modelu przez cechy `*_missing`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseTriageModel, compute_sample_weights
from src.utils.config import DEFAULT_SAMPLE_WEIGHT_STRATEGY, LOGREG_DEFAULT_PARAMS
from src.utils.logger import get_logger

log = get_logger(__name__)


class LogisticRegressionTriageModel(BaseTriageModel):
    """Multinomialna regresja logistyczna dla 5-klasowej klasyfikacji ESI."""

    name = "logreg"

    def __init__(self, params: dict[str, Any] | None = None):
        merged = {**LOGREG_DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged)

    def _build_model(self) -> Pipeline:
        return Pipeline([
            # keep_empty_features: kolumna bez wartości w próbce nie znika (stała liczba cech)
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(**self.params)),
        ])

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
    ) -> "LogisticRegressionTriageModel":
        """Trenuje pipeline imputacja → skalowanie → regresja logistyczna."""
        self._setup_tracking(run_id=run_id, use_mlflow=use_mlflow)
        self._log_data_info(X_train, y_train, X_val, y_val, feature_set=feature_set)

        self.train_logger.info(
            f"Trenowanie regresji logistycznej (n={len(X_train):,}, features={X_train.shape[1]})"
        )
        self.feature_names = list(X_train.columns)

        sample_weights = compute_sample_weights(
            y_train, strategy=sample_weight_strategy, class_weights=self.optuna_class_weights
        )
        self.train_logger.info(
            f"Wagi sampli: strategia='{sample_weight_strategy}', "
            f"min={sample_weights.min():.2f}, max={sample_weights.max():.2f}"
        )

        self.model = self._build_model()
        t0 = self._start_timer()
        self.model.fit(X_train, y_train, clf__sample_weight=sample_weights)
        duration = self._stop_timer(t0)

        self.classes_ = self.model.classes_
        self.is_fitted = True

        n_iter = int(np.max(self.model.named_steps["clf"].n_iter_))
        self.train_logger.info(f"Iteracje solvera: {n_iter}")
        self.tracker.log_metrics({"n_iter": n_iter})
        if X_val is not None and y_val is not None:
            val_acc = self.model.score(X_val, y_val)
            self.train_logger.info(f"Accuracy (zbiór ES): {val_acc:.4f}")
            self.tracker.log_metrics({"es_accuracy": val_acc})

        self.train_logger.info(f"Trening regresji logistycznej zakończony w {duration:.1f}s")
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

    def feature_importances(self) -> pd.DataFrame | None:
        """|współczynnik| na standaryzowanych cechach, uśredniony po klasach."""
        if self.model is None:
            return None
        coef = np.abs(self.model.named_steps["clf"].coef_).mean(axis=0)
        names = self.feature_names or [f"f_{i}" for i in range(len(coef))]
        df = pd.DataFrame({"feature": names, "importance": coef})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
