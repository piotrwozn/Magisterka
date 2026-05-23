"""
Explainable Boosting Machine — model "szklanej skrzynki".

EBM (Microsoft InterpretML) to algorytm GAM z interakcjami,
który osiąga wyniki bliskie XGBoost, ale jest INHERENTLY INTERPRETABLE.

UWAGA BADAWCZA (zob. TECHNICAL_ANALYSIS.md §3.4):
    Porównanie EBM z czarno-skrzynkowym XGBoost+SHAP to świetny materiał
    badawczy: EBM ma wbudowaną interpretowalność (każdy feature ma
    wizualizowalną funkcję), co w medycynie jest bezcenne.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseTriageModel, compute_sample_weights
from src.utils.config import EBM_DEFAULT_PARAMS
from src.utils.logger import get_logger

log = get_logger(__name__)


class EBMTriageModel(BaseTriageModel):
    """Explainable Boosting Machine dla 5-klasowej klasyfikacji triażu MTS."""

    name = "ebm"

    def __init__(self, params: dict[str, Any] | None = None):
        merged = {**EBM_DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged)

    def _build_model(self):
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(**self.params)

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
    ) -> "EBMTriageModel":
        """Trenuje EBM."""
        # 1. Setup trackingu
        self._setup_tracking(run_id=run_id, use_mlflow=use_mlflow)
        self._log_data_info(X_train, y_train, X_val, y_val, feature_set=feature_set)

        self.train_logger.info(f"Trenowanie EBM (n={len(X_train):,}, features={X_train.shape[1]})")
        self.feature_names = list(X_train.columns)

        # 2. Wagi
        sample_weights = compute_sample_weights(y_train, strategy=sample_weight_strategy, class_weights=self.optuna_class_weights)
        self.train_logger.info(
            f"Wagi sampli: strategia='{sample_weight_strategy}', "
            f"min={sample_weights.min():.2f}, max={sample_weights.max():.2f}"
        )

        # 3. Trenuj
        self.model = self._build_model()
        t0 = self._start_timer()

        try:
            self.model.fit(X_train, y_train, sample_weight=sample_weights)
        except TypeError:
            # Niektóre wersje EBM nie wspierają sample_weight — fallback
            self.train_logger.warning("EBM nie akceptuje sample_weight w tej wersji — używam wag domyślnych")
            self.model.fit(X_train, y_train)

        duration = self._stop_timer(t0)

        self.classes_ = self.model.classes_
        self.is_fitted = True

        # 4. Train + val accuracy
        train_acc = self.model.score(X_train, y_train)
        self.train_logger.info(f"Train accuracy: {train_acc:.4f}")
        self.tracker.log_metrics({"train_accuracy": train_acc})

        if X_val is not None and y_val is not None:
            val_acc = self.model.score(X_val, y_val)
            self.train_logger.info(f"Val accuracy:   {val_acc:.4f}")
            self.tracker.log_metrics({"val_accuracy": val_acc})

        self.train_logger.info(f"Trening EBM zakończony w {duration:.1f}s")

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
        return self.model.explain_local(X, y, name="EBM Local")
