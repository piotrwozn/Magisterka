"""
Fusion model — Stacking ensemble.

Late fusion w wersji prostszej: stacking z LogisticRegression jako meta-learner.

Architektura:
    [XGBoost, LightGBM, RandomForest] (base)
                ↓
    LogisticRegression (meta — łączy probabilities)
                ↓
    Final prediction
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from src.models.base import BaseTriageModel, compute_sample_weights
from src.models.lightgbm_model import LightGBMTriageModel
from src.models.random_forest import RandomForestTriageModel
from src.models.xgboost_model import XGBoostTriageModel
from src.utils.config import RANDOM_SEED
from src.utils.logger import get_logger

log = get_logger(__name__)


# Domyślne hiperparametry dla stacking
STACKING_DEFAULT_PARAMS: dict[str, Any] = {
    "cv": 5,
    "n_jobs": -1,
    "passthrough": False,        # nie dodawaj original features do meta-input
    "stack_method": "predict_proba",
}

META_LR_PARAMS: dict[str, Any] = {
    "max_iter": 1000,
    "C": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


class StackingTriageModel(BaseTriageModel):
    """
    Stacking ensemble: XGBoost + LightGBM + RandomForest → LogisticRegression.

    Domyślnie używa lekkich konfiguracji base learners (mniej drzew),
    by trening był rozsądnie szybki.
    """

    name = "stacking"

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        base_models: list[tuple[str, Any]] | None = None,
        meta_params: dict[str, Any] | None = None,
    ):
        merged = {**STACKING_DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged)
        self._user_base_models = base_models
        self.meta_params = {**META_LR_PARAMS, **(meta_params or {})}

    def _build_base_models(self) -> list[tuple[str, Any]]:
        """Domyślny zestaw base learners."""
        if self._user_base_models is not None:
            return self._user_base_models

        # Lżejsze wersje (mniej drzew) by stacking trenował się szybciej
        xgb_params = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1}
        lgbm_params = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1}
        rf_params = {"n_estimators": 200, "max_depth": 15}

        return [
            ("xgb", XGBoostTriageModel(params=xgb_params)._build_model()),
            ("lgbm", LightGBMTriageModel(params=lgbm_params)._build_model()),
            ("rf", RandomForestTriageModel(params=rf_params)._build_model()),
        ]

    def _build_model(self):
        base_models = self._build_base_models()
        meta = LogisticRegression(**self.meta_params)
        return StackingClassifier(
            estimators=base_models,
            final_estimator=meta,
            **self.params,
        )

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
    ) -> "StackingTriageModel":
        """Trenuje Stacking ensemble (CV wewnątrz)."""
        # 1. Setup trackingu
        self._setup_tracking(run_id=run_id, use_mlflow=use_mlflow)
        self._log_data_info(X_train, y_train, X_val, y_val, feature_set=feature_set)

        self.train_logger.info(f"Trenowanie Stacking (n={len(X_train):,}, features={X_train.shape[1]})")
        self.feature_names = list(X_train.columns)

        # 2. Wagi (uwaga: nie wszystkie base learners przyjmują sample_weight w stacking)
        sample_weights = compute_sample_weights(y_train, strategy=sample_weight_strategy)
        self.train_logger.info(
            f"Wagi sampli: strategia='{sample_weight_strategy}', "
            f"min={sample_weights.min():.2f}, max={sample_weights.max():.2f}"
        )

        base_models = self._build_base_models()
        self.train_logger.info(f"Base learners: {[name for name, _ in base_models]}")
        self.train_logger.info(f"Meta learner: LogisticRegression({self.meta_params})")

        # 3. Trenuj
        self.model = self._build_model()
        t0 = self._start_timer()

        try:
            self.model.fit(X_train, y_train, sample_weight=sample_weights)
        except (TypeError, ValueError) as e:
            self.train_logger.warning(f"Stacking nie wspiera sample_weight: {e}. Trenuje bez wag.")
            self.model.fit(X_train, y_train)

        duration = self._stop_timer(t0)

        self.classes_ = self.model.classes_
        self.is_fitted = True

        # 4. Metryki accuracy
        train_acc = self.model.score(X_train, y_train)
        self.train_logger.info(f"Train accuracy: {train_acc:.4f}")
        self.tracker.log_metrics({"train_accuracy": train_acc})

        if X_val is not None and y_val is not None:
            val_acc = self.model.score(X_val, y_val)
            self.train_logger.info(f"Val accuracy:   {val_acc:.4f}")
            self.tracker.log_metrics({"val_accuracy": val_acc})

        self.train_logger.info(f"Trening Stacking zakończony w {duration:.1f}s")

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

    def feature_importances(self) -> pd.DataFrame | None:
        """
        Stacking nie ma pojedynczego feature_importances_ —
        agregujemy z base learners (jeśli mają).
        """
        if not self.is_fitted:
            return None

        all_importances = []
        for name, base_estimator in self.model.named_estimators_.items():
            imp = getattr(base_estimator, "feature_importances_", None)
            if imp is not None:
                all_importances.append(imp)

        if not all_importances:
            return None

        # Średnia ważności across base learners (po normalizacji każdy do sumy=1)
        normalized = [imp / (imp.sum() + 1e-12) for imp in all_importances]
        avg_imp = np.mean(normalized, axis=0)

        names = self.feature_names or [f"f_{i}" for i in range(len(avg_imp))]
        df = pd.DataFrame({"feature": names, "importance": avg_imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
