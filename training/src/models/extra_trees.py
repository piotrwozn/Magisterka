from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseTriageModel, compute_sample_weights
from src.utils.config import ET_DEFAULT_PARAMS
from src.utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────
# GPU/CPU auto-selection
# ─────────────────────────────────────────
# UWAGA: cuML nie ma ExtraTreesClassifier — używamy RandomForestClassifier
# z bootstrap=False jako GPU-accelerated odpowiednika (brak baggingu).
# Oznacza to, że cuML wariant NIE ma losowych progów splitów
# (extremely randomized trees) — wykorzystuje kwantylowe splity z RF,
# ale na pełnym datassecie (bootstrap=False).
_USE_CUML = False
_CUML_BACKEND = None
try:
    from cuml.ensemble import RandomForestClassifier as _CUML_BACKEND

    _USE_CUML = True
    log.info("cuML RandomForestClassifier available — using GPU backend (bootstrap=False)")
except ImportError:
    from sklearn.ensemble import ExtraTreesClassifier as _CUML_BACKEND

    log.info("cuML not found — using sklearn CPU backend")


def _clean_params_for_backend(params: dict[str, Any]) -> dict[str, Any]:
    """
    Filtruje/mapuje hiperparametry do zestawu akceptowanego przez bieżący backend.

    Różnice sklearn ↔ cuML ExtraTreesClassifier:
      - criterion → split_criterion (cuML)
      - max_leaf_nodes → max_leaves (cuML, -1 = unlimited)
      - cuML nie wspiera: min_impurity_decrease, ccp_alpha,
        min_weight_fraction_leaf, warm_start, oob_score, class_weight
      - class_weight zawsze usuwany — wagi idą przez sample_weight w fit()
    """
    backend_is_cuml = _USE_CUML and _CUML_BACKEND is not None and hasattr(_CUML_BACKEND, "split_criterion")

    cleaned = {}

    for k, v in params.items():
        # Pomijamy parametry nieobsługiwane przez żaden backend
        if k in ("oob_score", "warm_start", "verbose"):
            continue

        # class_weight — cuML nie wspiera, sklearn tak
        if k == "class_weight" and backend_is_cuml:
            continue

        # Mapowanie criterion → split_criterion dla cuML
        if k == "criterion" and backend_is_cuml:
            criterion_map = {"gini": "gini", "entropy": "entropy", "log_loss": "entropy"}
            cleaned["split_criterion"] = criterion_map.get(v, v)
            continue

        if k == "max_leaf_nodes" and backend_is_cuml:
            cleaned["max_leaves"] = -1 if v is None else int(v)
            continue

        # Pomijamy parametry czysto sklearn gdy używamy cuML
        if backend_is_cuml and k in (
            "min_impurity_decrease",
            "ccp_alpha",
            "min_weight_fraction_leaf",
            "max_leaf_nodes",
        ):
            continue

        cleaned[k] = v

    return cleaned


class ExtraTreesTriageModel(BaseTriageModel):
    """ExtraTrees dla 5-klasowej klasyfikacji triażu MTS.

    Automatycznie wybiera GPU (cuML) lub CPU (sklearn) w zależności
    od dostępności biblioteki cuml.
    """

    name = "extra_trees"

    def __init__(self, params: dict[str, Any] | None = None):
        merged = {**ET_DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged)

    def _build_model(self):
        backend_params = _clean_params_for_backend(self.params)

        if _USE_CUML:
            # cuML RF z bootstrap=False zamiast ExtraTrees (którego cuML nie ma)
            backend_params.setdefault("bootstrap", False)
            model = _CUML_BACKEND(**backend_params)
            self.train_logger.info("Backend: cuML RandomForest(bootstrap=False) [GPU]")
        else:
            model = _CUML_BACKEND(**backend_params)
            self.train_logger.info("Backend: sklearn ExtraTreesClassifier [CPU]")

        return model

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

        # Wagi — cuML nie wspiera class_weight=dict, więc zawsze sample_weight
        sample_weights = None
        if self.optuna_class_weights:
            if _USE_CUML:
                # cuML: wagi przez sample_weight
                sample_weights = compute_sample_weights(y_train, strategy="none", class_weights=self.optuna_class_weights)
                self.train_logger.info(
                    f"Wagi (cuML): Optuna custom class_weights={self.optuna_class_weights} → sample_weight"
                )
            else:
                # sklearn: wagi przez class_weight param + sample_weight=1.0
                self.params["class_weight"] = self.optuna_class_weights
                sample_weights = compute_sample_weights(y_train, strategy="none", class_weights={})
                self.train_logger.info(
                    f"Wagi (sklearn): Optuna custom class_weight={self.optuna_class_weights}"
                )
        elif _USE_CUML:
            # cuML nie ma class_weight — wymuszamy sample_weight
            sample_weights = compute_sample_weights(y_train, strategy=sample_weight_strategy, class_weights=self.optuna_class_weights)
            self.train_logger.info(
                f"Wagi sampli (cuML forced): strategia='{sample_weight_strategy}', "
                f"min={sample_weights.min():.2f}, max={sample_weights.max():.2f}"
            )
        elif self.params.get("class_weight") not in (None, "balanced", "balanced_subsample"):
            sample_weights = compute_sample_weights(y_train, strategy=sample_weight_strategy, class_weights=self.optuna_class_weights)
            self.train_logger.info(
                f"Wagi sampli: strategia='{sample_weight_strategy}', "
                f"min={sample_weights.min():.2f}, max={sample_weights.max():.2f}"
            )
        else:
            self.train_logger.info(f"Wagi: class_weight='{self.params.get('class_weight', 'none')}' (sklearn internal)")

        self.model = self._build_model()
        t0 = self._start_timer()

        if sample_weights is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            self.model.fit(X_train, y_train)

        duration = self._stop_timer(t0)

        self.classes_ = self.model.classes_
        self.is_fitted = True

        # OOB score — tylko sklearn
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
