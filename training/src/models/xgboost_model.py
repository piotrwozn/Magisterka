"""
XGBoost — główny model produkcyjny.

XGBoost wybrany jako model bazowy z dwóch powodów:
    1. Doskonała wydajność na danych tabularnych (szczególnie z brakami).
    2. TreeSHAP — najszybsza i najdokładniejsza metoda wyjaśnień.

Cały trening jest w pełni logowany:
    - Per-iteration metrics (mlogloss, merror) → JSON
    - Logi tekstowe co N iteracji → plik .log
    - Hiperparametry, statystyki danych, czas treningu → JSON eksperymentu
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseTriageModel, compute_sample_weights
from src.utils.config import XGB_DEFAULT_PARAMS
from src.utils.logger import get_logger

log = get_logger(__name__)


class XGBoostTriageModel(BaseTriageModel):
    """Model XGBoost dla 5-klasowej klasyfikacji triażu MTS."""

    name = "xgboost"

    def __init__(self, params: dict[str, Any] | None = None):
        merged = {**XGB_DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged)

    def _build_model(self):
        import xgboost as xgb
        return xgb.XGBClassifier(**self.params)

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
        verbose: int | bool = 100,
        **kwargs,
    ) -> "XGBoostTriageModel":
        """
        Trenuje XGBoost z pełnym logowaniem.

        Parameters
        ----------
        X_train, y_train : dane treningowe
        X_val, y_val : dane walidacyjne (do early stopping)
        sample_weight_strategy : 'balanced' | 'custom' | 'none'
        run_id : str, optional
            Identyfikator eksperymentu (timestamp). Auto jeśli None.
        use_mlflow : bool
            Czy logować równolegle do MLflow.
        feature_set : str
            Etykieta zestawu cech (do metadanych).
        verbose : int | bool
            Co ile rund logować postęp.
        """
        # 1. Setup trackingu (logger, JSON eksperymentu)
        self._setup_tracking(run_id=run_id, use_mlflow=use_mlflow)
        self._log_data_info(X_train, y_train, X_val, y_val, feature_set=feature_set)

        self.train_logger.info(f"Trenowanie XGBoost (n={len(X_train):,}, features={X_train.shape[1]})")
        self.feature_names = list(X_train.columns)

        # 2. Wagi sampli
        sample_weights = compute_sample_weights(y_train, strategy=sample_weight_strategy, class_weights=self.optuna_class_weights)
        self.train_logger.info(
            f"Wagi sampli: strategia='{sample_weight_strategy}', "
            f"min={sample_weights.min():.2f}, max={sample_weights.max():.2f}"
        )

        # 3. Eval set
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

        # 4. Trenuj
        self.model = self._build_model()
        t0 = self._start_timer()

        fit_kwargs = {
            "sample_weight": sample_weights,
            "verbose": False,  # zastępujemy własnym logowaniem
        }
        if eval_set:
            fit_kwargs["eval_set"] = eval_set

        self.model.fit(X_train, y_train, **fit_kwargs)

        duration = self._stop_timer(t0)

        # 5. Wyciągnij training history z modelu
        self._extract_evals_result(verbose=verbose)

        self.classes_ = self.model.classes_
        self.is_fitted = True

        best_iter = getattr(self.model, "best_iteration", None)
        if best_iter is not None:
            self.train_logger.info(f"Best iteration: {best_iter}")
            self.tracker.add_note(f"Early stopping na iter {best_iter}")
            self.tracker.log_metrics({"best_iteration": best_iter})

        self.train_logger.info(f"Trening XGBoost zakończony w {duration:.1f}s")

        # 6. Finalizuj tracker (zapis JSON)
        self._finalize_tracking()

        return self

    def _extract_evals_result(self, verbose: int | bool = 100) -> None:
        """
        Wyciąga `evals_result_` z XGBoost i zapisuje do trackera + loggera.

        XGBoost przechowuje historię ewaluacji w słowniku:
            {validation_0: {mlogloss: [...], merror: [...]},
             validation_1: {mlogloss: [...], merror: [...]}}
        """
        evals_result = getattr(self.model, "evals_result_", None)
        if not evals_result:
            return

        # Zmapuj indeksy validation_0/1 → 'train'/'val'
        eval_set_keys = list(evals_result.keys())
        rename = {}
        if len(eval_set_keys) >= 1:
            rename[eval_set_keys[0]] = "train"
        if len(eval_set_keys) >= 2:
            rename[eval_set_keys[1]] = "val"

        # Wyciągnij liczbę iteracji
        first_key = eval_set_keys[0]
        first_metric = next(iter(evals_result[first_key].values()))
        n_iters = len(first_metric)

        log_every = verbose if isinstance(verbose, int) and verbose > 0 else 100

        for i in range(n_iters):
            metrics = {}
            for set_key, set_metrics in evals_result.items():
                set_name = rename.get(set_key, set_key)
                for metric_name, values in set_metrics.items():
                    metrics[f"{set_name}_{metric_name}"] = float(values[i])
            self.tracker.log_iteration(i, metrics)

            if i % log_every == 0 or i == n_iters - 1:
                metric_str = " | ".join(f"{k}={v:.5f}" for k, v in metrics.items())
                self.train_logger.info(f"Iter {i:>5} | {metric_str}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model nie wytrenowany.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model nie wytrenowany.")
        return self.model.predict_proba(X)

    # ──────── Pomocnicze ────────
    def get_booster(self):
        """Zwraca obiekt xgboost.Booster — przydatne dla TreeSHAP."""
        if not self.is_fitted:
            raise RuntimeError("Model nie wytrenowany.")
        return self.model.get_booster()
