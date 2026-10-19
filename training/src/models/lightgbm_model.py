"""
LightGBM — gradient boosting z liśćmi rosnącymi "leaf-wise".

Inny sposób budowy drzew niż XGBoost (depth-wise) i CatBoost (drzewa symetryczne),
więc błędy są częściowo nieskorelowane — przydatne przy łączeniu modeli.
Zwykle 5–10× szybszy od XGBoost na CPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseTriageModel, compute_sample_weights
from src.utils.config import (
    DEFAULT_SAMPLE_WEIGHT_STRATEGY,
    LGBM_DEFAULT_PARAMS,
    LGBM_EARLY_STOPPING_ROUNDS,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


class LightGBMTriageModel(BaseTriageModel):
    """Model LightGBM dla 5-klasowej klasyfikacji ESI."""

    name = "lightgbm"

    def __init__(self, params: dict[str, Any] | None = None):
        merged = {**LGBM_DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged)

    def _build_model(self):
        import lightgbm as lgb

        params = dict(self.params)
        # Wagi zawsze przez sample_weight (jedno źródło prawdy)
        params.pop("class_weight", None)
        return lgb.LGBMClassifier(**params)

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
        verbose: int | bool = 100,
        early_stopping_rounds: int = LGBM_EARLY_STOPPING_ROUNDS,
        **kwargs,
    ) -> "LightGBMTriageModel":
        """Trenuje LightGBM z pełnym logowaniem."""
        import lightgbm as lgb

        # 1. Tracking setup
        self._setup_tracking(run_id=run_id, use_mlflow=use_mlflow)
        self._log_data_info(X_train, y_train, X_val, y_val, feature_set=feature_set)

        self.train_logger.info(f"Trenowanie LightGBM (n={len(X_train):,}, features={X_train.shape[1]})")
        self.feature_names = list(X_train.columns)

        # 2. Wagi sampli (Optuna > strategia)
        sample_weights = compute_sample_weights(
            y_train, strategy=sample_weight_strategy, class_weights=self.optuna_class_weights
        )
        self.train_logger.info(
            f"Wagi sampli: strategia='{sample_weight_strategy}', "
            f"min={sample_weights.min():.2f}, max={sample_weights.max():.2f}"
        )

        # 3. Eval set — tylko zbiór early stopping
        has_val = X_val is not None and y_val is not None
        eval_set = [(X_val, y_val)] if has_val else None
        eval_names = ["val"] if has_val else None

        # 4. Callbacki — własny per-iteration logger
        log_every = verbose if isinstance(verbose, int) and verbose > 0 else 100
        callbacks: list = [
            lgb.log_evaluation(0),  # wyłącz domyślny output
            self._make_iteration_callback(log_every=log_every),
        ]
        if has_val:
            callbacks.insert(0, lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False))

        # 5. Trenuj
        self.model = self._build_model()
        t0 = self._start_timer()
        fit_kwargs: dict[str, Any] = {"sample_weight": sample_weights, "callbacks": callbacks}
        if has_val:
            fit_kwargs.update(eval_set=eval_set, eval_names=eval_names)
        self.model.fit(X_train, y_train, **fit_kwargs)
        duration = self._stop_timer(t0)

        self.classes_ = self.model.classes_
        self.is_fitted = True

        best_iter = getattr(self.model, "best_iteration_", None)
        if best_iter:
            self.train_logger.info(f"Best iteration: {best_iter}")
            self.tracker.add_note(f"Early stopping na iter {best_iter}")
            self.tracker.log_metrics({"best_iteration": best_iter})

        self.train_logger.info(f"Trening LightGBM zakończony w {duration:.1f}s")

        # 6. Finalizuj
        self._finalize_tracking()

        return self

    def _make_iteration_callback(self, log_every: int = 100):
        """Callback per-iteration zapisujący metryki do trackera."""
        tracker = self.tracker
        train_logger = self.train_logger

        def _callback(env) -> None:
            iteration = env.iteration
            metrics = {}
            for item in env.evaluation_result_list:
                # item: (set_name, metric_name, value, is_higher_better)
                metrics[f"{item[0]}_{item[1]}"] = float(item[2])

            tracker.log_iteration(iteration, metrics)

            if iteration % log_every == 0:
                metric_str = " | ".join(f"{k}={v:.5f}" for k, v in metrics.items())
                train_logger.info(f"Iter {iteration:>5} | {metric_str}")

        return _callback

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model nie wytrenowany.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model nie wytrenowany.")
        return self.model.predict_proba(X)
