"""
CatBoost — model komplementarny w ensemble.

CatBoost wybrany jako 5. model bazowy z trzech powodów:
    1. Ordered boosting — redukuje target leakage, lepsza generalizacja
       na małych klasach (Red, Yellow w naszym przypadku).
    2. Inny inductive bias niż XGB/LGBM — Symmetric Trees (oblivious
       decision trees) wprowadzają regularyzację strukturalną, co daje
       mniejszą korelację błędów z XGB/LGBM w ensemble.
    3. Świetna wydajność CPU multi-threaded — `thread_count=-1` skaluje
       liniowo do ~16 wątków, potem saturacja.

Cały trening jest w pełni logowany (per-iteration mlogloss → JSON tracker).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseTriageModel, compute_sample_weights
from src.utils.config import CATBOOST_DEFAULT_PARAMS
from src.utils.logger import get_logger

log = get_logger(__name__)


# Parametry, które są aliasami sklearn-style → CatBoost-style.
# Używane gdy parametry przychodzą z Optuny (random_state) lub
# z train.py (n_jobs).
_SKLEARN_TO_CATBOOST_ALIASES: dict[str, str] = {
    "n_jobs": "thread_count",
    "random_state": "random_seed",
}


def _normalize_catboost_params(params: dict[str, Any]) -> dict[str, Any]:
    """Tłumaczy sklearn-style nazwy na CatBoost-style i usuwa konflikty."""
    out = dict(params)
    for sk_name, cb_name in _SKLEARN_TO_CATBOOST_ALIASES.items():
        if sk_name in out and cb_name not in out:
            out[cb_name] = out.pop(sk_name)
        elif sk_name in out:
            # Jeśli oba ustawione, preferuj CatBoost-style i odrzuć alias.
            out.pop(sk_name, None)

    # Walidacja: subsample wymaga Bernoulli/MVS/Poisson, NIE Bayesian.
    bootstrap = out.get("bootstrap_type")
    if bootstrap == "Bayesian":
        out.pop("subsample", None)
    elif bootstrap in ("Bernoulli", "MVS", "Poisson"):
        out.pop("bagging_temperature", None)

    # GPU restrictions
    if out.get("task_type") == "GPU":
        out.pop("rsm", None)
        if out.get("bootstrap_type") == "MVS":
            out["bootstrap_type"] = "Bernoulli"

    # max_leaves dostępne tylko z Lossguide.
    if out.get("grow_policy") != "Lossguide":
        out.pop("max_leaves", None)

    # min_data_in_leaf nie działa z SymmetricTree.
    if out.get("grow_policy") == "SymmetricTree":
        out.pop("min_data_in_leaf", None)

    return out


class CatBoostTriageModel(BaseTriageModel):
    """Model CatBoost dla 5-klasowej klasyfikacji triażu MTS (CPU multi-threaded)."""

    name = "catboost"

    def __init__(self, params: dict[str, Any] | None = None):
        merged = {**CATBOOST_DEFAULT_PARAMS, **(params or {})}
        merged = _normalize_catboost_params(merged)
        super().__init__(params=merged)

    def _build_model(self):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**_normalize_catboost_params(self.params))

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
        early_stopping_rounds: int = 100,
        **kwargs,
    ) -> "CatBoostTriageModel":
        """Trenuje CatBoost z pełnym logowaniem."""
        from catboost import CatBoostClassifier, Pool

        # 1. Tracking setup
        self._setup_tracking(run_id=run_id, use_mlflow=use_mlflow)
        self._log_data_info(X_train, y_train, X_val, y_val, feature_set=feature_set)

        self.train_logger.info(
            f"Trenowanie CatBoost (n={len(X_train):,}, features={X_train.shape[1]}, "
            f"threads={self.params.get('thread_count', -1)})"
        )
        self.feature_names = list(X_train.columns)

        # 2. Wagi sampli
        sample_weights = compute_sample_weights(y_train, strategy=sample_weight_strategy, class_weights=self.optuna_class_weights)
        self.train_logger.info(
            f"Wagi sampli: strategia='{sample_weight_strategy}', "
            f"min={sample_weights.min():.2f}, max={sample_weights.max():.2f}"
        )

        # 3. Pool-e (natywny format CatBoost — szybszy niż przekazywanie DataFrame)
        train_pool = Pool(
            data=X_train,
            label=np.asarray(y_train),
            weight=sample_weights,
            feature_names=self.feature_names,
        )

        eval_pool = None
        if X_val is not None and y_val is not None:
            eval_pool = Pool(
                data=X_val,
                label=np.asarray(y_val),
                feature_names=self.feature_names,
            )

        # 4. Trenuj — bez własnego log_every (CatBoost loguje natywnie via verbose)
        self.model = self._build_model()

        # Podmień verbose w runtime gdy chcemy ciszej (np. podczas Optuny)
        fit_kwargs = {
            "eval_set": eval_pool,
            "verbose_eval": verbose if isinstance(verbose, int) and verbose > 0 else False,
            "early_stopping_rounds": early_stopping_rounds if eval_pool is not None else None,
            "use_best_model": eval_pool is not None,
        }
        # CatBoost nie akceptuje None dla early_stopping_rounds
        fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

        t0 = self._start_timer()
        self.model.fit(train_pool, **fit_kwargs)
        duration = self._stop_timer(t0)

        # 5. Wyciągnij training history → tracker
        self._extract_evals_result(verbose=verbose)

        self.classes_ = self.model.classes_
        self.is_fitted = True

        best_iter = getattr(self.model, "best_iteration_", None)
        if best_iter is not None:
            self.train_logger.info(f"Best iteration: {best_iter}")
            self.tracker.add_note(f"Early stopping na iter {best_iter}")
            self.tracker.log_metrics({"best_iteration": best_iter})

        tree_count = self.model.tree_count_ if hasattr(self.model, "tree_count_") else None
        if tree_count is not None:
            self.train_logger.info(f"Tree count (final): {tree_count}")
            self.tracker.log_metrics({"tree_count": tree_count})

        self.train_logger.info(f"Trening CatBoost zakończony w {duration:.1f}s")

        # 6. Finalizuj
        self._finalize_tracking()

        return self

    def _extract_evals_result(self, verbose: int | bool = 100) -> None:
        """
        Wyciąga `evals_result_` z CatBoost i zapisuje do trackera + loggera.

        CatBoost format:
            {"learn": {"MultiClass": [...]},
             "validation": {"MultiClass": [...]}}
        """
        if not hasattr(self.model, "get_evals_result"):
            return
        evals_result = self.model.get_evals_result()
        if not evals_result:
            return

        eval_keys = list(evals_result.keys())
        rename = {}
        if len(eval_keys) >= 1:
            rename[eval_keys[0]] = "train"
        if len(eval_keys) >= 2:
            rename[eval_keys[1]] = "val"

        first_key = eval_keys[0]
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
        # CatBoost zwraca shape (n,1) dla multiclass — flatten
        preds = self.model.predict(X)
        return np.asarray(preds).ravel().astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model nie wytrenowany.")
        return self.model.predict_proba(X)

    # ──────── Pomocnicze ────────
    def get_booster(self):
        """Zwraca wewnętrzny obiekt CatBoost — przydatne dla SHAP."""
        if not self.is_fitted:
            raise RuntimeError("Model nie wytrenowany.")
        return self.model
