"""
LightGBM — gradient boosting z liśćmi rosnącymi "leaf-wise".

Inny sposób budowy drzew niż XGBoost (depth-wise) i CatBoost (drzewa symetryczne),
więc błędy są częściowo nieskorelowane — przydatne przy łączeniu modeli.
Zwykle 5–10× szybszy od XGBoost na CPU. Tylko CPU (koło pip nie ma CUDA, a OpenCL
przy 5 klasach i kilkuset cechach nie jest szybszy od 24 rdzeni).

Logowanie (tylko fit()): krzywa early stopping `es_multi_logloss` → JSON + plik .log.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any

import pandas as pd

from src.models.base import BaseTriageModel
from src.utils.config import LGBM_DEFAULT_PARAMS, LGBM_EARLY_STOPPING_ROUNDS
from src.utils.logger import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _supports_eval_xy() -> bool:
    """LightGBM >= 4.7: `eval_X`/`eval_y` zamiast przestarzałego `eval_set` (4.0–4.6 tylko eval_set)."""
    import lightgbm as lgb

    return "eval_X" in inspect.signature(lgb.LGBMClassifier.fit).parameters


def lgbm_eval_kwargs(X_es: pd.DataFrame, y_es: Any, name: str = "es") -> dict[str, Any]:
    """Argumenty fit() ze zbiorem ewaluacyjnym, zgodne z LightGBM 4.0–4.7+ (bez ostrzeżeń)."""
    if _supports_eval_xy():
        return {"eval_X": X_es, "eval_y": y_es, "eval_names": [name]}
    return {"eval_set": [(X_es, y_es)], "eval_names": [name]}


class LightGBMTriageModel(BaseTriageModel):
    """Model LightGBM dla 5-klasowej klasyfikacji ESI."""

    name = "lightgbm"
    display_name = "LightGBM"
    supports_early_stopping = True
    iteration_param = "n_estimators"

    def __init__(self, params: dict[str, Any] | None = None):
        merged = {**LGBM_DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged)

    def _build_model(self):
        import lightgbm as lgb

        params = dict(self.params)
        # Wagi zawsze przez sample_weight (jedno źródło prawdy)
        params.pop("class_weight", None)
        return lgb.LGBMClassifier(**params)

    def _fit_impl(self, X, y, X_es, y_es, sample_weight) -> None:
        import lightgbm as lgb

        self.model = self._build_model()
        # log_evaluation(0) = cicho; w fit() postęp co `verbose` iteracji
        callbacks: list = [lgb.log_evaluation(self._verbose_period)]
        fit_kwargs: dict[str, Any] = {"sample_weight": sample_weight, "callbacks": callbacks}
        if X_es is not None:
            rounds = self.es_rounds or LGBM_EARLY_STOPPING_ROUNDS
            callbacks.insert(0, lgb.early_stopping(rounds, first_metric_only=True, verbose=False))
            fit_kwargs.update(lgbm_eval_kwargs(X_es, y_es))
        self.model.fit(X, y, **fit_kwargs)

    def _after_fit(self, has_es: bool) -> None:
        super()._after_fit(has_es)
        self.iteration_cap = int(self.params.get("n_estimators") or 100)
        best = int(getattr(self.model, "best_iteration_", 0) or 0)   # 1-based; 0 = bez ES
        self.best_iteration = best if has_es and best > 0 else None

    def _eval_history(self) -> list[dict[str, float]]:
        """{'es': {'multi_logloss': [...]}} → [{'es_multi_logloss': ...}, ...]."""
        evals_result = getattr(self.model, "evals_result_", None) or {}
        if not evals_result:
            return []
        first = next(iter(evals_result.values()))
        n_iters = len(next(iter(first.values())))
        return [
            {f"{set_name}_{metric}": float(values[i])
             for set_name, metrics in evals_result.items()
             for metric, values in metrics.items()}
            for i in range(n_iters)
        ]
