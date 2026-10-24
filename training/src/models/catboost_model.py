"""
CatBoost — model komplementarny w ensemble.

CatBoost wybrany jako model bazowy z trzech powodów:
    1. Silna regularyzacja — dobra generalizacja na małych klasach
       (ESI 1, ESI 5).
    2. Inny inductive bias niż XGB/LGBM — Symmetric Trees (oblivious
       decision trees) wprowadzają regularyzację strukturalną, co daje
       mniejszą korelację błędów z XGB/LGBM w ensemble.
    3. Działa na CPU i GPU — urządzenie wybiera `hardware.device_for("catboost")`
       (SOR_AI_DEVICE_CATBOOST / SOR_AI_DEVICE = auto | cpu | gpu).

Na GPU: `gpu_ram_part` = config.CATBOOST_GPU_RAM_PART, bez `rsm`/`colsample_bylevel`
(na GPU tylko dla trybów pairwise), MVS → Bernoulli (MVS nie działa z MultiClass na GPU).
Błąd CUDA (sterownik, brak pamięci) → ponowny trening na CPU (baza).

Wątki: budżet modelu (`thread_count` = config.N_JOBS) trafia także do budowy Pool
i do predict_proba — domyślne -1 CatBoost zajęłoby wszystkie wątki logiczne (zderzenie
z drugim torem orkiestratora).

Logowanie (tylko fit()): krzywe `train_MultiClass` i `es_MultiClass` → JSON + plik .log.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from src.models.base import BaseTriageModel
from src.utils.config import CATBOOST_DEFAULT_PARAMS, CATBOOST_GPU_RAM_PART
from src.utils.hardware import device_for
from src.utils.logger import get_logger

log = get_logger(__name__)


# Parametry, które są aliasami sklearn-style → CatBoost-style.
# Używane gdy parametry przychodzą z Optuny (random_state) lub
# z train.py (n_jobs).
_SKLEARN_TO_CATBOOST_ALIASES: dict[str, str] = {
    "n_jobs": "thread_count",
    "random_state": "random_seed",
}

# Parametry istniejące tylko na GPU
_GPU_ONLY_PARAMS: tuple[str, ...] = ("devices", "gpu_ram_part")


def _translate_aliases(params: dict[str, Any]) -> dict[str, Any]:
    """sklearn-style → CatBoost-style (gdy podano oba, wygrywa nazwa CatBoost)."""
    out = dict(params)
    for sk_name, cb_name in _SKLEARN_TO_CATBOOST_ALIASES.items():
        if sk_name in out:
            value = out.pop(sk_name)
            out.setdefault(cb_name, value)
    return out


def _normalize_catboost_params(params: dict[str, Any]) -> dict[str, Any]:
    """Tłumaczy sklearn-style nazwy na CatBoost-style i usuwa konflikty."""
    out = _translate_aliases(params)

    # Walidacja: subsample wymaga Bernoulli/MVS/Poisson, NIE Bayesian.
    bootstrap = out.get("bootstrap_type")
    if bootstrap == "Bayesian":
        out.pop("subsample", None)
    elif bootstrap in ("Bernoulli", "MVS", "Poisson"):
        out.pop("bagging_temperature", None)

    # Ograniczenia GPU
    if out.get("task_type") == "GPU":
        # rsm (alias colsample_bylevel) na GPU tylko dla trybów pairwise
        out.pop("rsm", None)
        out.pop("colsample_bylevel", None)
        if out.get("bootstrap_type") == "MVS":
            out["bootstrap_type"] = "Bernoulli"
    else:
        for key in _GPU_ONLY_PARAMS:
            out.pop(key, None)

    # max_leaves dostępne tylko z Lossguide.
    if out.get("grow_policy") != "Lossguide":
        out.pop("max_leaves", None)

    # min_data_in_leaf nie działa z SymmetricTree.
    if out.get("grow_policy") == "SymmetricTree":
        out.pop("min_data_in_leaf", None)

    return out


class CatBoostTriageModel(BaseTriageModel):
    """Model CatBoost dla 5-klasowej klasyfikacji ESI (CPU lub GPU)."""

    name = "catboost"
    display_name = "CatBoost"
    supports_early_stopping = True
    iteration_param = "iterations"

    def __init__(self, params: dict[str, Any] | None = None):
        # Aliasy użytkownika (np. random_state z Optuny) nadpisują domyślne random_seed/thread_count
        merged = {**CATBOOST_DEFAULT_PARAMS, **_translate_aliases(params or {})}
        if "task_type" not in merged:
            merged["task_type"] = "GPU" if device_for("catboost") == "gpu" else "CPU"
        merged["task_type"] = str(merged["task_type"]).upper()
        if merged["task_type"] == "GPU":
            merged.setdefault("devices", "0")
            merged.setdefault("gpu_ram_part", CATBOOST_GPU_RAM_PART)
        super().__init__(params=_normalize_catboost_params(merged))

    # ──────── Urządzenie ────────
    def _uses_gpu(self) -> bool:
        return str(self.params.get("task_type", "CPU")).upper() == "GPU"

    def _switch_to_cpu(self) -> None:
        self.params = _normalize_catboost_params({**self.params, "task_type": "CPU"})

    def _thread_count(self) -> int:
        """Budżet wątków modelu (`thread_count`, ≤ liczba CPU tej maszyny); brak / ≤ 0 → -1 (wszystkie)."""
        try:
            value = int(self.params.get("thread_count") or -1)
        except (TypeError, ValueError):
            return -1
        if value <= 0:
            return -1
        return min(value, os.cpu_count() or value)

    # ──────── Trening ────────
    def _build_model(self):
        from catboost import CatBoostClassifier

        params = _normalize_catboost_params(self.params)
        params["verbose"] = self._verbose_period      # 0 = cicho (fit_estimator)
        return CatBoostClassifier(**params)

    def _fit_impl(self, X, y, X_es, y_es, sample_weight) -> None:
        from catboost import Pool

        self._label_classes = np.unique(y)
        self.model = self._build_model()
        # Pool — natywny format CatBoost (szybszy niż przekazywanie DataFrame)
        threads = self._thread_count()
        train_pool = Pool(data=X, label=y, weight=sample_weight, feature_names=list(X.columns),
                          thread_count=threads)
        fit_kwargs: dict[str, Any] = {}
        if X_es is not None:
            fit_kwargs.update(
                eval_set=Pool(data=X_es, label=y_es, feature_names=list(X.columns), thread_count=threads),
                early_stopping_rounds=self.es_rounds or int(self.params.get("od_wait", 200)),
                use_best_model=True,
            )
        self.model.fit(train_pool, **fit_kwargs)

    def _after_fit(self, has_es: bool) -> None:
        # classes_count=5 daje zawsze 5 kolumn, ale klasa nieobecna w treningu dostaje
        # przypadkowe małe prawdopodobieństwo — zostawiamy tylko klasy z treningu (reszta = 0)
        self.classes_ = np.asarray(self._label_classes).astype(int)
        self.actual_device = "gpu" if self._uses_gpu() else "cpu"
        self.iteration_cap = int(self.params.get("iterations") or 1000)
        best = self.model.get_best_iteration() if has_es else None   # 0-based
        self.best_iteration = int(best) + 1 if best is not None else None

    def _predict_proba_raw(self, X):
        proba = np.asarray(self.model.predict_proba(X, thread_count=self._thread_count()))
        model_classes = np.asarray(self.model.classes_).astype(int)
        return proba[:, np.searchsorted(model_classes, self.classes_)]

    def _eval_history(self) -> list[dict[str, float]]:
        """{'learn': {...}, 'validation': {...}} → [{'train_MultiClass', 'es_MultiClass'}, ...]."""
        evals_result = self.model.get_evals_result() if hasattr(self.model, "get_evals_result") else {}
        if not evals_result:
            return []
        rename = {"learn": "train", "validation": "es"}
        first = next(iter(evals_result.values()))
        n_iters = len(next(iter(first.values())))
        history = []
        for i in range(n_iters):
            row = {}
            for set_key, metrics in evals_result.items():
                set_name = rename.get(set_key, set_key)
                for metric, values in metrics.items():
                    if i < len(values):
                        row[f"{set_name}_{metric}"] = float(values[i])
            history.append(row)
        return history

    def _log_extra_details(self) -> None:
        tree_count = getattr(self.model, "tree_count_", None)
        if tree_count is not None:
            self.train_logger.info(f"Tree count (final): {tree_count}")
            self.tracker.log_metrics({"tree_count": int(tree_count)})
        self.train_logger.info(
            f"Wątki CPU: {self.params.get('thread_count')}, task_type: {self.params.get('task_type')}"
        )

    # ──────── Pomocnicze ────────
    def get_booster(self):
        """Zwraca wewnętrzny obiekt CatBoost — przydatne dla SHAP."""
        if not self.is_fitted:
            raise RuntimeError("Model nie wytrenowany.")
        return self.model
