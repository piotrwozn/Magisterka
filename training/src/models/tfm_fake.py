"""
Atrapa estymatora TFM do testów (`SOR_AI_TFM_FAKE=1`) — bez torch, bez wag, deterministyczna.

Wrappery TabPFN / TabICL przy aktywnej atrapie przechodzą DOKŁADNIE tę samą ścieżkę kodu co z prawdziwą
biblioteką (worki kontekstu, porcje testu, mapowanie `classes_` na 5 kolumn ESI, korekta priorów,
ustawienia procesu, drabina awaryjna przy braku pamięci) — zamieniony jest tylko sam estymator:
    regresja logistyczna (wieloklasowa) na cechach standaryzowanych, braki → mediana kontekstu.

Symulacja braku pamięci GPU (OOM) — polityka z `FAKE_OOM` (testy: monkeypatch) albo ze zmiennej
`SOR_AI_TFM_FAKE_OOM` (JSON; działa także w podprocesach):
    {"predict_rows_gt": 2048}                 predict z więcej niż 2048 wierszami → OOM,
    {"predict_when": {"kv_cache": "kv"}}      predict estymatora z tymi opcjami → OOM,
    {"fit_when": {"keep_cache_on_device": true}}  fit estymatora z tymi opcjami → OOM,
    {"always": true}                          każdy predict → OOM (drabina kończy się błędem).
Klucze można łączyć; `predict_when` / `fit_when` mogą być listą słowników (dowolny pasujący → OOM).

`FAKE_SETTINGS` udaje procesowe ustawienia TabPFN (`tabpfn.settings.settings.tabpfn`) — wrapper TabPFN
zmienia je i przywraca tak samo jak prawdziwe. `FAKE_CALLS` liczy wywołania (testy: jeden fit na worek).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from types import SimpleNamespace
from typing import Any

import numpy as np

FAKE_OOM_ENV = "SOR_AI_TFM_FAKE_OOM"
# Polityka OOM ustawiana w testach (None → zmienna środowiskowa SOR_AI_TFM_FAKE_OOM)
FAKE_OOM: dict[str, Any] | None = None
# Udawane ustawienia procesu TabPFN (te same nazwy pól co tabpfn.settings.settings.tabpfn)
FAKE_SETTINGS = SimpleNamespace(tabpfn=SimpleNamespace(
    max_batched_test_rows=32_768, max_batched_estimator_rows=None, max_batched_estimator_cells=None,
))
# Licznik wywołań: "fit", "predict", "predict_rows"
FAKE_CALLS: Counter = Counter()


class FakeOutOfMemoryError(RuntimeError):
    """Symulowany brak pamięci GPU (nazwa klasy zawiera 'OutOfMemory', jak TabPFNOutOfMemoryError)."""


def oom_policy() -> dict[str, Any]:
    """Aktywna polityka symulacji OOM: `FAKE_OOM` albo JSON z `SOR_AI_TFM_FAKE_OOM` (pusta = brak OOM)."""
    if FAKE_OOM is not None:
        return dict(FAKE_OOM)
    raw = os.environ.get(FAKE_OOM_ENV, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{FAKE_OOM_ENV}={raw!r} nie jest poprawnym JSON-em: {exc}") from None
    return value if isinstance(value, dict) else {}


def _matches(options: dict[str, Any], rules: Any) -> bool:
    """Czy opcje estymatora pasują do reguły (słownik) albo którejkolwiek z listy reguł."""
    if not rules:
        return False
    for rule in rules if isinstance(rules, list) else [rules]:
        if isinstance(rule, dict) and rule and all(options.get(k) == v for k, v in rule.items()):
            return True
    return False


class FakeTFMClassifier:
    """
    Estymator w stylu scikit-learn udający TabPFNClassifier / TabICLClassifier.

    options : opcje, z którymi wrapper zbudowałby prawdziwy estymator (kv_cache, batch_size, fit_mode…);
        służą symulacji OOM i testom (`est.options`).
    """

    def __init__(self, *, library: str, options: dict[str, Any] | None = None, random_state: int = 0):
        self.library = library
        self.options = dict(options or {})
        self.random_state = int(random_state)
        self.classes_: np.ndarray | None = None
        self.n_train_samples_: int | None = None
        self._model: Any = None
        self._constant: np.ndarray | None = None

    def _oom(self, where: str, detail: str) -> FakeOutOfMemoryError:
        return FakeOutOfMemoryError(f"CUDA out of memory (symulacja atrapy TFM, {where}: {detail})")

    def fit(self, X: np.ndarray, y: np.ndarray) -> FakeTFMClassifier:
        from sklearn.linear_model import LogisticRegression

        policy = oom_policy()
        if _matches(self.options, policy.get("fit_when")):
            raise self._oom("fit", f"opcje {self.options}")
        FAKE_CALLS["fit"] += 1
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y)
        self.n_train_samples_ = int(len(y))
        with np.errstate(all="ignore"):
            med = np.nanmedian(X, axis=0) if len(X) else np.zeros(X.shape[1])
        self._median = np.where(np.isfinite(med), med, 0.0)
        Z = self._impute(X)
        self._mean = Z.mean(axis=0) if len(Z) else np.zeros(Z.shape[1])
        std = Z.std(axis=0) if len(Z) else np.ones(Z.shape[1])
        self._std = np.where(std > 1e-12, std, 1.0)
        if len(self.classes_) < 2:
            self._constant = np.ones(1)
            return self
        self._model = LogisticRegression(C=1.0, max_iter=300, random_state=self.random_state)
        self._model.fit((Z - self._mean) / self._std, y)
        return self

    def _impute(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64, copy=True)
        rows, cols = np.where(~np.isfinite(X))
        X[rows, cols] = self._median[cols]
        return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("FakeTFMClassifier: fit() przed predict_proba()")
        X = np.asarray(X, dtype=np.float64)
        policy = oom_policy()
        if policy.get("always"):
            raise self._oom("predict", "always")
        limit = policy.get("predict_rows_gt")
        if limit is not None and len(X) > int(limit):
            raise self._oom("predict", f"{len(X)} wierszy > {int(limit)}")
        if _matches(self.options, policy.get("predict_when")):
            raise self._oom("predict", f"opcje {self.options}")
        FAKE_CALLS["predict"] += 1
        FAKE_CALLS["predict_rows"] += len(X)
        if self._constant is not None:
            return np.ones((len(X), 1), dtype=np.float32)
        Z = (self._impute(X) - self._mean) / self._std
        return self._model.predict_proba(Z).astype(np.float32)   # float32 jak TabICL


def reset_calls() -> None:
    FAKE_CALLS.clear()


__all__ = ["FAKE_CALLS", "FAKE_OOM", "FAKE_OOM_ENV", "FAKE_SETTINGS", "FakeOutOfMemoryError",
           "FakeTFMClassifier", "oom_policy", "reset_calls"]
