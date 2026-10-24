"""
XGBoost — gradient boosting (drzewa poziomowe), główny model tabelaryczny.

XGBoost wybrany jako model bazowy z dwóch powodów:
    1. Doskonała wydajność na danych tabularnych (szczególnie z brakami).
    2. TreeSHAP — najszybsza i najdokładniejsza metoda wyjaśnień.

Urządzenie: `hardware.device_for("xgboost")` → device="cuda" | "cpu".
    - XGBoost bez widocznego GPU NIE rzuca błędu, tylko po cichu trenuje na CPU — faktyczne
      urządzenie odczytywane jest z konfiguracji boostera (`actual_device`, `device_fallback`),
    - błąd CUDA w trakcie treningu (np. brak pamięci) → ponowny trening na CPU (baza),
    - po treningu model przełączany na device="cpu": predykcja na danych z RAM bez
      kopiowania na kartę ("mismatched devices"), a zapisany model działa bez GPU.

Logowanie (tylko fit()): krzywa early stopping `es_mlogloss` → JSON + plik .log.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.models.base import BaseTriageModel
from src.utils.config import XGB_DEFAULT_PARAMS
from src.utils.hardware import device_for
from src.utils.logger import get_logger

log = get_logger(__name__)


class XGBoostTriageModel(BaseTriageModel):
    """Model XGBoost dla 5-klasowej klasyfikacji ESI."""

    name = "xgboost"
    display_name = "XGBoost"
    supports_early_stopping = True
    iteration_param = "n_estimators"

    def __init__(self, params: dict[str, Any] | None = None):
        merged = {**XGB_DEFAULT_PARAMS, **(params or {})}
        if "device" not in merged:
            merged["device"] = "cuda" if device_for("xgboost") == "gpu" else "cpu"
        super().__init__(params=merged)

    # ──────── Urządzenie ────────
    def _uses_gpu(self) -> bool:
        return str(self.params.get("device", "cpu")).lower().startswith(("cuda", "gpu"))

    def _switch_to_cpu(self) -> None:
        self.params["device"] = "cpu"

    # ──────── Trening ────────
    def _build_model(self, with_eval_set: bool = True):
        import xgboost as xgb

        params = dict(self.params)
        # num_class ustala XGBClassifier z etykiet; jawne 5 przy brakującej klasie psuje
        # booster po set_params (base_score o złej długości)
        params.pop("num_class", None)
        if self.es_rounds is not None:
            params["early_stopping_rounds"] = self.es_rounds
        if not with_eval_set:
            # Early stopping wymaga zbioru ewaluacyjnego
            params.pop("early_stopping_rounds", None)
        return xgb.XGBClassifier(**params)

    def _fit_impl(self, X, y, X_es, y_es, sample_weight) -> None:
        # XGBoost wymaga etykiet 0..k-1 — przy brakującej klasie (np. mała próbka bez ESI 1)
        # kodujemy obecne klasy kolejno, a mapę trzyma self.classes_.
        self._label_classes = np.unique(y)
        has_es = X_es is not None
        self.model = self._build_model(with_eval_set=has_es)
        fit_kwargs: dict[str, Any] = {
            "sample_weight": sample_weight,
            "verbose": self._verbose_period or False,
        }
        if has_es:
            fit_kwargs["eval_set"] = [(X_es, np.searchsorted(self._label_classes, y_es))]
        self.model.fit(X, np.searchsorted(self._label_classes, y), **fit_kwargs)

    def _after_fit(self, has_es: bool) -> None:
        self.classes_ = np.asarray(self._label_classes).astype(int)

        config = json.loads(self.model.get_booster().save_config())
        trained_on = str(config["learner"]["generic_param"].get("device", "cpu")).lower()
        self.actual_device = "cuda" if trained_on.startswith("cuda") else "cpu"
        if self._uses_gpu() and self.actual_device == "cpu" and self.device_fallback is None:
            self.device_fallback = (
                "XGBoost: zażądano device='cuda', ale booster trenował na CPU (brak widocznego GPU; "
                "koło xgboost >= 3.4 z CUDA 13 wymaga sterownika NVIDIA >= 580)"
            )
            log.warning(f"{self.device_fallback} — trening był wolniejszy, wynik jest poprawny")

        # Predykcja zawsze na CPU (dane w RAM) i przenośny artefakt
        self.model.set_params(device="cpu")

        self.iteration_cap = int(self.model.get_params().get("n_estimators") or 100)
        self.best_iteration = None
        if has_es:
            try:
                self.best_iteration = int(self.model.best_iteration) + 1   # liczba drzew w modelu
            except AttributeError:
                self.best_iteration = None

    def _eval_history(self) -> list[dict[str, float]]:
        """{'validation_0': {'mlogloss': [...]}} → [{'es_mlogloss': ...}, ...]."""
        evals_result = getattr(self.model, "evals_result_", None) or {}
        if not evals_result:
            return []
        sets = list(evals_result.items())
        names = ["es"] if len(sets) == 1 else [f"es{i}" for i in range(len(sets))]
        n_iters = len(next(iter(sets[0][1].values())))
        history = []
        for i in range(n_iters):
            row = {}
            for set_name, (_, set_metrics) in zip(names, sets):
                for metric_name, values in set_metrics.items():
                    row[f"{set_name}_{metric_name}"] = float(values[i])
            history.append(row)
        return history

    # ──────── Pomocnicze ────────
    def get_booster(self):
        """Zwraca obiekt xgboost.Booster — przydatne dla TreeSHAP."""
        if not self.is_fitted:
            raise RuntimeError("Model nie wytrenowany.")
        return self.model.get_booster()
