"""
Klasa bazowa dla wszystkich modeli triażowych.

Wszystkie modele dziedziczą po `BaseTriageModel`, co zapewnia spójny interfejs:
    - fit_estimator(X, y, X_es, y_es, sample_weight)  — cichy trening samego estymatora
      (tuning, CV): bez logu treningu i JSON-a eksperymentu
    - fit(X_train, y_train, X_val, y_val, ...)        — trening z pełnym trackingiem;
      samo dopasowanie estymatora deleguje do fit_estimator()
    - predict(X) / predict_proba(X)                    — zawsze (n, 5), float64, kolumny ESI 1–5
    - save(path) / load(path), `src.models.load_model(path)`

fit_estimator() wspólnie dla wszystkich modeli:
    - early stopping na (X_es, y_es), jeśli model go wspiera (XGBoost / LightGBM / CatBoost),
    - błąd GPU (CUDA, brak pamięci karty) → jedna ponowna próba na CPU z ostrzeżeniem,
    - ustawia actual_device, device_fallback, best_iteration, iteration_cap
      (+ WARNING, gdy najlepsza iteracja ≥ 95% limitu).

fit() dodatkowo:
    1. Tworzy dedykowany logger treningu (logs/training/<model>/<run_id>.log)
    2. Inicjuje ExperimentTracker (logs/experiments/<model>_<run_id>.json)
    3. Loguje hiperparametry, dane, urządzenie, krzywą early stopping (es_*), czas
    4. Zapisuje JSON eksperymentu i zamyka plik logu
"""

from __future__ import annotations

import abc
import gc
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from src.utils import config as _config
from src.utils.config import (
    CLASS_NAMES,
    CUSTOM_CLASS_WEIGHTS,
    DEFAULT_SAMPLE_WEIGHT_STRATEGY,
    MODELS_DIR,
    N_CLASSES,
)
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.hardware import is_gpu_error
from src.utils.logger import close_training_logger, get_logger, get_training_logger

log = get_logger(__name__)

# Najlepsza iteracja ≥ tej części limitu → model prawdopodobnie niedotrenowany
ITERATION_CAP_WARN_FRACTION: float = 0.95
# Kompresja joblib (zlib 3): modele drzewiaste kurczą się kilkukrotnie, odczyt nadal szybki
JOBLIB_COMPRESS: int = 3


# ─────────────────────────────────────────
# Helpery — wagi klas/sampli
# ─────────────────────────────────────────
def compute_sample_weights(
    y: np.ndarray | pd.Series,
    strategy: str = DEFAULT_SAMPLE_WEIGHT_STRATEGY,
    class_weights: dict[int, float] | None = None,
) -> np.ndarray:
    """
    Wagi sampli dla treningu.

    Parameters
    ----------
    y : target labels (0–4)
    strategy : str
        'none'          — brak wag (domyślne; skalibrowane prawdopodobieństwa)
        'balanced'      — n / (k · n_c)
        'sqrt_balanced' — pierwiastek z 'balanced' (łagodniejsze wyrównanie)
        'custom'        — CUSTOM_CLASS_WEIGHTS z config.py
        (ignorowane, gdy podano class_weights)
    class_weights : dict, optional
        Mapa {klasa: waga} z Optuny — nadpisuje strategy.

    Returns
    -------
    np.ndarray długości len(y).
    """
    y = np.asarray(y)

    # Priorytet: class_weights > strategy
    if class_weights is not None:
        return np.array([class_weights.get(int(yi), 1.0) for yi in y])

    if strategy == "balanced":
        return compute_sample_weight("balanced", y)

    if strategy == "sqrt_balanced":
        return np.sqrt(compute_sample_weight("balanced", y))

    if strategy == "custom":
        return np.array([CUSTOM_CLASS_WEIGHTS.get(int(yi), 1.0) for yi in y])

    if strategy == "none":
        return np.ones(len(y))

    raise ValueError(f"Nieznana strategia wag: {strategy}")


def compute_class_weights_dict(
    y: np.ndarray | pd.Series,
    strategy: str = DEFAULT_SAMPLE_WEIGHT_STRATEGY,
) -> dict[int, float]:
    """Wagi klas jako dict {class_id: waga} (ta sama logika co compute_sample_weights)."""
    y = np.asarray(y)
    classes = np.unique(y)
    weights = compute_sample_weights(y, strategy=strategy)
    return {int(c): float(weights[y == c][0]) for c in classes}


def full_class_proba(proba: np.ndarray, classes: np.ndarray | list | None) -> np.ndarray:
    """
    Prawdopodobieństwa estymatora → macierz (n, 5) float64, kolumny = klasy 0..4 (ESI 1–5).

    Klasa nieobecna w treningu dostaje 0; wiersze renormalizowane do sumy 1
    (wiersz zerowy → rozkład jednostajny).
    """
    raw = np.asarray(proba, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    cols = np.arange(raw.shape[1]) if classes is None else np.asarray(classes).astype(int).ravel()
    if len(cols) != raw.shape[1]:
        raise RuntimeError(
            f"Estymator zwrócił {raw.shape[1]} kolumn prawdopodobieństw, a zna klasy {cols.tolist()}"
        )
    if raw.shape[1] == N_CLASSES and np.array_equal(cols, np.arange(N_CLASSES)):
        full = raw.copy()
    else:
        full = np.zeros((raw.shape[0], N_CLASSES), dtype=np.float64)
        full[:, cols] = raw
    np.clip(full, 0.0, None, out=full)
    sums = full.sum(axis=1, keepdims=True)
    return np.divide(full, sums, out=np.full_like(full, 1.0 / N_CLASSES), where=sums > 0)


# ─────────────────────────────────────────
# Klasa bazowa
# ─────────────────────────────────────────
class BaseTriageModel(abc.ABC):
    """
    Abstrakcyjna klasa bazowa dla modeli triażowych.

    Podklasa implementuje `_fit_impl` (zbudowanie i dopasowanie `self.model`)
    i `_after_fit` (klasy, urządzenie, najlepsza iteracja); resztę daje baza.
    """

    name: str = "base"
    display_name: str = "model"            # w logach: "Trenowanie <display_name> …"
    supports_early_stopping: bool = False   # czy fit_estimator używa (X_es, y_es)
    iteration_param: str | None = None      # nazwa parametru limitu iteracji (do komunikatów)

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = dict(params or {})
        self.model: Any = None
        self.feature_names: list[str] | None = None
        self.feature_set: str | None = None
        self.classes_: np.ndarray | None = None
        self.is_fitted: bool = False

        # Wagi klas z Optuny (nadpisują strategy w compute_sample_weights)
        self.optuna_class_weights: dict[int, float] | None = None

        # Urządzenie i przebieg treningu — wypełniane w fit_estimator()
        self.requested_device: str = "gpu" if self._uses_gpu() else "cpu"
        self.actual_device: str | None = None
        self.device_fallback: str | None = None
        self.best_iteration: int | None = None
        self.iteration_cap: int | None = None
        self.es_rounds: int | None = None      # nadpisanie liczby rund early stopping
        self.data_version: str | None = None   # z artefaktu (load)
        self.hardware: dict | None = None       # opis maszyny (save / load)
        self._verbose_period: int = 0          # postęp biblioteki na stdout (tylko fit())

        # Tracking — wypełniane w fit()
        self.run_id: str | None = None
        self.train_logger = None
        self.train_log_path: Path | None = None
        self.tracker: ExperimentTracker | None = None
        self.training_duration_s: float | None = None

    # ──────── Hooki podklas ────────
    @abc.abstractmethod
    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_es: pd.DataFrame | None,
        y_es: np.ndarray | None,
        sample_weight: np.ndarray | None,
    ) -> None:
        """Buduje `self.model` i dopasowuje go (y = etykiety 0–4 jako int)."""

    def _after_fit(self, has_es: bool) -> None:
        """Po treningu: klasy, faktyczne urządzenie, najlepsza iteracja i limit."""
        self.classes_ = np.asarray(self.model.classes_).astype(int)
        self.actual_device = "cpu"

    def _uses_gpu(self) -> bool:
        """Czy bieżące parametry każą trenować na GPU."""
        return False

    def _switch_to_cpu(self) -> None:
        """Przełącza parametry na CPU (po błędzie GPU)."""
        raise NotImplementedError(f"{self.name}: brak trybu GPU")

    def _predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Surowe prawdopodobieństwa estymatora (kolumny w kolejności `self.classes_`)."""
        return self.model.predict_proba(X)

    def _eval_history(self) -> list[dict[str, float]]:
        """Krzywa early stopping per iteracja, klucze `es_<metryka>` (puste, gdy brak)."""
        return []

    def _log_extra_details(self) -> None:
        """Dodatkowe informacje modelu do logu treningu / trackera (np. liczba drzew)."""

    # ──────── Cichy trening estymatora ────────
    def fit_estimator(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        X_es: pd.DataFrame | None = None,
        y_es: np.ndarray | pd.Series | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """
        Trenuje sam estymator — bez logu treningu i JSON-a eksperymentu (tuning, CV).

        - (X_es, y_es): early stopping, jeśli model je wspiera (RF / EBM / logreg ignorują);
          wiersze ES z klasą nieobecną w treningu są pomijane,
        - błąd GPU (`hardware.is_gpu_error`) → jedna ponowna próba na CPU, WARNING
          i powód w `self.device_fallback`,
        - ustawia model, feature_names, classes_, is_fitted, actual_device, device_fallback,
          best_iteration, iteration_cap, training_duration_s.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("fit_estimator: X musi być pandas.DataFrame (nazwy cech są częścią modelu)")
        y_arr = np.asarray(y).astype(int).ravel()
        if len(y_arr) != len(X):
            raise ValueError(f"fit_estimator: len(y)={len(y_arr)} ≠ len(X)={len(X)}")
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).ravel()

        X_es_use, y_es_use = None, None
        if self.supports_early_stopping:
            if X_es is not None and y_es is not None and len(X_es) > 0:
                y_es_arr = np.asarray(y_es).astype(int).ravel()
                keep = np.isin(y_es_arr, np.unique(y_arr))
                if keep.all():
                    X_es_use, y_es_use = X_es[list(X.columns)], y_es_arr
                elif keep.any():
                    X_es_use, y_es_use = X_es[list(X.columns)].loc[keep], y_es_arr[keep]
            if X_es_use is None:
                log.warning(f"{self.name}: brak zbioru early stopping — trening pełnego limitu iteracji "
                            f"({self.iteration_param}={self.params.get(self.iteration_param)})")

        missing = sorted(set(range(N_CLASSES)) - set(np.unique(y_arr).tolist()))
        if missing:
            log.warning(f"{self.name}: w danych treningowych brak klas {missing} "
                        f"(ESI {[c + 1 for c in missing]}) — dostaną prawdopodobieństwo 0")

        self.feature_names = list(X.columns)
        self.model = None
        self.is_fitted = False
        self.device_fallback = None
        self.best_iteration = None
        self.iteration_cap = None

        t0 = time.time()
        try:
            self._fit_impl(X, y_arr, X_es_use, y_es_use, sw)
        except Exception as exc:
            if not (self._uses_gpu() and is_gpu_error(exc)):
                raise
            reason = f"{type(exc).__name__}: {str(exc).strip()[:500]}"
            log.warning(f"{self.name}: błąd GPU — PONAWIAM TRENING NA CPU ({reason})")
            self.model = None
            gc.collect()                      # zwolnij pamięć karty przed ponowną próbą
            self._switch_to_cpu()
            self._fit_impl(X, y_arr, X_es_use, y_es_use, sw)
            self.device_fallback = reason
        self.training_duration_s = time.time() - t0

        self._after_fit(has_es=X_es_use is not None)
        self.is_fitted = True

        if (
            self.best_iteration is not None and self.iteration_cap
            and self.best_iteration >= ITERATION_CAP_WARN_FRACTION * self.iteration_cap
        ):
            log.warning(
                f"{self.name}: najlepsza iteracja {self.best_iteration} ≥ "
                f"{ITERATION_CAP_WARN_FRACTION:.0%} limitu {self.iteration_cap} — model może być "
                f"niedotrenowany; zwiększ {self.iteration_param} albo learning_rate"
            )

    # ──────── Setup trackingu ────────
    def _setup_tracking(
        self,
        run_id: str | None = None,
        use_mlflow: bool = False,
    ) -> None:
        """Inicjuje logger treningu i tracker eksperymentu."""
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.train_logger, self.train_log_path = get_training_logger(
            model_name=self.name,
            run_id=self.run_id,
        )
        self.tracker = ExperimentTracker(
            model_name=self.name,
            run_id=self.run_id,
            use_mlflow=use_mlflow,
        )
        self.tracker.log_params(self.params)
        self.tracker.log_artifact("training_log", self.train_log_path)

        self.train_logger.info(f"Model:    {self.name}")
        self.train_logger.info(f"Params:   {self.params}")
        self.train_logger.info(f"Urządzenie (żądane): {self.requested_device}")

    def _log_data_info(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        feature_set: str = "triage",
    ) -> None:
        """Loguje statystyki danych treningowych."""
        self.feature_set = feature_set
        if self.tracker is None:
            return
        self.tracker.log_data_info(X_train, y_train, X_val, y_val, feature_set=feature_set)

        if self.train_logger:
            self.train_logger.info(f"Dane treningowe:  n={len(X_train):,}, features={X_train.shape[1]}")
            if X_val is not None:
                self.train_logger.info(f"Dane early stopping: n={len(X_val):,}")

            # Rozkład klas
            train_dist = pd.Series(np.asarray(y_train)).value_counts(normalize=True).sort_index()
            self.train_logger.info(f"Rozkład klas (train): {dict(train_dist.round(4))}")

    def _log_training_details(
        self,
        X_es: pd.DataFrame | None,
        y_es: np.ndarray | pd.Series | None,
        log_every: int,
    ) -> None:
        """Krzywa early stopping (es_*), najlepsza iteracja, urządzenie → log + tracker."""
        history = self._eval_history()
        for i, metrics in enumerate(history):
            self.tracker.log_iteration(i, metrics)
            if i % log_every == 0 or i == len(history) - 1:
                metric_str = " | ".join(f"{k}={v:.5f}" for k, v in metrics.items())
                self.train_logger.info(f"Iter {i:>5} | {metric_str}")

        if self.best_iteration is not None:
            self.train_logger.info(f"Najlepsza iteracja: {self.best_iteration} (limit {self.iteration_cap})")
            self.tracker.add_note(f"Early stopping: najlepsza iteracja {self.best_iteration} "
                                  f"z limitu {self.iteration_cap}")
            self.tracker.log_metrics({"best_iteration": self.best_iteration,
                                      "iteration_cap": self.iteration_cap})
            if self.iteration_cap and self.best_iteration >= ITERATION_CAP_WARN_FRACTION * self.iteration_cap:
                self.train_logger.warning(
                    f"Najlepsza iteracja ≥ {ITERATION_CAP_WARN_FRACTION:.0%} limitu — model może być "
                    f"niedotrenowany (zwiększ {self.iteration_param})"
                )
                self.tracker.add_note("UWAGA: najlepsza iteracja przy limicie iteracji")

        if not self.supports_early_stopping and X_es is not None and y_es is not None and len(X_es):
            es_acc = float(np.mean(self.predict(X_es) == np.asarray(y_es).astype(int)))
            self.train_logger.info(f"Accuracy (zbiór ES): {es_acc:.4f}")
            self.tracker.log_metrics({"es_accuracy": es_acc})

        self._log_extra_details()

        self.train_logger.info(f"Urządzenie (faktyczne): {self.actual_device}")
        if self.device_fallback:
            self.train_logger.warning(f"Zmiana urządzenia GPU → CPU: {self.device_fallback}")
            self.tracker.add_note(f"GPU → CPU: {self.device_fallback}")
        self.tracker.log_device(self.requested_device, self.actual_device, self.device_fallback)

    def _finalize_tracking(self, save_artifacts: bool = True) -> None:
        """Zapisuje JSON eksperymentu + feature importances."""
        if self.tracker is None:
            return

        # Feature importances
        importances = self.feature_importances()
        if importances is not None:
            self.tracker.log_feature_importances(importances, top_n=30)

        if self.train_logger:
            self.train_logger.info(f"Trening zakończony w {self.training_duration_s:.1f} s")
            if importances is not None:
                top5 = importances.head(5)
                self.train_logger.info("Top 5 cech (importance):")
                for _, row in top5.iterrows():
                    self.train_logger.info(f"  {row['feature']:30s} {row['importance']:.4f}")

        if save_artifacts:
            json_path = self.tracker.save()
            if self.train_logger:
                self.train_logger.info(f"Zapisano metadane eksperymentu: {json_path}")

    # ──────── Interfejs publiczny ────────
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
        early_stopping_rounds: int | None = None,
        **kwargs,
    ) -> "BaseTriageModel":
        """
        Trenuje model z pełnym logowaniem. Zwraca self.

        X_val / y_val służą WYŁĄCZNIE do early stopping (w pipeline treningu
        to wydzielona część TRAIN — zbiór `val` zostaje nietknięty).

        Parameters
        ----------
        sample_weight_strategy : 'none' | 'balanced' | 'sqrt_balanced' | 'custom'
            (wagi z Optuny w `optuna_class_weights` mają pierwszeństwo)
        run_id : identyfikator eksperymentu (timestamp; auto, jeśli None)
        use_mlflow : czy logować równolegle do MLflow
        feature_set : etykieta zestawu cech (do metadanych)
        verbose : co ile iteracji biblioteka wypisuje postęp na stdout (0/False = cicho)
        early_stopping_rounds : nadpisanie liczby rund bez poprawy (modele z early stopping)
        """
        if early_stopping_rounds is not None:
            self.es_rounds = int(early_stopping_rounds)
        log_every = verbose if isinstance(verbose, int) and not isinstance(verbose, bool) and verbose > 0 else 100

        # 1. Setup trackingu (logger, JSON eksperymentu)
        self._setup_tracking(run_id=run_id, use_mlflow=use_mlflow)
        try:
            self._log_data_info(X_train, y_train, X_val, y_val, feature_set=feature_set)
            self.train_logger.info(
                f"Trenowanie {self.display_name} (n={len(X_train):,}, features={X_train.shape[1]})"
            )

            # 2. Wagi sampli (Optuna > strategia)
            sample_weights = compute_sample_weights(
                y_train, strategy=sample_weight_strategy, class_weights=self.optuna_class_weights
            )
            self.train_logger.info(
                f"Wagi sampli: strategia='{sample_weight_strategy}', "
                f"min={sample_weights.min():.2f}, max={sample_weights.max():.2f}"
            )

            # 3. Trening estymatora (postęp biblioteki na stdout co `verbose` iteracji)
            self._verbose_period = log_every if verbose else 0
            try:
                self.fit_estimator(X_train, y_train, X_val, y_val, sample_weight=sample_weights)
            finally:
                self._verbose_period = 0

            # 4. Historia, urządzenie, finalizacja
            self._log_training_details(X_val, y_val, log_every=log_every)
            self.train_logger.info(f"Trening {self.display_name} zakończony w {self.training_duration_s:.1f}s")
            self._finalize_tracking()
        except Exception as exc:
            self.train_logger.exception(f"Błąd treningu {self.display_name}: {exc}")
            raise
        finally:
            close_training_logger(self.train_logger)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prawdopodobieństwa ESI 1–5: float64, shape (n, 5), wiersze sumują się do 1,
        kolumny w kolejności klas 0..4 (klasa nieobecna w treningu → 0).
        Kolumny X ustawiane w kolejności `feature_names` z treningu.
        """
        if not self.is_fitted:
            raise RuntimeError("Model nie wytrenowany.")
        return full_class_proba(self._predict_proba_raw(self._align_features(X)), self.classes_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Klasa 0–4 o największym prawdopodobieństwie (spójna z predict_proba)."""
        return self.predict_proba(X).argmax(axis=1)

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Kolumny w kolejności z treningu; brak cechy → KeyError z listą braków."""
        if not isinstance(X, pd.DataFrame) or self.feature_names is None:
            return X
        if list(X.columns) == self.feature_names:
            return X
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            more = " …" if len(missing) > 10 else ""
            raise KeyError(f"Brak {len(missing)} cech wymaganych przez model {self.name}: {missing[:10]}{more}")
        return X[self.feature_names]

    # ──────── Cechy dodatkowe ────────
    def feature_importances(self) -> pd.DataFrame | None:
        """
        DataFrame z ważnościami cech (jeśli model je udostępnia).

        Returns
        -------
        pd.DataFrame z kolumnami: feature, importance
        """
        if self.model is None:
            return None

        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return None

        names = self.feature_names or [f"f_{i}" for i in range(len(importances))]
        df = pd.DataFrame({"feature": names, "importance": importances})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    # ──────── Persystencja ────────
    def save(self, path: Path | str | None = None) -> Path:
        """
        Zapisuje model na dysk (joblib, kompresja zlib; zapis atomowy przez plik .tmp).

        Artefakt zawiera estymator i metadane: cechy, klasy, wersję danych, faktyczne
        urządzenie, ewentualny powód przejścia GPU → CPU, najlepszą iterację, opis sprzętu.
        """
        if not self.is_fitted:
            raise RuntimeError("Model nie został wytrenowany — nie ma czego zapisać.")

        from src.utils.hardware import hardware_summary

        if path is None:
            suffix = f"_{self.run_id}" if self.run_id else ""
            path = MODELS_DIR / f"{self.name}{suffix}.joblib"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.hardware = hardware_summary()
        except Exception as e:  # noqa: BLE001 — opis sprzętu nie może zablokować zapisu
            self.hardware = {"error": f"{type(e).__name__}: {e}"}

        artifact = {
            "name": self.name,
            "run_id": self.run_id,
            "params": self.params,
            "model": self.model,
            "feature_names": self.feature_names,
            "feature_set": self.feature_set,
            "classes_": self.classes_,
            "class_names": CLASS_NAMES,
            "label": "esi",                # indeks 0–4 = ESI 1–5
            "data_version": _config.DATA_VERSION,
            "training_duration_s": self.training_duration_s,
            "requested_device": self.requested_device,
            "actual_device": self.actual_device,
            "device_fallback": self.device_fallback,
            "best_iteration": self.best_iteration,
            "iteration_cap": self.iteration_cap,
            "hardware": self.hardware,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            joblib.dump(artifact, tmp, compress=JOBLIB_COMPRESS)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

        size_mb = path.stat().st_size / 1e6
        log.info(f"Zapisano model do: {path} ({size_mb:.2f} MB)")

        if self.tracker is not None:
            self.tracker.log_artifact("model_file", path)

        return path

    @classmethod
    def load(cls, path: Path | str) -> "BaseTriageModel":
        """
        Ładuje model z dysku. Wywołane na `BaseTriageModel` wybiera klasę po nazwie
        modelu w artefakcie (jak `src.models.load_model`); na podklasie sprawdza zgodność.
        Nie uruchamia wyboru urządzenia ani prób GPU.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku modelu: {path}")
        artifact = joblib.load(path)
        instance = cls._from_artifact(artifact, path)
        log.info(f"Załadowano model z: {path}")
        return instance

    @classmethod
    def _from_artifact(cls, artifact: Any, path: Path | str = "?") -> "BaseTriageModel":
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ValueError(f"{path} nie jest artefaktem modelu SOR-AI (brak słownika z kluczem 'model')")
        name = str(artifact.get("name"))
        if cls is BaseTriageModel:
            from src.models import MODEL_REGISTRY, resolve_model_name

            target: type[BaseTriageModel] = MODEL_REGISTRY[resolve_model_name(name)]
        elif name != cls.name:
            raise ValueError(f"{path} zawiera model '{name}', a nie '{cls.name}' — użyj src.models.load_model()")
        else:
            target = cls

        # Bez __init__ podklasy: parametry dokładnie jak w artefakcie, bez wyboru urządzenia
        instance = target.__new__(target)
        BaseTriageModel.__init__(instance, params=artifact.get("params"))
        instance.model = artifact["model"]
        instance.feature_names = artifact.get("feature_names")
        instance.feature_set = artifact.get("feature_set")
        classes = artifact.get("classes_")
        instance.classes_ = None if classes is None else np.asarray(classes).astype(int)
        instance.run_id = artifact.get("run_id")
        instance.training_duration_s = artifact.get("training_duration_s")
        instance.data_version = artifact.get("data_version")
        instance.requested_device = artifact.get("requested_device", instance.requested_device)
        instance.actual_device = artifact.get("actual_device")
        instance.device_fallback = artifact.get("device_fallback")
        instance.best_iteration = artifact.get("best_iteration")
        instance.iteration_cap = artifact.get("iteration_cap")
        instance.hardware = artifact.get("hardware")
        instance.is_fitted = True
        return instance

    # ──────── Repr ────────
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        run = f", run='{self.run_id}'" if self.run_id else ""
        return f"{self.__class__.__name__}(name='{self.name}', {status}{run})"

    # ──────── Pomiar czasu ────────
    def _start_timer(self) -> float:
        return time.time()

    def _stop_timer(self, t0: float) -> float:
        self.training_duration_s = time.time() - t0
        return self.training_duration_s
