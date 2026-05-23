"""
Klasa bazowa dla wszystkich modeli triażowych.

Wszystkie modele dziedziczą po `BaseTriageModel`, co zapewnia spójny interfejs:
    - fit(X_train, y_train, X_val, y_val)
    - predict(X)
    - predict_proba(X)
    - save(path) / load(path)

UWAGA: Każdy fit() automatycznie:
    1. Tworzy dedykowany logger treningu (logs/training/<model>/<run_id>.log)
    2. Inicjuje ExperimentTracker (logs/experiments/<model>_<run_id>.json)
    3. Loguje hiperparametry, statystyki danych, metryki per-iteracja
    4. Zapisuje pełen artefakt JSON po treningu
"""

from __future__ import annotations

import abc
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from src.utils.config import CUSTOM_CLASS_WEIGHTS, MODELS_DIR
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.logger import get_logger, get_training_logger

log = get_logger(__name__)


# ─────────────────────────────────────────
# Helpery — wagi klas/sampli
# ─────────────────────────────────────────
def compute_sample_weights(
    y: np.ndarray | pd.Series,
    strategy: str = "custom",
    class_weights: dict[int, float] | None = None,
) -> np.ndarray:
    """
    Wagi sampli dla treningu.

    Parameters
    ----------
    y : target labels
    strategy : str
        'balanced' | 'custom' | 'none' (ignorowane gdy class_weights podane)
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

    if strategy == "custom":
        return np.array([CUSTOM_CLASS_WEIGHTS.get(int(yi), 1.0) for yi in y])

    if strategy == "none":
        return np.ones(len(y))

    raise ValueError(f"Nieznana strategia wag: {strategy}")


def compute_class_weights_dict(
    y: np.ndarray | pd.Series,
    strategy: str = "custom",
) -> dict[int, float]:
    """Wagi klas jako dict {class_id: waga}."""
    y = np.asarray(y)
    classes = np.unique(y)

    if strategy == "balanced":
        weights = compute_class_weight("balanced", classes=classes, y=y)
        return dict(zip(classes.tolist(), weights.tolist()))

    if strategy == "custom":
        return {int(c): CUSTOM_CLASS_WEIGHTS.get(int(c), 1.0) for c in classes}

    if strategy == "none":
        return {int(c): 1.0 for c in classes}

    raise ValueError(f"Nieznana strategia wag: {strategy}")


# ─────────────────────────────────────────
# Klasa bazowa
# ─────────────────────────────────────────
class BaseTriageModel(abc.ABC):
    """
    Abstrakcyjna klasa bazowa dla modeli triażowych.

    Zapewnia infrastrukturę logowania i trackingu eksperymentów,
    którą reusują wszystkie konkretne modele.
    """

    name: str = "base"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = dict(params or {})
        self.model: Any = None
        self.feature_names: list[str] | None = None
        self.classes_: np.ndarray | None = None
        self.is_fitted: bool = False

        # Wagi klas z Optuny (nadpisują strategy w compute_sample_weights)
        self.optuna_class_weights: dict[int, float] | None = None

        # Tracking — wypełniane w fit()
        self.run_id: str | None = None
        self.train_logger = None
        self.train_log_path: Path | None = None
        self.tracker: ExperimentTracker | None = None
        self.training_duration_s: float | None = None

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

    def _log_data_info(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        feature_set: str = "triage_only",
    ) -> None:
        """Loguje statystyki danych treningowych."""
        if self.tracker is None:
            return
        self.tracker.log_data_info(X_train, y_train, X_val, y_val, feature_set=feature_set)

        if self.train_logger:
            self.train_logger.info(f"Dane treningowe:  n={len(X_train):,}, features={X_train.shape[1]}")
            if X_val is not None:
                self.train_logger.info(f"Dane walidacyjne: n={len(X_val):,}")

            # Rozkład klas
            train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
            self.train_logger.info(f"Rozkład klas (train): {dict(train_dist.round(4))}")

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
    @abc.abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        sample_weight_strategy: str = "custom",
        run_id: str | None = None,
        use_mlflow: bool = False,
        **kwargs,
    ) -> "BaseTriageModel":
        """Trenuje model. Zwraca self."""

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predykcja klas."""

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Prawdopodobieństwa per klasa, shape (n, 5)."""

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
        """Zapisuje model na dysk (joblib)."""
        if not self.is_fitted:
            raise RuntimeError("Model nie został wytrenowany — nie ma czego zapisać.")

        if path is None:
            suffix = f"_{self.run_id}" if self.run_id else ""
            path = MODELS_DIR / f"{self.name}{suffix}.joblib"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "name": self.name,
            "run_id": self.run_id,
            "params": self.params,
            "model": self.model,
            "feature_names": self.feature_names,
            "classes_": self.classes_,
            "training_duration_s": self.training_duration_s,
        }
        joblib.dump(artifact, path)

        size_mb = path.stat().st_size / 1e6
        log.info(f"Zapisano model do: {path} ({size_mb:.2f} MB)")

        if self.tracker is not None:
            self.tracker.log_artifact("model_file", path)

        return path

    @classmethod
    def load(cls, path: Path | str) -> "BaseTriageModel":
        """Ładuje model z dysku."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku modelu: {path}")

        artifact = joblib.load(path)
        instance = cls(params=artifact.get("params"))
        instance.model = artifact["model"]
        instance.feature_names = artifact.get("feature_names")
        instance.classes_ = artifact.get("classes_")
        instance.run_id = artifact.get("run_id")
        instance.training_duration_s = artifact.get("training_duration_s")
        instance.is_fitted = True
        log.info(f"Załadowano model z: {path}")
        return instance

    # ──────── Repr ────────
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        run = f", run='{self.run_id}'" if self.run_id else ""
        return f"{self.__class__.__name__}(name='{self.name}', {status}{run})"

    # ──────── Context manager dla pomiaru czasu ────────
    def _start_timer(self) -> float:
        return time.time()

    def _stop_timer(self, t0: float) -> float:
        self.training_duration_s = time.time() - t0
        return self.training_duration_s
