"""
Tracker eksperymentów — zapisuje pełne metadane treningu do JSON.

Każdy run treningu generuje:
    logs/experiments/<model>_<run_id>.json
        ├── run_id, model_name, timestamp_start/end, duration_seconds
        ├── git_commit, hostname, python_version
        ├── data: {n_train, n_val, n_test, feature_count, class_distribution}
        ├── params (hiperparametry modelu)
        ├── training_history (per-iteration metrics: mlogloss, merror)
        ├── metrics: {val_*, test_*}  — pełna ewaluacja
        ├── feature_importances (top-N)
        └── artifacts: {model_path, log_path, config_path}

Dodatkowo opcjonalnie loguje do MLflow (jeśli włączone).

Użycie:
    tracker = ExperimentTracker(model_name="xgboost", run_id="...")
    tracker.log_params({"max_depth": 8, ...})
    tracker.log_data_info(X_train, y_train, X_val, y_val)
    tracker.log_iteration(iteration=100, metrics={"val_mlogloss": 0.45})
    tracker.log_metrics({"val_qwk": 0.87, "test_undertriage": 0.02})
    tracker.log_artifact("model_path", "/models/xgb.joblib")
    tracker.save()
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import config as _config
from src.utils.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI


# ─────────────────────────────────────────
# Helpery
# ─────────────────────────────────────────
def _git_commit() -> str | None:
    """Krótki SHA aktualnego commita (jeśli to repo git)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _to_jsonable(obj: Any) -> Any:
    """Rekursywnie konwertuje obiekty do JSON-serializowalnych typów."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


# ─────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────
class ExperimentTracker:
    """Zapisuje metadane pojedynczego eksperymentu (run) do JSON + MLflow."""

    def __init__(
        self,
        model_name: str,
        run_id: str | None = None,
        output_dir: Path | str | None = None,
        use_mlflow: bool = False,
        mlflow_experiment: str = MLFLOW_EXPERIMENT_NAME,
    ):
        self.model_name = model_name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        # UWAGA: Pobieramy EXPERIMENTS_DIR przez `_config` (lazy lookup),
        # by monkeypatch w testach działał poprawnie.
        self.output_dir = Path(output_dir or _config.EXPERIMENTS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.json_path = self.output_dir / f"{model_name}_{self.run_id}.json"
        self.start_time = datetime.now()

        # Główny słownik z metadanymi
        self.data: dict[str, Any] = {
            "run_id": self.run_id,
            "model_name": model_name,
            "timestamp_start": self.start_time.isoformat(),
            "timestamp_end": None,
            "duration_seconds": None,
            "environment": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python_version": sys.version.split()[0],
                "git_commit": _git_commit(),
            },
            "params": {},
            "data": {},
            "training_history": [],
            "metrics": {},
            "feature_importances": [],
            "artifacts": {},
            "notes": [],
        }

        # MLflow (opcjonalnie)
        self.use_mlflow = use_mlflow
        self.mlflow_run = None
        if use_mlflow:
            self._init_mlflow(mlflow_experiment)

    # ────────────── MLflow ──────────────
    def _init_mlflow(self, experiment_name: str) -> None:
        try:
            import mlflow

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(experiment_name)
            self.mlflow_run = mlflow.start_run(run_name=f"{self.model_name}_{self.run_id}")
        except ImportError:
            self.use_mlflow = False
            self.data["notes"].append("MLflow nieinstalowany — pomijam tracking.")

    # ────────────── Logowanie ──────────────
    def log_params(self, params: dict[str, Any]) -> None:
        """Hiperparametry modelu."""
        self.data["params"].update(_to_jsonable(params))
        if self.use_mlflow:
            try:
                import mlflow
                for k, v in params.items():
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        mlflow.log_param(k, v)
            except Exception:
                pass

    def log_data_info(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        X_test: pd.DataFrame | None = None,
        y_test: np.ndarray | pd.Series | None = None,
        feature_set: str = "triage_only",
    ) -> None:
        """Statystyki danych treningowych."""
        info = {
            "feature_set": feature_set,
            "n_features": X_train.shape[1],
            "feature_names": list(X_train.columns),
            "n_train": int(len(X_train)),
            "class_distribution_train": _to_jsonable(
                pd.Series(y_train).value_counts(normalize=True).sort_index().to_dict()
            ),
        }
        if X_val is not None:
            info["n_val"] = int(len(X_val))
            info["class_distribution_val"] = _to_jsonable(
                pd.Series(y_val).value_counts(normalize=True).sort_index().to_dict()
            )
        if X_test is not None:
            info["n_test"] = int(len(X_test))
            info["class_distribution_test"] = _to_jsonable(
                pd.Series(y_test).value_counts(normalize=True).sort_index().to_dict()
            )
        self.data["data"].update(info)

    def log_iteration(
        self,
        iteration: int,
        metrics: dict[str, float],
    ) -> None:
        """Metryki per iteracja (np. mlogloss/merror per round XGBoost)."""
        entry = {"iteration": iteration, **_to_jsonable(metrics)}
        self.data["training_history"].append(entry)
        if self.use_mlflow:
            try:
                import mlflow
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                        mlflow.log_metric(k, v, step=iteration)
            except Exception:
                pass

    def log_metrics(self, metrics: dict[str, Any], prefix: str = "") -> None:
        """Końcowe metryki (np. {val_qwk: 0.87, test_undertriage: 0.02})."""
        if prefix:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
        self.data["metrics"].update(_to_jsonable(metrics))
        if self.use_mlflow:
            try:
                import mlflow
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                        mlflow.log_metric(k, v)
            except Exception:
                pass

    def log_feature_importances(
        self,
        importances: pd.DataFrame,
        top_n: int = 30,
    ) -> None:
        """Top-N najważniejszych cech."""
        if importances is None or importances.empty:
            return
        top = importances.head(top_n).to_dict(orient="records")
        self.data["feature_importances"] = _to_jsonable(top)

    def log_artifact(self, name: str, path: Path | str) -> None:
        """Zarejestruj ścieżkę do artefaktu (model, log, config)."""
        self.data["artifacts"][name] = str(Path(path).resolve())
        if self.use_mlflow and Path(path).exists() and Path(path).is_file():
            try:
                import mlflow
                mlflow.log_artifact(str(path))
            except Exception:
                pass

    def add_note(self, note: str) -> None:
        """Dopisz notatkę tekstową (np. 'early stopping na iter 234')."""
        self.data["notes"].append(note)

    # ────────────── Persystencja ──────────────
    def save(self) -> Path:
        """Zapisuje JSON + zamyka MLflow."""
        end = datetime.now()
        self.data["timestamp_end"] = end.isoformat()
        self.data["duration_seconds"] = (end - self.start_time).total_seconds()

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False, default=_to_jsonable)

        if self.use_mlflow:
            try:
                import mlflow
                mlflow.log_artifact(str(self.json_path))
                mlflow.end_run()
            except Exception:
                pass

        return self.json_path

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.add_note(f"Błąd: {exc_type.__name__}: {exc_val}")
        self.save()


# ─────────────────────────────────────────
# Pomocnicze: callbacki dla XGBoost / LightGBM
# ─────────────────────────────────────────
class XGBoostTrackingCallback:
    """Callback XGBoost zapisujący każdą iterację do trackera + loggera."""

    def __init__(
        self,
        tracker: ExperimentTracker,
        logger=None,
        log_every: int = 10,
    ):
        self.tracker = tracker
        self.logger = logger
        self.log_every = log_every

    def __call__(self, env) -> None:
        """env to obiekt CallbackEnv (legacy XGBoost API)."""
        iteration = env.iteration
        metrics_at_iter = {}
        for item in env.evaluation_result_list:
            # item to (eval_set_name, metric_name, value, std) lub (set, metric, value)
            if len(item) >= 3:
                set_name, metric_name, value = item[0], item[1], item[2]
                key = f"{set_name}_{metric_name}"
                metrics_at_iter[key] = float(value)

        self.tracker.log_iteration(iteration, metrics_at_iter)

        if self.logger and (iteration % self.log_every == 0):
            metric_str = " | ".join(f"{k}={v:.5f}" for k, v in metrics_at_iter.items())
            self.logger.info(f"Iter {iteration:>5} | {metric_str}")


def lightgbm_tracking_callback(
    tracker: ExperimentTracker,
    logger=None,
    log_every: int = 10,
):
    """
    Callback LightGBM zapisujący metryki per iteracja.

    Użycie:
        model.fit(..., callbacks=[lightgbm_tracking_callback(tracker, log)])
    """
    def _callback(env) -> None:
        iteration = env.iteration
        metrics_at_iter = {}
        for item in env.evaluation_result_list:
            # (set_name, metric_name, value, is_higher_better)
            if len(item) >= 3:
                set_name, metric_name, value = item[0], item[1], item[2]
                key = f"{set_name}_{metric_name}"
                metrics_at_iter[key] = float(value)

        tracker.log_iteration(iteration, metrics_at_iter)

        if logger and (iteration % log_every == 0):
            metric_str = " | ".join(f"{k}={v:.5f}" for k, v in metrics_at_iter.items())
            logger.info(f"Iter {iteration:>5} | {metric_str}")

    return _callback
