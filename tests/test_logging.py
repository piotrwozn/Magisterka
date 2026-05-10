"""
Testy systemu logowania treningu i trackingu eksperymentów.

Sprawdza, że każdy fit() modelu:
    1. Tworzy plik logu w logs/training/<model>/<run_id>.log
    2. Tworzy JSON eksperymentu w logs/experiments/<model>_<run_id>.json
    3. Loguje hiperparametry, statystyki danych, czas treningu
    4. Zapisuje training_history (dla XGBoost/LightGBM)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestExperimentTracker:
    def test_basic_save(self, tmp_path):
        from src.utils.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(
            model_name="testmodel",
            run_id="test_basic",
            output_dir=tmp_path,
        )
        tracker.log_params({"lr": 0.1, "depth": 5})
        tracker.log_metrics({"qwk": 0.85})
        tracker.log_iteration(0, {"loss": 1.0})
        tracker.log_iteration(1, {"loss": 0.5})
        tracker.add_note("test note")

        path = tracker.save()
        assert path.exists()

        with open(path) as f:
            data = json.load(f)

        assert data["model_name"] == "testmodel"
        assert data["run_id"] == "test_basic"
        assert data["params"] == {"lr": 0.1, "depth": 5}
        assert data["metrics"]["qwk"] == 0.85
        assert len(data["training_history"]) == 2
        assert "test note" in data["notes"]
        assert data["duration_seconds"] >= 0

    def test_environment_metadata(self, tmp_path):
        from src.utils.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker("test", "envtest", output_dir=tmp_path)
        path = tracker.save()
        with open(path) as f:
            data = json.load(f)

        env = data["environment"]
        assert "hostname" in env
        assert "platform" in env
        assert "python_version" in env

    def test_log_data_info(self, tmp_path):
        from src.utils.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker("test", "datatest", output_dir=tmp_path)
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
        y = pd.Series([0, 1, 2, 0])

        tracker.log_data_info(X, y, feature_set="custom")
        path = tracker.save()

        with open(path) as f:
            data = json.load(f)

        assert data["data"]["n_features"] == 2
        assert data["data"]["n_train"] == 4
        assert data["data"]["feature_set"] == "custom"
        assert "class_distribution_train" in data["data"]

    def test_context_manager(self, tmp_path):
        from src.utils.experiment_tracker import ExperimentTracker

        json_path = None
        with ExperimentTracker("test", "ctxtest", output_dir=tmp_path) as tracker:
            tracker.log_params({"x": 1})
            json_path = tracker.json_path

        assert json_path.exists()


class TestTrainingLogger:
    def test_creates_file(self, tmp_path):
        from src.utils.logger import get_training_logger

        logger, log_path = get_training_logger(
            model_name="test_logger",
            run_id="logtest_001",
            log_dir=tmp_path,
        )

        logger.info("test message")
        logger.warning("test warning")

        # Wymuś flush wszystkich handlerów
        for h in logger.handlers:
            h.flush()

        assert log_path.exists()
        content = log_path.read_text(encoding="utf-8")
        assert "test message" in content
        assert "test warning" in content

    def test_separate_runs_separate_files(self, tmp_path):
        from src.utils.logger import get_training_logger

        _, path1 = get_training_logger("model1", "run1", log_dir=tmp_path)
        _, path2 = get_training_logger("model1", "run2", log_dir=tmp_path)
        _, path3 = get_training_logger("model2", "run1", log_dir=tmp_path)

        assert path1 != path2
        assert path1 != path3
        assert path2 != path3


class TestModelLoggingIntegration:
    """Integracja modelu z trackingiem — full pipeline."""

    def test_xgboost_creates_logs(self, synthetic_X_y, tmp_path, monkeypatch):
        """Pełen test: XGBoost.fit() musi utworzyć log + JSON eksperymentu."""
        # Przekierowujemy logi do tmp_path
        from src.utils import config

        monkeypatch.setattr(config, "TRAINING_LOGS_DIR", tmp_path / "training")
        monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path / "experiments")

        from src.models.xgboost_model import XGBoostTriageModel

        X, y = synthetic_X_y
        model = XGBoostTriageModel(params={
            "n_estimators": 5,
            "max_depth": 3,
            "early_stopping_rounds": None,
        })

        run_id = "integration_test"
        model.fit(X, y, X_val=X, y_val=y, run_id=run_id)

        # Sprawdź log
        log_path = tmp_path / "training" / "xgboost" / f"{run_id}.log"
        assert log_path.exists(), f"Brak logu: {log_path}"
        log_text = log_path.read_text(encoding="utf-8")
        assert "Trenowanie XGBoost" in log_text
        assert "Iter" in log_text
        assert "Trening XGBoost zakończony" in log_text

        # Sprawdź JSON
        json_path = tmp_path / "experiments" / f"xgboost_{run_id}.json"
        assert json_path.exists(), f"Brak JSON: {json_path}"
        with open(json_path) as f:
            data = json.load(f)

        assert data["model_name"] == "xgboost"
        assert data["run_id"] == run_id
        assert len(data["training_history"]) > 0
        assert "params" in data
        assert "data" in data
        assert data["data"]["n_features"] > 0
