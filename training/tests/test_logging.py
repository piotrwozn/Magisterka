"""
Testy systemu logowania treningu i trackingu eksperymentów.

Sprawdza, że każdy fit() modelu:
    1. Tworzy plik logu w logs/training/<model>/<run_id>.log
    2. Tworzy JSON eksperymentu w logs/experiments/<model>_<run_id>.json
    3. Loguje hiperparametry, statystyki danych, czas treningu
    4. Zapisuje training_history (modele z early stopping: XGBoost / LightGBM / CatBoost)
"""

from __future__ import annotations

import json

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
    """Integracja modelu z trackingiem — każdy fit() zapisuje log i JSON eksperymentu."""

    FAST_PARAMS = {
        "logreg": {"max_iter": 300},
        "lightgbm": {"n_estimators": 10, "num_leaves": 7, "min_child_samples": 5},
        "xgboost": {"n_estimators": 10, "max_depth": 3, "early_stopping_rounds": 5},
        "catboost": {"iterations": 10, "depth": 3, "verbose": 0},
        "random_forest": {"n_estimators": 10},
        "ebm": {"max_rounds": 50, "outer_bags": 1, "early_stopping_rounds": 10},
    }
    # Fragment komunikatu w logu treningu (logi po polsku)
    LOG_LABEL = {
        "logreg": "regresji logistycznej",
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "catboost": "CatBoost",
        "random_forest": "Random Forest",
        "ebm": "EBM",
    }
    # Modele z historią iteracji (early stopping na zbiorze ES)
    WITH_HISTORY = {"lightgbm", "xgboost", "catboost"}

    @pytest.fixture
    def log_dirs(self, tmp_path, monkeypatch):
        from src.utils import config

        monkeypatch.setattr(config, "TRAINING_LOGS_DIR", tmp_path / "training")
        monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path / "experiments")
        return tmp_path

    def test_registry_matches_checked_models(self):
        from src.models import MODEL_REGISTRY

        assert set(MODEL_REGISTRY) == set(self.FAST_PARAMS)
        assert "extra_trees" not in MODEL_REGISTRY
        assert "hist_gbt" not in MODEL_REGISTRY

    @pytest.mark.parametrize("name", ["logreg", "lightgbm", "xgboost", "catboost", "random_forest", "ebm"])
    def test_fit_creates_log_and_experiment_json(self, synthetic_X_y, log_dirs, name):
        from src.models import get_model

        X, y = synthetic_X_y
        X_fit, y_fit, X_es, y_es = X.iloc[:320], y.iloc[:320], X.iloc[320:], y.iloc[320:]
        model = get_model(name, params=self.FAST_PARAMS[name])

        run_id = f"integration_{name}"
        model.fit(X_fit, y_fit, X_val=X_es, y_val=y_es, run_id=run_id, feature_set="triage")

        # Log treningu
        log_path = log_dirs / "training" / name / f"{run_id}.log"
        assert log_path.exists(), f"Brak logu: {log_path}"
        assert model.train_log_path == log_path
        log_text = log_path.read_text(encoding="utf-8")
        label = self.LOG_LABEL[name]
        assert f"Trenowanie {label}" in log_text
        assert f"Trening {label} zakończony" in log_text
        assert "Wagi sampli" in log_text
        assert "Rozkład klas (train)" in log_text

        # JSON eksperymentu
        json_path = log_dirs / "experiments" / f"{name}_{run_id}.json"
        assert json_path.exists(), f"Brak JSON: {json_path}"
        assert model.tracker.json_path == json_path
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["model_name"] == name
        assert data["run_id"] == run_id
        assert data["params"]
        assert data["data"]["n_features"] == X.shape[1]
        assert data["data"]["n_train"] == len(X_fit)
        assert data["data"]["n_val"] == len(X_es)
        assert data["data"]["feature_set"] == "triage"
        assert data["duration_seconds"] >= 0
        assert data["artifacts"]["training_log"] == str(log_path.resolve())
        if name in self.WITH_HISTORY:
            assert len(data["training_history"]) > 0
            assert "Iter" in log_text
        if model.feature_importances() is not None:
            assert data["feature_importances"]
