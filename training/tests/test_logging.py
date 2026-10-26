"""
Testy systemu logowania treningu i trackingu eksperymentów.

Sprawdza, że każdy fit() modelu:
    1. Tworzy plik logu w logs/training/<model>/<run_id>.log (UTF-8) i zamyka go po treningu
    2. Tworzy JSON eksperymentu w logs/experiments/<model>_<run_id>.json
    3. Loguje hiperparametry, statystyki danych, urządzenie, sprzęt, czas treningu
    4. Zapisuje training_history `es_*` (modele z early stopping: XGBoost / LightGBM / CatBoost)

oraz że logowanie przeżywa przekierowany stdout w stronie kodowej Windows (cp1250).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def _cpu_only():
    """Testy zawsze na CPU, także gdy powłoka eksportuje SOR_AI_DEVICE_* (np. z run_all)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SOR_AI_DEVICE", "cpu")
        for var in ("SOR_AI_DEVICE_XGBOOST", "SOR_AI_DEVICE_CATBOOST"):
            mp.delenv(var, raising=False)
        yield


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

        with open(path, encoding="utf-8") as f:
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
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        env = data["environment"]
        assert "hostname" in env
        assert "platform" in env
        assert "python_version" in env
        hw = env["hardware"]
        assert hw["physical_cores"] >= 1
        assert "libraries" in hw and "nvidia_smi" in hw

    def test_log_device(self, tmp_path):
        from src.utils.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker("test", "devtest", output_dir=tmp_path)
        tracker.log_device(requested="gpu", actual="cpu", fallback_reason="CUDA error 35 — żółć")
        data = json.loads(tracker.save().read_text(encoding="utf-8"))
        assert data["device"] == {"requested": "gpu", "actual": "cpu", "fallback_reason": "CUDA error 35 — żółć"}

    def test_json_is_utf8_with_polish_text(self, tmp_path):
        from src.utils.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker("test", "utf8test", output_dir=tmp_path)
        tracker.add_note("Pacjent → ESI ≥ 2, SpO₂ niska, żółty")
        path = tracker.save()
        assert "Pacjent → ESI ≥ 2, SpO₂ niska, żółty" in path.read_bytes().decode("utf-8")

    def test_log_data_info(self, tmp_path):
        from src.utils.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker("test", "datatest", output_dir=tmp_path)
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
        y = pd.Series([0, 1, 2, 0])

        tracker.log_data_info(X, y, feature_set="custom")
        path = tracker.save()

        with open(path, encoding="utf-8") as f:
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

    def test_file_handler_utf8_without_fsync(self, tmp_path):
        from src.utils.logger import close_training_logger, get_training_logger

        logger, log_path = get_training_logger("utf8_logger", "run_utf8", log_dir=tmp_path)
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert type(file_handlers[0]) is logging.FileHandler        # bez fsync po każdym rekordzie
        assert file_handlers[0].encoding == "utf-8"
        logger.info("═ → ≥ ₂ ł ✓")
        close_training_logger(logger)
        assert "═ → ≥ ₂ ł ✓" in log_path.read_text(encoding="utf-8")

    def test_close_training_logger(self, tmp_path):
        from src.utils.logger import close_training_logger, get_training_logger

        logger, log_path = get_training_logger("close_logger", "run_close", log_dir=tmp_path)
        file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
        close_training_logger(logger)
        ours = [h for h in logger.handlers if not type(h).__module__.startswith("_pytest")]
        assert ours == []
        assert file_handler.stream is None                          # plik zamknięty (Windows go zwalnia)
        log_path.rename(tmp_path / "moved.log")                      # da się przenieść
        close_training_logger(logger)                                # idempotentne
        close_training_logger(None)

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
        from src.utils.config import TFM_MODELS

        # 6 modeli klasycznych; modele fundamentalne (TFM) mają logowanie w tests/test_tfm_models.py
        assert {m for m, c in MODEL_REGISTRY.items() if not c.pretrained} == set(self.FAST_PARAMS)
        assert {m for m, c in MODEL_REGISTRY.items() if c.pretrained} == set(TFM_MODELS)
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
            # Krzywa early stopping to 10% TRAIN, nie split `val` — nazwy es_*
            assert any(k.startswith("es_") for k in data["training_history"][0])
            assert not any(k.startswith("val_") for k in data["training_history"][0])
        if model.feature_importances() is not None:
            assert data["feature_importances"]

        # Urządzenie i sprzęt
        assert data["device"] == {"requested": "cpu", "actual": "cpu", "fallback_reason": None}
        assert "Urządzenie (faktyczne): cpu" in log_text
        assert data["environment"]["hardware"]["physical_cores"] >= 1

        # Plik logu zamknięty po treningu
        assert not [h for h in model.train_logger.handlers if isinstance(h, logging.FileHandler)]


# ─────────────────────────────────────────
# Windows: przekierowany stdout w cp1250
# ─────────────────────────────────────────
class TestStdoutEncoding:
    """Na Windows `> plik` / potok używa strony kodowej (cp1250) — bez znaków ═ → ≥ ₂."""

    TEXT = "═ → ≥ ₂ ł"

    def _run(self, code: str, tmp_path: Path) -> tuple[subprocess.CompletedProcess, bytes]:
        env = {**os.environ, "PYTHONIOENCODING": "cp1250"}
        env.pop("PYTHONUTF8", None)
        out_path = tmp_path / "out.txt"
        with open(out_path, "wb") as out:
            proc = subprocess.run(
                [sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env,
                stdout=out, stderr=subprocess.PIPE, timeout=120,
            )
        return proc, out_path.read_bytes()

    def test_logger_and_print_survive_cp1250(self, tmp_path):
        code = (
            "from src.utils.logger import get_logger\n"
            "log = get_logger('enc_test')\n"
            f"log.info({self.TEXT!r})\n"
            f"print('print:', {self.TEXT!r})\n"
        )
        proc, out = self._run(code, tmp_path)
        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        assert b"Logging error" not in proc.stderr
        text = out.decode("utf-8")
        assert self.TEXT in text
        assert f"print: {self.TEXT}" in text

    def test_without_fix_cp1250_breaks(self, tmp_path):
        """Kontrola: bez importu loggera ten sam print kończy proces (dowód, że test coś sprawdza)."""
        proc, _ = self._run(f"print({self.TEXT!r})", tmp_path)
        assert proc.returncode != 0
