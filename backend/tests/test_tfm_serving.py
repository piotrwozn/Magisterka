"""Tests for opt-in foundation-model (TFM) serving and the no-TFM fallback recipe.

TFM artifacts are real `TFMPredictor` objects fitted by the training wrappers with the deterministic fake
estimator (`SOR_AI_TFM_FAKE=1`, no torch needed); torch / CUDA are mocked through
`registry.tfm_runtime_status` (or a fake `torch` module for the status function itself).
"""
from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from app.config import ensure_training_on_path
from app.models import registry as registry_module
from app.models.ensemble import (
    ENSEMBLE_MODELS,
    ENSEMBLE_WEIGHTS_FILE,
    ENSEMBLE_WEIGHTS_NO_TFM_FILE,
    KNOWN_MODELS,
    MODEL_WEIGHTS,
    TFM_MODELS,
    EnsembleConfig,
    load_ensemble_config,
)
from app.models.registry import (
    TARGET_MODELS,
    ModelRegistry,
    TFMRuntime,
    training_data_version,
)
from app.pipeline import layer1a_ml

FEATURES = ["age", "triage_vital_hr", "cc_chestpain"]
CLASS_NAMES = ["ESI1", "ESI2", "ESI3", "ESI4", "ESI5"]
DATA_VERSION = training_data_version()
ROW = pd.DataFrame([{"age": 50.0, "triage_vital_hr": 95.0, "cc_chestpain": 1.0}])
GB = 2 ** 30


def _frame(n: int = 240, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"age": rng.normal(50, 15, n), "triage_vital_hr": rng.normal(85, 15, n),
                      "cc_chestpain": rng.integers(0, 2, n).astype(float)})
    y = np.arange(n) % 5
    return X, y


def _classical(path: Path, name: str = "lightgbm") -> Path:
    X, y = _frame()
    artifact = {"name": name, "run_id": "20260101_000000", "params": {},
                "model": LogisticRegression(max_iter=300).fit(X, y), "feature_names": FEATURES,
                "feature_set": "triage", "classes_": np.arange(5), "class_names": CLASS_NAMES, "label": "esi",
                "data_version": DATA_VERSION}
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


@pytest.fixture
def fake_tfm(monkeypatch):
    """Training wrappers with the fake estimator (CPU, no torch)."""
    ensure_training_on_path()
    monkeypatch.setenv("SOR_AI_TFM_FAKE", "1")
    for var in ("SOR_AI_TABPFN_MODEL_PATH", "SOR_AI_TABICL_MODEL_PATH", "SOR_AI_TFM_FAKE_OOM"):
        monkeypatch.delenv(var, raising=False)


def _tfm(path: Path, name: str = "tabpfn", n_bags: int = 2) -> Path:
    """TFM artifact saved by the training wrapper (bundle["model"] = TFMPredictor, `pretrained: True`)."""
    from src.models import get_model

    X, y = _frame(seed=1)
    model = get_model(name, params={"n_ctx": 80, "n_bags": n_bags, "max_bags": n_bags, "n_estimators": 1,
                                    "device": "cpu", "n_jobs": 1})
    model.fit_estimator(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def _weights(path: Path, weights: dict, **extra) -> Path:
    path.write_text(json.dumps({"weights": weights, "method": "mean", "temperature": 1.1, "rule": "argmax",
                                **extra}), encoding="utf-8")
    return path


@pytest.fixture
def deploy(tmp_path):
    """<root>/backend/models with a classical LightGBM and an XGBoost artifact; runs dir empty."""
    models_dir = tmp_path / "backend" / "models"
    runs_dir = tmp_path / "training" / "runs"
    runs_dir.mkdir(parents=True)
    _classical(models_dir / "lightgbm_1.joblib", "lightgbm")
    _classical(models_dir / "xgboost_1.joblib", "xgboost")
    return models_dir, runs_dir


def _cuda(free_gb: float = 12.0, libraries: tuple[str, ...] = TFM_MODELS):
    def status(models):
        return TFMRuntime(True, None, int(free_gb * GB), int(16 * GB), {m: m in libraries for m in models})
    return status


def _load(models_dir: Path, runs_dir: Path) -> ModelRegistry:
    reg = ModelRegistry()
    reg.load_from_dir(models_dir, runs_dir=runs_dir)
    return reg


class TestConstants:
    def test_classical_lists_unchanged_and_tfm_separate(self):
        assert ENSEMBLE_MODELS == ("xgboost", "lightgbm", "catboost", "random_forest", "ebm", "logreg")
        assert TARGET_MODELS == ENSEMBLE_MODELS
        assert set(MODEL_WEIGHTS) == set(ENSEMBLE_MODELS)              # TFMs never in the default blend
        assert TFM_MODELS == ("tabpfn", "tabicl")
        assert KNOWN_MODELS == ENSEMBLE_MODELS + TFM_MODELS

    def test_tfm_weights_are_known(self, tmp_path):
        cfg = load_ensemble_config(_weights(tmp_path / "w.json", {"lightgbm": 0.6, "tabpfn": 0.4}))
        assert cfg.weights == {"lightgbm": 0.6, "tabpfn": 0.4}
        assert cfg.tfm_members == ["tabpfn"]
        assert EnsembleConfig().tfm_members == []

    @pytest.mark.parametrize("value, expected", [("1", True), ("true", True), ("ON", True), ("0", False),
                                                 ("", False), ("no", False)])
    def test_opt_in_env(self, monkeypatch, value, expected):
        monkeypatch.setenv(registry_module.TFM_SERVING_ENV, value)
        assert registry_module.tfm_serving_enabled() is expected

    def test_reserve_env(self, monkeypatch):
        monkeypatch.delenv(registry_module.TFM_RESERVE_ENV, raising=False)
        assert registry_module.tfm_reserve_bytes() == 2048 * 2 ** 20
        monkeypatch.setenv(registry_module.TFM_RESERVE_ENV, "512")
        assert registry_module.tfm_reserve_bytes() == 512 * 2 ** 20
        monkeypatch.setenv(registry_module.TFM_RESERVE_ENV, "a lot")
        assert registry_module.tfm_reserve_bytes() == 2048 * 2 ** 20


class TestRuntimeStatus:
    def test_torch_absent(self, monkeypatch):
        monkeypatch.setattr(registry_module, "_has_module", lambda name: False)
        status = registry_module.tfm_runtime_status(["tabpfn"])
        assert not status.cuda and "torch is not installed" in status.reason
        assert status.libraries == {"tabpfn": False}

    def _fake_torch(self, monkeypatch, available: bool, free: int = 10 * GB):
        torch = types.ModuleType("torch")
        torch.__version__ = "2.14.1+cpu"
        torch.version = types.SimpleNamespace(cuda=None if not available else "13.0")
        torch.cuda = types.SimpleNamespace(is_available=lambda: available, mem_get_info=lambda: (free, 16 * GB))
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setattr(registry_module, "_has_module", lambda name: True)

    def test_cuda_absent(self, monkeypatch):
        self._fake_torch(monkeypatch, available=False)
        status = registry_module.tfm_runtime_status(["tabpfn", "tabicl"])
        assert not status.cuda and "CUDA is not available" in status.reason

    def test_cuda_present_reports_free_vram(self, monkeypatch):
        self._fake_torch(monkeypatch, available=True, free=7 * GB)
        status = registry_module.tfm_runtime_status(["tabicl"])
        assert status.cuda and status.free_vram_bytes == 7 * GB and status.libraries == {"tabicl": True}


class TestFallback:
    def test_default_off_uses_no_tfm_recipe(self, deploy, fake_tfm, monkeypatch, caplog):
        models_dir, runs_dir = deploy
        monkeypatch.delenv(registry_module.TFM_SERVING_ENV, raising=False)
        _tfm(models_dir / "tabpfn_1.joblib")
        _weights(models_dir / ENSEMBLE_WEIGHTS_FILE, {"lightgbm": 0.5, "tabpfn": 0.5}, temperature=0.9)
        _weights(models_dir / ENSEMBLE_WEIGHTS_NO_TFM_FILE, {"lightgbm": 0.7, "xgboost": 0.3}, temperature=1.2)
        monkeypatch.setattr(registry_module, "tfm_runtime_status",
                            lambda models: pytest.fail("torch must not be probed without the opt-in"))
        with caplog.at_level(logging.WARNING):
            reg = _load(models_dir, runs_dir)
        assert "tabpfn" not in reg.models and reg.loaded_ids == ["lightgbm", "xgboost"]
        assert reg.ensemble.source.endswith(ENSEMBLE_WEIGHTS_NO_TFM_FILE)
        assert reg.ensemble.weights == {"lightgbm": 0.7, "xgboost": 0.3} and reg.ensemble.temperature == 1.2
        assert reg.ensemble_full.weights == {"lightgbm": 0.5, "tabpfn": 0.5}
        assert "disabled" in reg.tfm_status["tabpfn"]
        assert "re-fitted on VAL without foundation models" in caplog.text
        category, _, preds = layer1a_ml.run_ensemble(ROW, reg)
        assert {p.model_name for p in preds} == {"lightgbm", "xgboost"} and 0 <= category <= 4

    def test_without_no_tfm_file_renormalises_with_warning(self, deploy, monkeypatch, caplog):
        models_dir, runs_dir = deploy
        monkeypatch.delenv(registry_module.TFM_SERVING_ENV, raising=False)
        _weights(models_dir / ENSEMBLE_WEIGHTS_FILE, {"lightgbm": 0.5, "tabicl": 0.5})
        with caplog.at_level(logging.WARNING):
            reg = _load(models_dir, runs_dir)
        assert reg.ensemble is reg.ensemble_full and reg.ensemble_no_tfm is None
        assert "fitted for the FULL blend" in caplog.text
        assert reg.ensemble.active(reg.loaded_ids) == ["lightgbm"]

    def test_classical_recipe_ignores_no_tfm_file(self, deploy, monkeypatch):
        models_dir, runs_dir = deploy
        monkeypatch.setenv(registry_module.TFM_SERVING_ENV, "1")
        _weights(models_dir / ENSEMBLE_WEIGHTS_FILE, {"lightgbm": 0.5, "xgboost": 0.5})
        _weights(models_dir / ENSEMBLE_WEIGHTS_NO_TFM_FILE, {"lightgbm": 1.0})
        monkeypatch.setattr(registry_module, "tfm_runtime_status", lambda models: pytest.fail("no TFM weighted"))
        reg = _load(models_dir, runs_dir)
        assert reg.ensemble.weights == {"lightgbm": 0.5, "xgboost": 0.5} and reg.tfm_status == {}

    def test_no_tfm_file_with_tfm_weight_is_loud(self, deploy, monkeypatch):
        models_dir, runs_dir = deploy
        _weights(models_dir / ENSEMBLE_WEIGHTS_FILE, {"lightgbm": 0.5, "tabpfn": 0.5})
        _weights(models_dir / ENSEMBLE_WEIGHTS_NO_TFM_FILE, {"lightgbm": 0.5, "tabpfn": 0.5})
        with pytest.raises(ValueError, match="no-TFM recipe"):
            _load(models_dir, runs_dir)

    @pytest.mark.parametrize("reason", ["torch is not installed (training/requirements-tfm.txt + a CUDA wheel)",
                                        "CUDA is not available to torch 2.14.1+cpu"])
    def test_opt_in_without_torch_or_cuda_falls_back(self, deploy, fake_tfm, monkeypatch, caplog, reason):
        models_dir, runs_dir = deploy
        monkeypatch.setenv(registry_module.TFM_SERVING_ENV, "1")
        _tfm(models_dir / "tabpfn_1.joblib")
        _weights(models_dir / ENSEMBLE_WEIGHTS_FILE, {"lightgbm": 0.5, "tabpfn": 0.5})
        _weights(models_dir / ENSEMBLE_WEIGHTS_NO_TFM_FILE, {"lightgbm": 1.0})
        monkeypatch.setattr(registry_module, "tfm_runtime_status",
                            lambda models: TFMRuntime(False, reason, libraries={m: False for m in models}))
        with caplog.at_level(logging.WARNING):
            reg = _load(models_dir, runs_dir)
        assert "tabpfn" not in reg.models and reason in reg.tfm_status["tabpfn"]
        assert reg.ensemble.weights == {"lightgbm": 1.0}
        assert reason in caplog.text


class TestOptInServing:
    @pytest.fixture
    def served(self, deploy, fake_tfm, monkeypatch):
        models_dir, runs_dir = deploy
        monkeypatch.setenv(registry_module.TFM_SERVING_ENV, "1")
        monkeypatch.setattr(registry_module, "tfm_runtime_status", _cuda())
        _tfm(models_dir / "tabpfn_1.joblib")
        _weights(models_dir / ENSEMBLE_WEIGHTS_FILE, {"lightgbm": 0.5, "tabpfn": 0.5})
        _weights(models_dir / ENSEMBLE_WEIGHTS_NO_TFM_FILE, {"lightgbm": 1.0}, temperature=1.3)
        return models_dir, runs_dir

    def test_served_and_warmed_up(self, served):
        reg = _load(*served)
        bundle = reg.models["tabpfn"]
        model = bundle["model"]
        assert reg.tfm_status["tabpfn"] == "served" and bundle["pretrained"] is True
        assert model.keep_fitted_bags == "all" and len(model._fitted) == model.n_bags == 2
        assert reg.ensemble is reg.ensemble_full and reg.ensemble.weights == {"lightgbm": 0.5, "tabpfn": 0.5}
        assert reg.ensemble_no_tfm.temperature == 1.3                    # kept for request-time failures
        category, confidence, preds = layer1a_ml.run_ensemble(ROW, reg)
        assert {p.model_name for p in preds} == {"lightgbm", "tabpfn"}
        tfm_pred = next(p for p in preds if p.model_name == "tabpfn")
        assert len(tfm_pred.probabilities) == 5 and abs(sum(tfm_pred.probabilities) - 1) < 1e-9
        assert len(model._fitted) == 2                                  # no re-fit per request
        expected = reg.ensemble.combine({p.model_name: np.array(p.probabilities) for p in preds})
        assert confidence == pytest.approx(float(expected[category]))

    def test_run_artifact_found_with_training_on_path(self, deploy, fake_tfm, monkeypatch):
        models_dir, runs_dir = deploy
        monkeypatch.setenv(registry_module.TFM_SERVING_ENV, "1")
        monkeypatch.setattr(registry_module, "tfm_runtime_status", _cuda())
        run_dir = runs_dir / "20260101_000000_tabicl_triage_standard"
        _tfm(run_dir / "model.joblib", name="tabicl", n_bags=1)
        (run_dir / "run_info.json").write_text(json.dumps({
            "model": "tabicl", "feature_set": "triage", "tuned": True, "data_version": DATA_VERSION,
            "weights_strategy": "none", "train_sample": None, "pretrained": True}), encoding="utf-8")
        (run_dir / "DONE.json").write_text(json.dumps({"model": "tabicl", "tuned": True}), encoding="utf-8")
        _weights(models_dir / ENSEMBLE_WEIGHTS_FILE, {"lightgbm": 0.5, "tabicl": 0.5})
        reg = _load(models_dir, runs_dir)
        assert reg.models["tabicl"]["path"] == str(run_dir / "model.joblib")
        assert reg.tfm_status["tabicl"] == "served"

    def test_vram_budget_exceeded_falls_back(self, served, monkeypatch, caplog):
        monkeypatch.setattr(registry_module, "tfm_runtime_status", _cuda(free_gb=1.0))   # < 2 GB reserve
        with caplog.at_level(logging.WARNING):
            reg = _load(*served)
        assert "tabpfn" not in reg.models and "VRAM budget exceeded" in reg.tfm_status["tabpfn"]
        assert reg.ensemble.weights == {"lightgbm": 1.0}

    def test_warm_up_failure_falls_back(self, served, monkeypatch):
        from src.models.tfm_base import TFMPredictor

        def boom(self):
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(TFMPredictor, "warm_up", boom)
        reg = _load(*served)
        assert "tabpfn" not in reg.models and "warm-up of tabpfn failed" in reg.tfm_status["tabpfn"]
        assert reg.ensemble.source.endswith(ENSEMBLE_WEIGHTS_NO_TFM_FILE)

    def test_library_missing_falls_back(self, served, monkeypatch):
        monkeypatch.setattr(registry_module, "tfm_runtime_status", _cuda(libraries=()))
        reg = _load(*served)
        assert "library not installed" in reg.tfm_status["tabpfn"] and "tabpfn" not in reg.models

    def test_all_weighted_tfms_or_none(self, served):
        models_dir, runs_dir = served
        _weights(models_dir / ENSEMBLE_WEIGHTS_FILE, {"lightgbm": 0.4, "tabpfn": 0.3, "tabicl": 0.3})
        reg = _load(models_dir, runs_dir)                                # no tabicl artifact anywhere
        assert "tabpfn" not in reg.models and "tabicl" not in reg.models
        assert "full recipe cannot be served" in reg.tfm_status["tabpfn"]
        assert reg.ensemble.weights == {"lightgbm": 1.0}

    def test_request_failure_switches_to_no_tfm_recipe(self, served, caplog):
        reg = _load(*served)
        reg.models["tabpfn"]["model"].predict_proba = lambda X: (_ for _ in ()).throw(RuntimeError("CUDA error"))
        with caplog.at_level(logging.WARNING):
            category, confidence, preds = layer1a_ml.run_ensemble(ROW, reg)
        assert "tabpfn" not in reg.models and "dropped" in reg.tfm_status["tabpfn"]
        assert reg.ensemble is reg.ensemble_no_tfm
        assert [p.model_name for p in preds] == ["lightgbm"]
        expected = reg.ensemble.combine({"lightgbm": np.array(preds[0].probabilities)})   # T = 1.3 recipe
        assert confidence == pytest.approx(float(expected[category]))
        assert "Switching the ensemble" in caplog.text

    def test_request_failure_without_no_tfm_file_renormalises(self, served):
        models_dir, runs_dir = served
        (models_dir / ENSEMBLE_WEIGHTS_NO_TFM_FILE).unlink()
        reg = _load(models_dir, runs_dir)
        reg.models["tabpfn"]["model"].predict_proba = lambda X: (_ for _ in ()).throw(RuntimeError("CUDA error"))
        _, _, preds = layer1a_ml.run_ensemble(ROW, reg)
        assert [p.model_name for p in preds] == ["lightgbm"] and reg.ensemble is reg.ensemble_full
        assert "tabpfn" not in reg.models

    def test_explicit_config_does_not_drop_models(self, served):
        reg = _load(*served)
        reg.models["tabpfn"]["model"].predict_proba = lambda X: (_ for _ in ()).throw(RuntimeError("boom"))
        cfg = EnsembleConfig(weights={"lightgbm": 1.0, "tabpfn": 1.0})
        _, _, preds = layer1a_ml.run_ensemble(ROW, reg, cfg)
        assert [p.model_name for p in preds] == ["lightgbm"] and "tabpfn" in reg.models
