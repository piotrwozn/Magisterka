"""Tests for the ensemble configuration (weights + decision rule) and Layer 1A."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.config import get_settings
from app.models.ensemble import (
    ENSEMBLE_MODELS,
    MODEL_WEIGHTS,
    EnsembleConfig,
    cost_matrix,
    decide,
    load_ensemble_config,
)
from app.models.registry import ModelRegistry
from app.pipeline import layer1a_ml
from app.pipeline.orchestrator import shap_candidates

FEATURES = ["age", "triage_vital_hr"]


def _model(proba: list[float]) -> MagicMock:
    model = MagicMock()
    model.predict_proba.return_value = np.array([proba])
    return model


def _registry(probas: dict[str, list[float]], config: EnsembleConfig | None = None) -> ModelRegistry:
    reg = ModelRegistry()
    reg.models = {
        name: {"model": _model(p), "feature_names": FEATURES, "classes": list(range(5)), "path": "(mock)"}
        for name, p in probas.items()
    }
    if config is not None:
        reg.ensemble = config
    return reg


ROW = pd.DataFrame([{"age": 50.0, "triage_vital_hr": 90.0}])


def _write(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload, encoding="utf-8")
    return path


class TestDefaults:
    def test_equal_weights_for_the_six_current_models(self):
        assert set(MODEL_WEIGHTS) == {"xgboost", "lightgbm", "catboost", "random_forest", "ebm", "logreg"}
        assert set(MODEL_WEIGHTS.values()) == {1.0}
        assert ENSEMBLE_MODELS == tuple(MODEL_WEIGHTS)
        assert layer1a_ml.MODEL_WEIGHTS is MODEL_WEIGHTS

    def test_registry_default_config(self):
        cfg = ModelRegistry().ensemble
        assert cfg.rule == "argmax"
        assert cfg.is_default
        assert cfg.weights == MODEL_WEIGHTS


class TestLoadConfig:
    def test_missing_file_gives_defaults(self, tmp_path):
        cfg = load_ensemble_config(tmp_path / "ensemble_weights.json")
        assert cfg == EnsembleConfig()

    def test_weights_and_rule(self, tmp_path):
        path = _write(tmp_path / "w.json", {"weights": {"xgboost": 0.6, "catboost": 0.4}, "rule": "cost"})
        cfg = load_ensemble_config(path)
        assert cfg.weights == {"xgboost": 0.6, "catboost": 0.4}
        assert cfg.rule == "cost"
        assert not cfg.is_default
        # Unlisted models get weight 0 → not part of the ensemble
        assert cfg.active(["catboost", "lightgbm", "xgboost"]) == ["catboost", "xgboost"]

    def test_rule_only_keeps_equal_weights(self, tmp_path):
        cfg = load_ensemble_config(_write(tmp_path / "w.json", {"rule": "cost"}))
        assert cfg.weights == MODEL_WEIGHTS
        assert cfg.rule == "cost"

    def test_unknown_model_ignored(self, tmp_path):
        cfg = load_ensemble_config(_write(tmp_path / "w.json", {"weights": {"xgboost": 1, "extra_trees": 2}}))
        assert cfg.weights == {"xgboost": 1.0}

    @pytest.mark.parametrize("payload", [
        "{not json",
        [1, 2],
        {"weights": {"xgboost": -1}},
        {"weights": {"xgboost": "a lot"}},
        {"weights": {"xgboost": 0, "catboost": 0}},
        {"weights": {"extra_trees": 1}},
        {"weights": {}},
        {"weights": {"xgboost": 1}, "rule": "median"},
    ])
    def test_invalid_file_is_loud(self, tmp_path, payload):
        with pytest.raises(ValueError):
            load_ensemble_config(_write(tmp_path / "w.json", payload))

    def test_registry_loads_file_from_models_dir(self, tmp_path):
        models_dir = tmp_path / "backend" / "models"
        models_dir.mkdir(parents=True)
        _write(models_dir / "ensemble_weights.json", {"weights": {"lightgbm": 1}, "rule": "cost"})
        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=tmp_path / "training" / "runs")
        assert reg.ensemble.weights == {"lightgbm": 1.0}
        assert reg.ensemble.rule == "cost"


class TestDecisionRule:
    def test_cost_matrix_matches_training(self):
        training_root = get_settings().project_root / "training"
        if str(training_root) not in sys.path:
            sys.path.insert(0, str(training_root))
        config = pytest.importorskip("src.utils.config")
        assert config.UNDERTRIAGE_COST_MULTIPLIER == 3.0
        assert config.OVERTRIAGE_COST_MULTIPLIER == 1.0
        expected = np.array([[(p - t) ** 2 * (3.0 if p > t else 1.0) for p in range(5)] for t in range(5)])
        np.testing.assert_array_equal(cost_matrix(), expected)

    def test_argmax(self):
        assert decide(np.array([0.1, 0.2, 0.5, 0.1, 0.1]), "argmax") == 2

    def test_cost_rule_prefers_more_urgent_level(self):
        proba = np.array([0.0, 0.4, 0.6, 0.0, 0.0])
        assert decide(proba, "argmax") == 2
        assert decide(proba, "cost") == 1          # undertriage costs 3×

    def test_unknown_rule(self):
        with pytest.raises(ValueError):
            decide(np.full(5, 0.2), "median")


class TestRunEnsemble:
    def test_equal_weights_average(self):
        reg = _registry({
            "catboost": [0.0, 0.8, 0.2, 0.0, 0.0],
            "lightgbm": [0.0, 0.2, 0.8, 0.0, 0.0],
            "xgboost":  [0.0, 0.0, 0.9, 0.1, 0.0],
        })
        cat, conf, preds = layer1a_ml.run_ensemble(ROW, reg)
        assert cat == 2
        assert conf == pytest.approx((0.2 + 0.8 + 0.9) / 3)
        assert [p.model_name for p in preds] == ["catboost", "lightgbm", "xgboost"]

    def test_custom_weights_change_the_decision(self):
        cfg = EnsembleConfig(weights={"catboost": 0.9, "lightgbm": 0.1}, rule="argmax", source="test")
        reg = _registry({
            "catboost": [0.0, 0.8, 0.2, 0.0, 0.0],
            "lightgbm": [0.0, 0.2, 0.8, 0.0, 0.0],
            "xgboost":  [0.0, 0.0, 0.9, 0.1, 0.0],
        }, cfg)
        cat, conf, preds = layer1a_ml.run_ensemble(ROW, reg)
        assert cat == 1
        assert conf == pytest.approx(0.9 * 0.8 + 0.1 * 0.2)
        # xgboost has weight 0 → not run, not shown
        assert [p.model_name for p in preds] == ["catboost", "lightgbm"]
        reg.models["xgboost"]["model"].predict_proba.assert_not_called()

    def test_cost_rule_applies_to_ensemble_and_models(self):
        cfg = EnsembleConfig(rule="cost", source="test")
        reg = _registry({"lightgbm": [0.0, 0.4, 0.6, 0.0, 0.0]}, cfg)
        cat, conf, preds = layer1a_ml.run_ensemble(ROW, reg)
        assert cat == 1
        assert conf == pytest.approx(0.4)
        assert preds[0].category == 1

    def test_no_weighted_model_loaded_raises(self):
        cfg = EnsembleConfig(weights={"ebm": 1.0}, source="test")
        reg = _registry({"lightgbm": [0.2] * 5}, cfg)
        with pytest.raises(RuntimeError, match="weight"):
            layer1a_ml.run_ensemble(ROW, reg)

    def test_failing_model_is_skipped(self):
        reg = _registry({"catboost": [0.0, 0.0, 1.0, 0.0, 0.0], "lightgbm": [0.2] * 5})
        reg.models["lightgbm"]["model"].predict_proba.side_effect = RuntimeError("boom")
        cat, _, preds = layer1a_ml.run_ensemble(ROW, reg)
        assert cat == 2
        assert [p.model_name for p in preds] == ["catboost"]


class TestFullProba:
    def test_missing_class_filled_with_zero(self):
        model = _model([0.25, 0.75])                # trained without ESI 1 / 3 / 5
        proba = layer1a_ml.full_proba(model, ROW, classes=[1, 3])
        np.testing.assert_allclose(proba, [0.0, 0.25, 0.0, 0.75, 0.0])

    def test_renormalised_float64(self):
        proba = layer1a_ml.full_proba(_model([0.2, 0.2, 0.2, 0.2, 0.1]), ROW)
        assert proba.dtype == np.float64
        assert proba.sum() == pytest.approx(1.0)

    def test_column_count_mismatch_raises(self):
        with pytest.raises(ValueError):
            layer1a_ml.full_proba(_model([0.5, 0.5]), ROW, classes=[0, 1, 2])


class TestShapModelChoice:
    def test_highest_weight_tree_model_first(self):
        weights = {"catboost": 0.5, "lightgbm": 0.2, "ebm": 0.9, "logreg": 0.3}
        assert shap_candidates(["catboost", "ebm", "lightgbm", "logreg"], weights) == ["catboost", "lightgbm"]

    def test_equal_weights_use_fixed_order(self):
        ids = ["catboost", "ebm", "random_forest", "xgboost"]
        assert shap_candidates(ids, MODEL_WEIGHTS) == ["xgboost", "catboost", "random_forest"]

    def test_no_tree_model(self):
        assert shap_candidates(["ebm", "logreg"], MODEL_WEIGHTS) == []
