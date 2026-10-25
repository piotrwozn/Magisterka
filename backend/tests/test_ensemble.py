"""Tests for the ensemble configuration (weights + decision rule) and Layer 1A."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.config import ensure_training_on_path, get_settings
from app.models.ensemble import (
    DECISION_RULES,
    ENSEMBLE_MODELS,
    MODEL_WEIGHTS,
    DecisionRule,
    EnsembleConfig,
    apply_temperature,
    bayes_cost_matrix,
    blend,
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
        {"method": "median"},
        {"temperature": 0},
        {"temperature": -1.5},
        {"temperature": "warm"},
        {"temperature": True},
        {"temprature": 1.2},                         # typo → unknown key
        {"weights": {"xgboost": 1}, "note": "val logloss 0.71"},
        {"rule": "target"},                                         # cutpoints required
        {"rule": "target", "cutpoints": [1.0, 2.0, 3.0]},           # not 4
        {"rule": "target", "cutpoints": [1.0, 2.5, 2.0, 3.0]},      # not sorted
        {"rule": "target", "cutpoints": [1.0, 2.0, "x", 3.0]},
        {"rule": "target", "cutpoints": [1.0, 2.0, True, 3.0]},
        {"rule": "target", "cutpoints": "1,2,3,4"},
        {"rule": "target", "cutpoints": [1.0, 2.0, 3.0, float("inf")]},
        {"rule": "target", "cutpoints": [1.0, 2.0, 2.5, 3.0], "safety_q": 0},
        {"rule": "target", "cutpoints": [1.0, 2.0, 2.5, 3.0], "safety_q": 1.5},
        {"rule": "target", "cutpoints": [1.0, 2.0, 2.5, 3.0], "lam": 2.0},     # key of another rule
        {"rule": "target", "cutpoints": [1.0, 2.0, 2.5, 3.0], "safty_q": 0.1},  # typo
        {"rule": "bayes_target"},                                   # lam required
        {"rule": "bayes_target", "lam": 0},
        {"rule": "bayes_target", "lam": -1.0},
        {"rule": "bayes_target", "lam": True},
        {"rule": "bayes_target", "lam": 2.0, "safety_q": 0.1},
        {"rule": "cost", "cutpoints": [1.0, 2.0, 2.5, 3.0]},
        {"cutpoints": [1.0, 2.0, 2.5, 3.0]},                        # default rule argmax
        {"rule": "target_rule", "cutpoints": [1.0, 2.0, 2.5, 3.0]},
    ])
    def test_invalid_file_is_loud(self, tmp_path, payload):
        with pytest.raises(ValueError):
            load_ensemble_config(_write(tmp_path / "w.json", payload))

    def test_method_and_temperature(self, tmp_path):
        cfg = load_ensemble_config(_write(tmp_path / "w.json", {
            "weights": {"xgboost": 1, "catboost": 1}, "method": "geometric", "temperature": 1.25,
        }))
        assert (cfg.method, cfg.temperature, cfg.rule) == ("geometric", 1.25, "argmax")

    def test_defaults_mean_without_temperature(self, tmp_path):
        cfg = load_ensemble_config(_write(tmp_path / "w.json", {"rule": "cost", "temperature": None}))
        assert (cfg.method, cfg.temperature) == ("mean", None)
        assert EnsembleConfig().method == "mean" and EnsembleConfig().temperature is None

    def test_target_rule_with_parameters(self, tmp_path):
        cfg = load_ensemble_config(_write(tmp_path / "w.json", {
            "weights": {"lightgbm": 1}, "rule": "target",
            "cutpoints": [1, 1.9, 2.4, 3.0], "safety_q": 0.13,
        }))
        assert cfg.rule == "target"
        assert isinstance(cfg.rule, DecisionRule)
        assert cfg.rule.cutpoints == (1.0, 1.9, 2.4, 3.0)
        assert cfg.rule.safety_q == 0.13
        assert cfg.rule.spec() == {"rule": "target", "cutpoints": [1.0, 1.9, 2.4, 3.0], "safety_q": 0.13}

    def test_target_rule_safety_q_optional(self, tmp_path):
        for extra in ({}, {"safety_q": None}):
            cfg = load_ensemble_config(_write(tmp_path / "w.json", {
                "rule": "target", "cutpoints": [1.0, 1.9, 2.4, 3.0], **extra}))
            assert cfg.rule.safety_q is None

    def test_bayes_target_rule(self, tmp_path):
        cfg = load_ensemble_config(_write(tmp_path / "w.json", {"rule": "bayes_target", "lam": 3.75}))
        assert cfg.rule == "bayes_target"
        assert cfg.rule.lam == 3.75
        assert cfg.weights == MODEL_WEIGHTS

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

    def test_rules_need_their_parameters(self):
        assert DECISION_RULES == ("argmax", "cost", "target", "bayes_target")
        for name in ("target", "bayes_target"):
            with pytest.raises(ValueError):
                decide(np.full(5, 0.2), name)
            with pytest.raises(ValueError):
                EnsembleConfig(rule=name)

    def test_target_rule_counts_strictly_exceeded_cutpoints(self):
        rule = DecisionRule("target", cutpoints=[0.5, 1.5, 2.5, 3.5])
        assert decide(np.array([0.0, 0.0, 1.0, 0.0, 0.0]), rule) == 2      # EL = 2.0
        assert decide(np.array([0.0, 0.5, 0.0, 0.5, 0.0]), rule) == 2      # EL = 2.0
        assert decide(np.array([0.0, 0.0, 0.5, 0.5, 0.0]), rule) == 2      # EL = 2.5 = cutpoint → lower
        assert decide(np.array([0.0, 0.0, 0.4, 0.6, 0.0]), rule) == 3      # EL = 2.6
        assert decide(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), rule) == 0
        assert decide(np.array([0.0, 0.0, 0.0, 0.0, 1.0]), rule) == 4

    def test_target_rule_safety_net_caps_at_esi3(self):
        proba = np.array([0.05, 0.10, 0.0, 0.25, 0.60])                    # EL = 3.2 → ESI 4
        assert decide(proba, DecisionRule("target", cutpoints=[0.5, 1.5, 2.5, 3.5])) == 3
        assert decide(proba, DecisionRule("target", cutpoints=[0.5, 1.5, 2.5, 3.5], safety_q=0.15)) == 2
        assert decide(proba, DecisionRule("target", cutpoints=[0.5, 1.5, 2.5, 3.5], safety_q=0.16)) == 3
        # the net never lowers urgency
        urgent = np.array([0.6, 0.4, 0.0, 0.0, 0.0])
        assert decide(urgent, DecisionRule("target", cutpoints=[0.5, 1.5, 2.5, 3.5], safety_q=0.1)) == 0

    def test_bayes_target_rule(self):
        np.testing.assert_array_equal(bayes_cost_matrix(0.0), (np.arange(5)[None, :] - np.arange(5)[:, None]) ** 2.0)
        lam = bayes_cost_matrix(4.0) - bayes_cost_matrix(0.0)
        assert lam[:2, 2:].tolist() == [[4.0] * 3] * 2 and lam.sum() == 4.0 * 6
        proba = np.array([0.0, 0.3, 0.7, 0.0, 0.0])
        assert decide(proba, {"rule": "bayes_target", "lam": 0.5}) == 2
        assert decide(proba, {"rule": "bayes_target", "lam": 5.0}) == 1   # undertriage of ESI 2 now expensive

    def test_rule_is_a_str_with_parameters(self):
        rule = DecisionRule("target", cutpoints=[1, 2, 2.5, 3], safety_q=0.1)
        assert rule == "target" and "target" == rule and rule in DECISION_RULES
        assert rule != DecisionRule("target", cutpoints=[1, 2, 2.5, 3])
        assert EnsembleConfig(rule="cost") == EnsembleConfig(rule=DecisionRule("cost"))
        import copy
        import pickle
        assert pickle.loads(pickle.dumps(rule)) == rule
        assert copy.deepcopy(rule).safety_q == 0.1


@pytest.fixture(scope="module")
def training_decision():
    ensure_training_on_path()
    return pytest.importorskip("src.evaluation.decision")


def _random_validation(seed: int, n: int = 4000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrated random probabilities: labels drawn from them; ~70% "vitals complete"."""
    rng = np.random.default_rng(seed)
    proba = rng.dirichlet([0.3, 1.0, 1.4, 1.0, 0.5], n)
    y = np.array([rng.choice(5, p=p) for p in proba])
    vc = rng.random(n) < 0.7
    return proba, y, vc


class TestSameDecisionAsTraining:
    """A rule fitted by training code, written to ensemble_weights.json and loaded by the
    backend, decides exactly like training `decide_rule` (row by row)."""

    @pytest.mark.parametrize("kind", ["target", "target_safety", "target_guarantee", "bayes_target"])
    def test_fitted_rule_round_trip(self, tmp_path, training_decision, kind):
        dec = training_decision
        proba, y, vc = _random_validation(0)
        if kind == "bayes_target":
            fitted = dec.fit_bayes_rule(y, proba, 0.15, mask=vc)
        else:
            fitted = dec.fit_target_rule(y, proba, 0.15, guarantee=kind == "target_guarantee", mask=vc,
                                         safety_net=kind == "target_safety")
        if kind == "target_safety":
            assert fitted.safety_q is not None
        # training JSON round trip
        restored = dec.rule_from_dict(json.loads(json.dumps(fitted.to_dict())))
        spec = restored.spec()
        assert spec == fitted.spec()
        # backend loader (spec keys next to weights/method/temperature)
        path = _write(tmp_path / "ensemble_weights.json",
                      {"weights": {"lightgbm": 1.0}, "method": "mean", "temperature": None, **spec})
        cfg = load_ensemble_config(path)
        assert cfg.rule.spec() == spec

        test_proba, _, _ = _random_validation(1)
        expected = dec.decide_rule(test_proba, spec)
        np.testing.assert_array_equal(expected, fitted.predict(test_proba))
        got = np.array([decide(p, cfg.rule) for p in test_proba])
        np.testing.assert_array_equal(got, expected)
        assert len(np.unique(got)) >= 3

    @pytest.mark.parametrize("rule", ["argmax", "cost"])
    def test_simple_rules(self, training_decision, rule):
        proba, _, _ = _random_validation(2, n=2000)
        expected = training_decision.decide_rule(proba, {"rule": rule})
        np.testing.assert_array_equal([decide(p, rule) for p in proba], expected)

    def test_same_validation_as_training(self, training_decision):
        assert training_decision.RULES == DECISION_RULES
        bad = [{"rule": "target", "cutpoints": [1, 3, 2, 4]}, {"rule": "bayes_target", "lam": 0},
               {"rule": "cost", "lam": 1.0}, {"rule": "target", "cutpoints": [1, 2, 3, 4], "safety_q": 2}]
        for spec in bad:
            with pytest.raises(ValueError):
                training_decision.normalize_spec(spec)
            with pytest.raises(ValueError):
                DecisionRule.from_spec(spec)
        np.testing.assert_array_equal(bayes_cost_matrix(2.5), training_decision.bayes_cost_matrix(2.5))


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

    def test_target_rule_on_blend_and_models(self):
        cfg = EnsembleConfig(rule=DecisionRule("target", cutpoints=[0.5, 1.5, 2.5, 3.5], safety_q=0.3),
                             source="test")
        reg = _registry({
            "catboost": [0.0, 0.1, 0.2, 0.3, 0.4],       # EL 3.0 → ESI 4
            "lightgbm": [0.0, 0.5, 0.1, 0.0, 0.4],       # EL 2.3 → ESI 3
        }, cfg)
        cat, conf, preds = layer1a_ml.run_ensemble(ROW, reg)
        # blend [0, .3, .15, .15, .4]: EL 2.65 → ESI 4, but P(ESI1-2) = 0.3 ≥ q → ESI 3
        assert cat == 2
        assert conf == pytest.approx(0.15)
        assert [p.category for p in preds] == [3, 2]

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


class TestBlendAndTemperature:
    P1 = np.array([0.05, 0.60, 0.30, 0.04, 0.01])
    P2 = np.array([0.01, 0.20, 0.70, 0.08, 0.01])

    def test_mean_and_geometric(self):
        np.testing.assert_allclose(blend([self.P1, self.P2], [3, 1], "mean"), 0.75 * self.P1 + 0.25 * self.P2)
        geo = np.exp(0.75 * np.log(self.P1) + 0.25 * np.log(self.P2))
        np.testing.assert_allclose(blend([self.P1, self.P2], [3, 1], "geometric"), geo / geo.sum())
        with pytest.raises(ValueError):
            blend([self.P1], [1.0], "median")

    def test_temperature(self):
        np.testing.assert_allclose(apply_temperature(self.P1, 1.0), self.P1)
        flat = apply_temperature(self.P1, 2.0)
        assert flat.sum() == pytest.approx(1.0)
        assert flat.max() < self.P1.max()                # T > 1 flattens
        assert apply_temperature(self.P1, 0.5).max() > self.P1.max()

    def test_matches_training_blend(self):
        ensure_training_on_path()
        runs = pytest.importorskip("src.ensemble.runs")
        rng = np.random.default_rng(0)
        a, b = rng.dirichlet(np.ones(5), 20), rng.dirichlet(np.ones(5), 20)
        weights = {"lightgbm": 0.7, "xgboost": 0.3}
        for method in ("mean", "geometric"):
            expected = runs.apply_temperature(runs.blend({"lightgbm": a, "xgboost": b}, weights, method), 1.3)
            cfg = EnsembleConfig(weights=weights, method=method, temperature=1.3, source="test")
            got = np.stack([cfg.combine({"lightgbm": a[i], "xgboost": b[i]}) for i in range(20)])
            np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_temperature_applied_to_blend_before_rule(self):
        # Blend [0, .45, .55, 0, 0]: argmax ESI 3; a temperature does not change the argmax
        # but changes the confidence; per-model predictions stay unscaled
        probas = {"catboost": [0.0, 0.5, 0.5, 0.0, 0.0], "lightgbm": [0.0, 0.4, 0.6, 0.0, 0.0]}
        plain = layer1a_ml.run_ensemble(ROW, _registry(probas))
        cfg = EnsembleConfig(temperature=3.0, source="test")
        cat, conf, preds = layer1a_ml.run_ensemble(ROW, _registry(probas, cfg))
        assert cat == plain[0] == 2
        assert conf < plain[1]
        assert [p.probabilities for p in preds] == [p.probabilities for p in plain[2]]

    def test_geometric_blend_in_run_ensemble(self):
        cfg = EnsembleConfig(method="geometric", source="test")
        reg = _registry({"catboost": [0.1, 0.6, 0.1, 0.1, 0.1], "lightgbm": [0.1, 0.1, 0.6, 0.1, 0.1]}, cfg)
        _, conf, _ = layer1a_ml.run_ensemble(ROW, reg)
        geo = np.sqrt(np.array([0.1, 0.6, 0.1, 0.1, 0.1]) * np.array([0.1, 0.1, 0.6, 0.1, 0.1]))
        assert conf == pytest.approx((geo / geo.sum()).max())


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
