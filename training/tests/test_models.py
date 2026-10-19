"""
Testy modeli ESI (`src.models`): wszystkie 6 modeli z rejestru.

Każdy model na małym numerycznym zbiorze z brakami (NaN) musi:
    - wytrenować się (z early stopping na osobnym zbiorze),
    - zwrócić prawdopodobieństwa (n, 5) sumujące się do 1 (indeks 0–4 = ESI 1–5),
    - po zapisie i odczycie dawać identyczne prawdopodobieństwa,
    - zwrócić ważności cech jako DataFrame[feature, importance] albo None.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models import (
    MODEL_ALIASES,
    MODEL_REGISTRY,
    BaseTriageModel,
    compute_sample_weights,
    get_model,
    resolve_model_name,
)
from src.models.base import compute_class_weights_dict
from src.utils.config import CLASS_NAMES, CUSTOM_CLASS_WEIGHTS, DATA_VERSION, N_CLASSES

# Małe konfiguracje — szybki trening na kilkuset wierszach
FAST_PARAMS: dict[str, dict] = {
    "logreg": {"max_iter": 500},
    "lightgbm": {"n_estimators": 40, "num_leaves": 7, "min_child_samples": 5},
    "xgboost": {"n_estimators": 40, "max_depth": 3, "early_stopping_rounds": 10},
    "catboost": {"iterations": 40, "depth": 4, "od_wait": 10, "verbose": 0},
    "random_forest": {"n_estimators": 30, "min_samples_leaf": 2},
    "ebm": {"max_rounds": 200, "outer_bags": 2, "early_stopping_rounds": 20},
}

EXPECTED_MODELS = ["logreg", "lightgbm", "xgboost", "catboost", "random_forest", "ebm"]


@pytest.fixture(scope="module")
def data(make_xy):
    """Train / early stopping / test z syntetycznego zbioru (klasy 0–4, NaN w danych)."""
    X, y = make_xy(n=500, seed=1)
    X_es, y_es = X.iloc[400:450], y.iloc[400:450]
    X_te = X.iloc[450:]
    return X.iloc[:400], y.iloc[:400], X_es, y_es, X_te


@pytest.fixture(scope="module")
def fitted(data, tmp_path_factory):
    """Każdy model trenowany raz na moduł (logi do katalogu tymczasowego)."""
    from src.utils import config

    X, y, X_es, y_es, _ = data
    logs = tmp_path_factory.mktemp("model_logs")
    models = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "TRAINING_LOGS_DIR", logs / "training")
        mp.setattr(config, "EXPERIMENTS_DIR", logs / "experiments")
        for name in EXPECTED_MODELS:
            model = get_model(name, params=FAST_PARAMS[name])
            model.fit(X, y, X_val=X_es, y_val=y_es, run_id=f"test_{name}", feature_set="triage")
            models[name] = model
    return models


# ─────────────────────────────────────────
# Rejestr
# ─────────────────────────────────────────
class TestRegistry:
    def test_registry_contents_and_order(self):
        assert list(MODEL_REGISTRY) == EXPECTED_MODELS

    @pytest.mark.parametrize("removed", ["extra_trees", "hist_gbt", "stacking", "fusion"])
    def test_removed_models_absent(self, removed):
        assert removed not in MODEL_REGISTRY
        with pytest.raises(ValueError, match="Nieznany model"):
            get_model(removed)

    @pytest.mark.parametrize("alias, name", list(MODEL_ALIASES.items()))
    def test_aliases(self, alias, name):
        assert resolve_model_name(alias) == name
        assert resolve_model_name(alias.upper()) == name
        assert get_model(alias).name == name

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Nieznany model"):
            get_model("xxx-nonexistent-xxx")

    @pytest.mark.parametrize("name", EXPECTED_MODELS)
    def test_instances(self, name):
        model = get_model(name)
        assert isinstance(model, BaseTriageModel)
        assert model.name == name
        assert not model.is_fitted

    @pytest.mark.parametrize("name", EXPECTED_MODELS)
    def test_custom_params_override_defaults(self, name):
        model = get_model(name, params=FAST_PARAMS[name])
        for key, value in FAST_PARAMS[name].items():
            assert model.params[key] == value

    def test_single_thread_in_tests(self):
        assert get_model("lightgbm").params["n_jobs"] == 1
        assert get_model("catboost").params["thread_count"] == 1

    def test_device_cpu_in_tests(self):
        if os.environ.get("SOR_AI_DEVICE") != "cpu":
            pytest.skip("SOR_AI_DEVICE nadpisane przez użytkownika")
        assert get_model("xgboost").params["device"] == "cpu"
        assert get_model("catboost").params["task_type"] == "CPU"


# ─────────────────────────────────────────
# Trening i predykcje
# ─────────────────────────────────────────
@pytest.mark.parametrize("name", EXPECTED_MODELS)
class TestEachModel:
    def test_fitted_state(self, fitted, data, name):
        model = fitted[name]
        assert model.is_fitted
        assert model.feature_names == list(data[0].columns)
        assert model.feature_set == "triage"
        np.testing.assert_array_equal(np.asarray(model.classes_).ravel(), np.arange(N_CLASSES))
        assert model.training_duration_s is not None and model.training_duration_s >= 0

    def test_predict_proba_shape_and_sum(self, fitted, data, name):
        X_te = data[4]
        proba = fitted[name].predict_proba(X_te)
        assert proba.shape == (len(X_te), N_CLASSES)
        assert np.all(proba >= 0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_consistent_with_proba(self, fitted, data, name):
        X_te = data[4]
        model = fitted[name]
        preds = np.asarray(model.predict(X_te)).ravel()
        assert preds.shape == (len(X_te),)
        assert set(preds.tolist()) <= set(range(N_CLASSES))
        # Klasa przewidziana = argmax prawdopodobieństw (z tolerancją na remisy)
        proba = model.predict_proba(X_te)
        chosen = proba[np.arange(len(preds)), preds]
        np.testing.assert_allclose(chosen, proba.max(axis=1), atol=1e-6)

    def test_learns_signal(self, fitted, data, name):
        """Syntetyczne dane mają sygnał — model musi być lepszy niż stała predykcja."""
        X, y = data[0], data[1]
        proba = fitted[name].predict_proba(X)
        assert (proba.argmax(axis=1) == y.to_numpy()).mean() > 0.3

    def test_save_load_roundtrip(self, fitted, data, name, tmp_path):
        model = fitted[name]
        X_te = data[4]
        path = model.save(tmp_path / f"{name}.joblib")
        assert path.exists()

        artifact = joblib.load(path)
        for key in ["name", "run_id", "params", "model", "feature_names", "feature_set",
                    "classes_", "class_names", "label", "data_version", "training_duration_s"]:
            assert key in artifact, key
        assert artifact["name"] == name
        assert artifact["label"] == "esi"
        assert artifact["class_names"] == CLASS_NAMES
        assert artifact["data_version"] == DATA_VERSION

        loaded = MODEL_REGISTRY[artifact["name"]].load(path)
        assert loaded.is_fitted
        assert loaded.feature_names == model.feature_names
        assert loaded.feature_set == "triage"
        np.testing.assert_allclose(loaded.predict_proba(X_te), model.predict_proba(X_te), rtol=0, atol=1e-12)

    def test_feature_importances(self, fitted, data, name):
        imp = fitted[name].feature_importances()
        if imp is None:
            return
        assert isinstance(imp, pd.DataFrame)
        assert list(imp.columns) == ["feature", "importance"]
        assert set(imp["feature"]) <= set(data[0].columns)
        assert imp["importance"].is_monotonic_decreasing

    def test_logs_and_experiment_json_written(self, fitted, name):
        model = fitted[name]
        assert model.train_log_path is not None and model.train_log_path.exists()
        assert model.tracker is not None and model.tracker.json_path.exists()


class TestFitWithoutEarlyStoppingSet:
    @pytest.mark.parametrize("name", ["logreg", "lightgbm", "xgboost", "random_forest"])
    def test_fit_without_val(self, data, name):
        X, y, _, _, X_te = data
        model = get_model(name, params=FAST_PARAMS[name]).fit(X, y, run_id=f"noval_{name}")
        assert model.predict_proba(X_te).shape == (len(X_te), N_CLASSES)

    def test_unfitted_raises(self, data):
        with pytest.raises(RuntimeError):
            get_model("logreg").predict_proba(data[4])
        with pytest.raises(RuntimeError, match="nie został wytrenowany"):
            get_model("lightgbm").save()

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BaseTriageModel.load(tmp_path / "brak.joblib")


class TestSampleWeightStrategies:
    @pytest.mark.parametrize("strategy", ["none", "balanced", "sqrt_balanced", "custom"])
    def test_fit_with_strategy(self, data, strategy):
        X, y, X_es, y_es, X_te = data
        model = get_model("lightgbm", params=FAST_PARAMS["lightgbm"])
        model.fit(X, y, X_val=X_es, y_val=y_es, sample_weight_strategy=strategy, run_id=f"sw_{strategy}")
        assert model.predict_proba(X_te).shape == (len(X_te), N_CLASSES)


# ─────────────────────────────────────────
# Wagi sampli
# ─────────────────────────────────────────
class TestComputeSampleWeights:
    Y = np.array([0, 1, 1, 2, 2, 2, 2, 3, 3, 4])

    def test_none(self):
        w = compute_sample_weights(self.Y, strategy="none")
        assert w.shape == (len(self.Y),)
        assert (w == 1.0).all()

    def test_balanced(self):
        w = compute_sample_weights(self.Y, strategy="balanced")
        # n / (k · n_c): ESI 1 (1 przypadek) = 10 / (5·1) = 2.0; ESI 3 (4) = 0.5
        assert w[0] == pytest.approx(2.0)
        assert w[3] == pytest.approx(0.5)
        # Suma wag na klasę równa
        per_class = pd.Series(w).groupby(self.Y).sum()
        assert per_class.nunique() == 1

    def test_sqrt_balanced(self):
        w = compute_sample_weights(self.Y, strategy="sqrt_balanced")
        np.testing.assert_allclose(w, np.sqrt(compute_sample_weights(self.Y, strategy="balanced")))

    def test_custom_uses_config(self):
        w = compute_sample_weights(self.Y, strategy="custom")
        expected = np.array([CUSTOM_CLASS_WEIGHTS[int(c)] for c in self.Y])
        np.testing.assert_array_equal(w, expected)
        assert w[0] > w[3]   # ESI 1 ważniejsze niż ESI 3

    def test_class_weights_override_strategy(self):
        w = compute_sample_weights(self.Y, strategy="balanced", class_weights={0: 7.0, 4: 3.0})
        assert w[0] == 7.0
        assert w[-1] == 3.0
        assert w[3] == 1.0   # brak w słowniku → 1.0

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Nieznana strategia"):
            compute_sample_weights(self.Y, strategy="magic")

    def test_accepts_series(self):
        w = compute_sample_weights(pd.Series(self.Y), strategy="balanced")
        assert len(w) == len(self.Y)

    def test_class_weights_dict(self):
        d = compute_class_weights_dict(self.Y, strategy="balanced")
        assert set(d) == {0, 1, 2, 3, 4}
        assert d[0] == pytest.approx(2.0)
