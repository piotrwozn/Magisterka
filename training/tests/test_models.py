"""
Testy modeli ESI (`src.models`): wszystkie 6 modeli z rejestru.

Każdy model na małym numerycznym zbiorze z brakami (NaN) musi:
    - wytrenować się (z early stopping na osobnym zbiorze),
    - zwrócić prawdopodobieństwa (n, 5) float64 sumujące się do 1 (indeks 0–4 = ESI 1–5),
      także gdy klasy brakowało w treningu,
    - po zapisie i odczycie dawać identyczne prawdopodobieństwa (z metadanymi urządzenia),
    - zwrócić ważności cech jako DataFrame[feature, importance] albo None.

`fit_estimator()` (tuning / CV): cichy trening bez logów, early stopping tylko w modelach,
które go wspierają, i ponowna próba na CPU po błędzie GPU (symulowanym — testy na CPU).
"""

from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager

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
    load_model,
    resolve_model_name,
)
from src.models.base import compute_class_weights_dict, full_class_proba
from src.utils import config
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
EARLY_STOPPING_MODELS = {"lightgbm", "xgboost", "catboost"}

ARTIFACT_KEYS = [
    "name", "run_id", "params", "model", "feature_names", "feature_set", "classes_", "class_names",
    "label", "data_version", "training_duration_s", "requested_device", "actual_device",
    "device_fallback", "best_iteration", "iteration_cap", "hardware", "saved_at",
]


@pytest.fixture(scope="module", autouse=True)
def _cpu_only():
    """Testy zawsze na CPU, także gdy powłoka eksportuje SOR_AI_DEVICE_* (np. z run_all)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SOR_AI_DEVICE", "cpu")
        for var in ("SOR_AI_DEVICE_XGBOOST", "SOR_AI_DEVICE_CATBOOST"):
            mp.delenv(var, raising=False)
        yield


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


@contextmanager
def capture_logs(logger_name: str):
    """Rekordy loggera projektu (loggery mają propagate=False, więc caplog ich nie widzi)."""
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collector(level=logging.DEBUG)
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def _warnings(records: list[logging.LogRecord]) -> str:
    return "\n".join(r.getMessage() for r in records if r.levelno >= logging.WARNING)


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
        assert os.environ.get("SOR_AI_DEVICE") == "cpu"
        assert get_model("xgboost").params["device"] == "cpu"
        assert get_model("catboost").params["task_type"] == "CPU"
        assert get_model("xgboost").requested_device == "cpu"


class TestDefaultParams:
    """Domyślne parametry z config (sekcja HARDWARE kontraktu)."""

    def test_iteration_caps(self):
        assert config.XGB_DEFAULT_PARAMS["n_estimators"] == 10_000
        assert config.LGBM_DEFAULT_PARAMS["n_estimators"] == 20_000
        assert config.CATBOOST_DEFAULT_PARAMS["iterations"] == 20_000

    def test_random_forest(self):
        p = config.RF_DEFAULT_PARAMS
        assert (p["n_estimators"], p["min_samples_leaf"], p["max_samples"], p["max_features"]) == (
            500, 5, 0.5, "sqrt")

    def test_ebm(self):
        p = config.EBM_DEFAULT_PARAMS
        assert (p["interactions"], p["outer_bags"], p["max_rounds"]) == (0, 14, 5000)

    def test_logreg(self):
        assert config.LOGREG_DEFAULT_PARAMS["C"] == 1.0
        assert config.LOGREG_DEFAULT_PARAMS["max_iter"] == 2000

    def test_n_jobs_is_positive_int(self):
        assert isinstance(config.N_JOBS, int) and config.N_JOBS >= 1
        assert config.CATBOOST_GPU_RAM_PART == 0.85


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
        assert model.actual_device == "cpu"
        assert model.device_fallback is None

    def test_predict_proba_shape_and_sum(self, fitted, data, name):
        X_te = data[4]
        proba = fitted[name].predict_proba(X_te)
        assert proba.shape == (len(X_te), N_CLASSES)
        assert proba.dtype == np.float64
        assert np.all(proba >= 0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)

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

    def test_predict_proba_uses_training_column_order(self, fitted, data, name):
        X_te = data[4]
        shuffled = X_te[list(reversed(X_te.columns))]
        np.testing.assert_array_equal(fitted[name].predict_proba(shuffled), fitted[name].predict_proba(X_te))
        with pytest.raises(KeyError, match="Brak 1 cech"):
            fitted[name].predict_proba(X_te.drop(columns=["age"]))

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
        assert not list(tmp_path.glob("*.tmp"))          # zapis atomowy — bez śmieci
        assert path.read_bytes()[:1] == b"\x78"           # kompresja zlib (joblib compress=3)

        artifact = joblib.load(path)
        for key in ARTIFACT_KEYS:
            assert key in artifact, key
        assert artifact["name"] == name
        assert artifact["label"] == "esi"
        assert artifact["class_names"] == CLASS_NAMES
        assert artifact["data_version"] == DATA_VERSION
        assert artifact["actual_device"] == "cpu"
        assert artifact["device_fallback"] is None
        assert artifact["best_iteration"] == model.best_iteration
        assert artifact["iteration_cap"] == model.iteration_cap
        assert artifact["hardware"]["physical_cores"] >= 1

        for loaded in (MODEL_REGISTRY[artifact["name"]].load(path), load_model(path),
                       BaseTriageModel.load(path)):
            assert type(loaded) is type(model)
            assert loaded.is_fitted
            assert loaded.feature_names == model.feature_names
            assert loaded.feature_set == "triage"
            assert loaded.params == model.params
            assert loaded.data_version == DATA_VERSION
            assert loaded.actual_device == "cpu"
            assert loaded.device_fallback is None
            assert loaded.best_iteration == model.best_iteration
            assert loaded.iteration_cap == model.iteration_cap
            assert loaded.hardware == artifact["hardware"]
            np.testing.assert_allclose(loaded.predict_proba(X_te), model.predict_proba(X_te), rtol=0, atol=1e-12)

    def test_feature_importances(self, fitted, data, name):
        imp = fitted[name].feature_importances()
        if imp is None:
            return
        assert isinstance(imp, pd.DataFrame)
        assert list(imp.columns) == ["feature", "importance"]
        # logreg: także wskaźniki braków (missingindicator_<cecha>)
        base_names = {f.removeprefix("missingindicator_") for f in imp["feature"]}
        assert base_names <= set(data[0].columns)
        assert imp["importance"].is_monotonic_decreasing

    def test_logs_and_experiment_json_written(self, fitted, name):
        model = fitted[name]
        assert model.train_log_path is not None and model.train_log_path.exists()
        assert model.tracker is not None and model.tracker.json_path.exists()

    def test_training_logger_closed_after_fit(self, fitted, name):
        # pytest dopina własne LogCaptureHandler do loggerów bez propagacji — pomijamy je
        ours = [h for h in fitted[name].train_logger.handlers if not type(h).__module__.startswith("_pytest")]
        assert ours == []

    def test_best_iteration(self, fitted, name):
        model = fitted[name]
        if name in EARLY_STOPPING_MODELS:
            assert model.iteration_cap == FAST_PARAMS[name].get("n_estimators", FAST_PARAMS[name].get("iterations"))
            assert 1 <= model.best_iteration <= model.iteration_cap
        elif name == "ebm":
            assert model.iteration_cap == FAST_PARAMS["ebm"]["max_rounds"]
            assert model.best_iteration is None or 1 <= model.best_iteration <= model.iteration_cap
        else:
            assert model.best_iteration is None


class TestFitWithoutEarlyStoppingSet:
    @pytest.mark.parametrize("name", ["logreg", "lightgbm", "xgboost", "random_forest"])
    def test_fit_without_val(self, data, name):
        X, y, _, _, X_te = data
        model = get_model(name, params=FAST_PARAMS[name]).fit(X, y, run_id=f"noval_{name}")
        assert model.predict_proba(X_te).shape == (len(X_te), N_CLASSES)
        assert model.best_iteration is None

    def test_unfitted_raises(self, data):
        with pytest.raises(RuntimeError):
            get_model("logreg").predict_proba(data[4])
        with pytest.raises(RuntimeError, match="nie został wytrenowany"):
            get_model("lightgbm").save()

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BaseTriageModel.load(tmp_path / "brak.joblib")
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "brak.joblib")


class TestSampleWeightStrategies:
    @pytest.mark.parametrize("strategy", ["none", "balanced", "sqrt_balanced", "custom"])
    def test_fit_with_strategy(self, data, strategy):
        X, y, X_es, y_es, X_te = data
        model = get_model("lightgbm", params=FAST_PARAMS["lightgbm"])
        model.fit(X, y, X_val=X_es, y_val=y_es, sample_weight_strategy=strategy, run_id=f"sw_{strategy}")
        assert model.predict_proba(X_te).shape == (len(X_te), N_CLASSES)


# ─────────────────────────────────────────
# fit_estimator — cichy trening (tuning / CV)
# ─────────────────────────────────────────
@pytest.mark.parametrize("name", EXPECTED_MODELS)
class TestFitEstimator:
    def test_quiet_with_early_stopping_data(self, data, name, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "TRAINING_LOGS_DIR", tmp_path / "training")
        monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path / "experiments")
        X, y, X_es, y_es, X_te = data
        model = get_model(name, params=FAST_PARAMS[name])
        result = model.fit_estimator(X, y, X_es, y_es, sample_weight=np.ones(len(y)))

        assert result is None
        assert model.is_fitted
        assert model.feature_names == list(X.columns)
        np.testing.assert_array_equal(model.classes_, np.arange(N_CLASSES))
        assert model.actual_device == "cpu" and model.device_fallback is None
        assert model.training_duration_s >= 0
        assert model.tracker is None and model.train_logger is None and model.run_id is None
        assert not (tmp_path / "training").exists() and not (tmp_path / "experiments").exists()
        proba = model.predict_proba(X_te)
        assert proba.shape == (len(X_te), N_CLASSES) and proba.dtype == np.float64
        if name in EARLY_STOPPING_MODELS:
            assert model.best_iteration is not None and model.best_iteration <= model.iteration_cap

    def test_without_early_stopping_data(self, data, name):
        X, y, _, _, X_te = data
        model = get_model(name, params=FAST_PARAMS[name])
        model.fit_estimator(X, y)
        assert model.is_fitted
        assert model.predict_proba(X_te).shape == (len(X_te), N_CLASSES)
        if name in EARLY_STOPPING_MODELS:
            assert model.best_iteration is None
            assert model.iteration_cap is not None

    def test_fit_delegates_to_fit_estimator(self, data, name, monkeypatch):
        X, y, X_es, y_es, _ = data
        model = get_model(name, params=FAST_PARAMS[name])
        calls = []
        original = model.fit_estimator

        def spy(*args, **kwargs):
            calls.append(kwargs.get("sample_weight"))
            return original(*args, **kwargs)

        monkeypatch.setattr(model, "fit_estimator", spy)
        model.fit(X, y, X_val=X_es, y_val=y_es, sample_weight_strategy="balanced", run_id=f"spy_{name}")
        assert len(calls) == 1
        np.testing.assert_allclose(calls[0], compute_sample_weights(y, strategy="balanced"))


@pytest.mark.parametrize("name", ["logreg", "random_forest", "ebm"])
def test_models_without_early_stopping_ignore_es_data(data, name):
    X, y, X_es, y_es, X_te = data
    with_es = get_model(name, params=FAST_PARAMS[name])
    with_es.fit_estimator(X, y, X_es, y_es)
    without = get_model(name, params=FAST_PARAMS[name])
    without.fit_estimator(X, y)
    np.testing.assert_allclose(with_es.predict_proba(X_te), without.predict_proba(X_te), rtol=0, atol=1e-12)
    assert with_es.best_iteration == without.best_iteration


def test_iteration_cap_warning(data):
    X, y, X_es, y_es, _ = data
    model = get_model("xgboost", params={"n_estimators": 5, "max_depth": 3, "early_stopping_rounds": 50,
                                         "learning_rate": 0.05})
    with capture_logs("src.models.base") as records:
        model.fit_estimator(X, y, X_es, y_es)
    assert model.best_iteration == 5 and model.iteration_cap == 5
    assert "limitu 5" in _warnings(records)

    model.fit(X, y, X_val=X_es, y_val=y_es, run_id="cap_warning")
    log_text = model.train_log_path.read_text(encoding="utf-8")
    assert "WARNING" in log_text and "niedotrenowany" in log_text


def test_early_stopping_rounds_override(data):
    X, y, X_es, y_es, _ = data
    model = get_model("lightgbm", params={**FAST_PARAMS["lightgbm"], "n_estimators": 400,
                                          "learning_rate": 0.3})
    model.fit(X, y, X_val=X_es, y_val=y_es, early_stopping_rounds=3, run_id="es_override")
    assert model.es_rounds == 3
    assert model.best_iteration < 400


# ─────────────────────────────────────────
# Klasa nieobecna w treningu
# ─────────────────────────────────────────
@pytest.mark.parametrize("name", EXPECTED_MODELS)
@pytest.mark.parametrize("missing", [0, 4])
def test_predict_proba_with_class_missing_from_training(data, name, missing):
    X, y, X_es, y_es, X_te = data
    keep = (y != missing).to_numpy()
    model = get_model(name, params=FAST_PARAMS[name])
    model.fit_estimator(X.loc[keep], y.loc[keep], X_es, y_es)   # ES z wszystkimi klasami

    assert missing not in set(np.asarray(model.classes_).tolist())
    proba = model.predict_proba(X_te)
    assert proba.shape == (len(X_te), N_CLASSES)
    assert proba.dtype == np.float64
    np.testing.assert_array_equal(proba[:, missing], 0.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
    assert missing not in set(model.predict(X_te).tolist())


class TestFullClassProba:
    def test_fills_missing_and_renormalizes(self):
        raw = np.array([[0.2, 0.3, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32)
        out = full_class_proba(raw, [1, 2, 4])
        assert out.dtype == np.float64 and out.shape == (2, N_CLASSES)
        np.testing.assert_allclose(out[0], [0, 0.2, 0.3, 0, 0.5])
        np.testing.assert_allclose(out[1], np.full(N_CLASSES, 0.2))     # zero → jednostajny

    def test_shape_mismatch_raises(self):
        with pytest.raises(RuntimeError, match="kolumn"):
            full_class_proba(np.ones((2, 3)), [0, 1])


# ─────────────────────────────────────────
# GPU → CPU (symulowane błędy GPU)
# ─────────────────────────────────────────
CUDA_35 = ("catboost/cuda/cuda_lib/cuda_base.h:265: CUDA error 35: "
           "CUDA driver version is insufficient for CUDA runtime version")


def _fail_first_call(monkeypatch, cls, exc: Exception) -> dict:
    """Pierwsze cls.fit rzuca `exc`, kolejne wywołują oryginał."""
    original = cls.fit
    state = {"calls": 0}

    def flaky(self, *args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise exc
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, "fit", flaky)
    return state


class TestGpuFallback:
    def test_xgboost_gpu_error_retries_on_cpu(self, data, monkeypatch):
        import xgboost as xgb

        X, y, X_es, y_es, X_te = data
        state = _fail_first_call(monkeypatch, xgb.XGBClassifier,
                                 xgb.core.XGBoostError("cudaErrorInsufficientDriver: CUDA error 35"))
        model = get_model("xgboost", params={**FAST_PARAMS["xgboost"], "device": "cuda"})
        assert model.requested_device == "gpu"
        with capture_logs("src.models.base") as records:
            model.fit_estimator(X, y, X_es, y_es)
        assert state["calls"] == 2
        assert model.params["device"] == "cpu"
        assert model.actual_device == "cpu"
        assert "CUDA error 35" in model.device_fallback
        assert "PONAWIAM TRENING NA CPU" in _warnings(records)
        assert model.requested_device == "gpu"
        assert model.predict_proba(X_te).shape == (len(X_te), N_CLASSES)

    def test_catboost_gpu_error_retries_on_cpu(self, data, monkeypatch, tmp_path):
        from catboost import CatBoostClassifier, CatBoostError

        X, y, X_es, y_es, X_te = data
        state = _fail_first_call(monkeypatch, CatBoostClassifier, CatBoostError(CUDA_35))
        model = get_model("catboost", params={**FAST_PARAMS["catboost"], "task_type": "GPU"})
        assert model.params["devices"] == "0" and model.params["gpu_ram_part"] == 0.85
        assert model.requested_device == "gpu"
        with capture_logs("src.models.base") as records:
            model.fit(X, y, X_val=X_es, y_val=y_es, run_id="cb_fallback")
        assert state["calls"] == 2
        assert model.params["task_type"] == "CPU"
        assert "devices" not in model.params and "gpu_ram_part" not in model.params
        assert model.actual_device == "cpu"
        assert "CUDA error 35" in model.device_fallback
        assert "PONAWIAM" in _warnings(records)
        # Tracker i log treningu znają faktyczne urządzenie i powód
        assert model.tracker.data["device"] == {
            "requested": "gpu", "actual": "cpu", "fallback_reason": model.device_fallback,
        }
        assert "GPU → CPU" in model.train_log_path.read_text(encoding="utf-8")
        loaded = load_model(model.save(tmp_path / "cb.joblib"))
        assert loaded.device_fallback == model.device_fallback
        assert loaded.requested_device == "gpu" and loaded.actual_device == "cpu"

    def test_non_gpu_error_is_not_retried(self, data, monkeypatch):
        import xgboost as xgb

        X, y, X_es, y_es, _ = data
        state = _fail_first_call(monkeypatch, xgb.XGBClassifier, ValueError("zły kształt danych"))
        model = get_model("xgboost", params={**FAST_PARAMS["xgboost"], "device": "cuda"})
        with pytest.raises(ValueError, match="zły kształt"):
            model.fit_estimator(X, y, X_es, y_es)
        assert state["calls"] == 1
        assert not model.is_fitted

    def test_cpu_model_gpu_like_error_is_not_retried(self, data, monkeypatch):
        from catboost import CatBoostClassifier, CatBoostError

        X, y, X_es, y_es, _ = data
        state = _fail_first_call(monkeypatch, CatBoostClassifier, CatBoostError(CUDA_35))
        model = get_model("catboost", params={**FAST_PARAMS["catboost"], "task_type": "CPU"})
        with pytest.raises(CatBoostError):
            model.fit_estimator(X, y, X_es, y_es)
        assert state["calls"] == 1

    def test_xgboost_silent_cpu_training_is_detected(self, data):
        """Bez widocznego GPU xgboost z device='cuda' trenuje na CPU bez wyjątku — wykrywamy to."""
        from src.utils.hardware import probe_gpu

        if probe_gpu("xgboost")[0]:
            pytest.skip("Na tej maszynie xgboost naprawdę trenuje na GPU")
        X, y, X_es, y_es, X_te = data
        model = get_model("xgboost", params={**FAST_PARAMS["xgboost"], "device": "cuda"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_estimator(X, y, X_es, y_es)
        assert model.actual_device == "cpu"
        assert model.device_fallback and "CPU" in model.device_fallback
        assert model.model.get_params()["device"] == "cpu"


class TestDeviceParams:
    def test_xgboost_predicts_on_cpu_after_fit(self, fitted):
        assert fitted["xgboost"].model.get_params()["device"] == "cpu"

    def test_catboost_random_state_override(self):
        model = get_model("catboost", params={"random_state": 7, "n_jobs": 3})
        assert model.params["random_seed"] == 7
        assert model.params["thread_count"] == 3
        assert "random_state" not in model.params and "n_jobs" not in model.params

    def test_catboost_native_name_wins_over_alias(self):
        model = get_model("catboost", params={"random_state": 7, "random_seed": 11})
        assert model.params["random_seed"] == 11

    def test_catboost_gpu_drops_rsm_and_alias(self):
        gpu = get_model("catboost", params={"task_type": "GPU", "rsm": 0.5, "colsample_bylevel": 0.5,
                                            "bootstrap_type": "MVS"})
        assert "rsm" not in gpu.params and "colsample_bylevel" not in gpu.params
        assert gpu.params["bootstrap_type"] == "Bernoulli"
        assert gpu.params["gpu_ram_part"] == config.CATBOOST_GPU_RAM_PART
        cpu = get_model("catboost", params={"task_type": "CPU", "rsm": 0.5, "devices": "0"})
        assert cpu.params["rsm"] == 0.5
        assert "devices" not in cpu.params

    def test_catboost_quiet_and_verbose(self, data, capfd):
        X, y, X_es, y_es, _ = data
        model = get_model("catboost", params={"iterations": 20, "depth": 3, "verbose": 1})
        model.fit_estimator(X, y, X_es, y_es)
        assert "learn:" not in capfd.readouterr().out          # fit_estimator = cicho
        model.fit(X, y, X_val=X_es, y_val=y_es, verbose=5, run_id="cb_verbose")
        assert "learn:" in capfd.readouterr().out               # fit() = postęp co 5 iteracji

    def test_catboost_thread_budget_in_pools_and_predict(self, data, monkeypatch):
        """Pool (trening i ES) i predict_proba dostają budżet wątków modelu, nie domyślne -1 (wszystkie)."""
        import catboost
        from catboost import CatBoostClassifier

        X, y, X_es, y_es, X_te = data
        pools: list = []
        real_pool = catboost.Pool

        def spy_pool(*args, **kwargs):
            pools.append(kwargs.get("thread_count"))
            return real_pool(*args, **kwargs)

        predict_threads: list = []
        real_predict = CatBoostClassifier.predict_proba

        def spy_predict(self, X, *args, **kwargs):
            predict_threads.append(kwargs.get("thread_count"))
            return real_predict(self, X, *args, **kwargs)

        monkeypatch.setattr(catboost, "Pool", spy_pool)
        monkeypatch.setattr(CatBoostClassifier, "predict_proba", spy_predict)
        model = get_model("catboost", params={**FAST_PARAMS["catboost"], "thread_count": 2})
        model.fit_estimator(X, y, X_es, y_es)
        model.predict_proba(X_te)
        expected = min(2, os.cpu_count() or 2)
        assert pools == [expected, expected]                     # trening + early stopping
        assert predict_threads == [expected]
        model.params["thread_count"] = None                      # brak budżetu → -1 (CatBoost: wszystkie)
        assert model._thread_count() == -1

    def test_save_retries_locked_target(self, fitted, tmp_path, monkeypatch):
        """Windows: stary model.joblib chwilowo zablokowany (backend / antywirus) — zapis ponawiany."""
        from src.utils import fileio

        real_replace = os.replace
        attempts: list = []

        def flaky(src, dst):
            attempts.append(dst)
            if len(attempts) < 3:
                raise PermissionError(13, "plik używany przez inny proces")
            return real_replace(src, dst)

        monkeypatch.setattr(fileio.os, "replace", flaky)
        monkeypatch.setattr(fileio.time, "sleep", lambda s: None)
        path = fitted["logreg"].save(tmp_path / "model.joblib")
        assert len(attempts) == 3 and joblib.load(path)["name"] == "logreg"
        assert [p.name for p in tmp_path.iterdir()] == ["model.joblib"]

    def test_save_failure_cleans_tmp(self, fitted, tmp_path, monkeypatch):
        from src.utils import fileio

        def locked(src, dst):
            raise PermissionError(13, "zablokowany")

        monkeypatch.setattr(fileio.os, "replace", locked)
        monkeypatch.setattr(fileio.time, "sleep", lambda s: None)
        with pytest.raises(PermissionError):
            fitted["logreg"].save(tmp_path / "model.joblib")
        assert list(tmp_path.iterdir()) == []

    def test_lightgbm_no_deprecation_warning(self, data):
        import lightgbm as lgb

        X, y, X_es, y_es, _ = data
        model = get_model("lightgbm", params=FAST_PARAMS["lightgbm"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", getattr(lgb.basic, "LGBMDeprecationWarning", DeprecationWarning))
            model.fit_estimator(X, y, X_es, y_es)
        assert model.best_iteration is not None

    def test_early_stopping_curve_named_es(self, fitted):
        for name in EARLY_STOPPING_MODELS:
            history = fitted[name].tracker.data["training_history"]
            keys = set().union(*(row.keys() for row in history)) - {"iteration"}
            assert any(k.startswith("es_") for k in keys), (name, keys)
            assert not any(k.startswith("val_") for k in keys), (name, keys)


# ─────────────────────────────────────────
# Regresja logistyczna — wskaźniki braków
# ─────────────────────────────────────────
class TestLogreg:
    def test_missing_indicators(self, fitted, data):
        model = fitted["logreg"]
        imputer = model.model.named_steps["impute"]
        assert imputer.add_indicator and imputer.keep_empty_features
        names = model.transformed_feature_names()
        assert names[: len(model.feature_names)] == model.feature_names
        assert "missingindicator_triage_vital_hr" in names
        assert len(names) == model.model.named_steps["clf"].coef_.shape[1]
        imp = model.feature_importances()
        assert set(imp["feature"]) == set(names)

    def test_empty_column_kept(self, data):
        X, y, _, _, X_te = data
        X = X.copy()
        X["triage_vital_temp"] = np.nan
        model = get_model("logreg", params=FAST_PARAMS["logreg"])
        model.fit_estimator(X, y)
        assert model.predict_proba(X_te).shape == (len(X_te), N_CLASSES)


# ─────────────────────────────────────────
# Odczyt artefaktów
# ─────────────────────────────────────────
class TestLoad:
    def test_wrong_class_raises(self, fitted, tmp_path):
        path = fitted["lightgbm"].save(tmp_path / "lgbm.joblib")
        with pytest.raises(ValueError, match="'lightgbm'"):
            MODEL_REGISTRY["xgboost"].load(path)

    def test_not_an_artifact(self, tmp_path):
        path = tmp_path / "x.joblib"
        joblib.dump({"foo": 1}, path)
        with pytest.raises(ValueError, match="nie jest artefaktem"):
            load_model(path)

    def test_load_does_not_probe_gpu(self, fitted, tmp_path, monkeypatch):
        from src.utils import hardware

        path = fitted["catboost"].save(tmp_path / "cb.joblib")
        monkeypatch.setenv("SOR_AI_DEVICE", "gpu")   # wymuszone GPU bez karty → device_for rzuciłby błąd
        monkeypatch.setattr(hardware, "probe_gpu", lambda lib: pytest.fail("load() nie może próbować GPU"))
        loaded = load_model(path)
        assert loaded.params["task_type"] == "CPU"

    def test_legacy_artifact_without_new_keys(self, fitted, tmp_path, data):
        model = fitted["random_forest"]
        legacy = {k: v for k, v in joblib.load(model.save(tmp_path / "rf.joblib")).items()
                  if k in ARTIFACT_KEYS[:11]}
        joblib.dump(legacy, tmp_path / "legacy.joblib")
        loaded = load_model(tmp_path / "legacy.joblib")
        assert loaded.actual_device is None and loaded.best_iteration is None
        np.testing.assert_allclose(loaded.predict_proba(data[4]), model.predict_proba(data[4]))


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
