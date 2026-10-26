"""
Testy wrapperów modeli fundamentalnych (TFM: tabpfn, tabicl) z ATRAPĄ estymatora (`SOR_AI_TFM_FAKE=1`).

Atrapa (src/models/tfm_fake.py) przechodzi tę samą ścieżkę kodu co prawdziwa biblioteka — testy działają
bez torch (główne środowisko) i sprawdzają: worki kontekstu (podział / losowanie, ≥ 2 wiersze każdej klasy,
warstwy ESI × oddział), dokładną korektę priorów strategii "balanced", regułę `max_features`,
`predict_proba_many`, `keep_fitted_bags`, drabinę awaryjną OOM, strażnika CPU, zapis / odczyt artefaktu
(haki `_artifact_extra` / `_restore_extra`, estymatory nigdy w pickle), przywracanie ustawień procesu
TabPFN, haki przekierowania wag oraz próbę GPU "torch" w hardware.py.
"""

from __future__ import annotations

import logging
import os
import pickle
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models import MODEL_REGISTRY, BaseTriageModel, get_model, load_model, tfm_base, tfm_fake
from src.models.tabicl_model import TabICLAdapter, TabICLTriageModel
from src.models.tabpfn_model import TabPFNAdapter, TabPFNTriageModel
from src.models.tfm_base import (
    TFMDeviceError,
    TFMOutOfMemoryError,
    TFMPredictor,
    TFMWeightsError,
    build_bags,
    check_cpu_context,
    is_tfm_oom,
    planned_max_context,
    resolve_n_bags,
    select_features,
)
from src.utils import config, hardware
from src.utils.config import N_CLASSES
from tests.tfm_testkit import site_strata, triage_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TFM_NAMES = ("tabpfn", "tabicl")
ARTIFACT_KEYS = [
    "name", "run_id", "params", "model", "feature_names", "feature_set", "classes_", "class_names",
    "label", "data_version", "training_duration_s", "requested_device", "actual_device",
    "device_fallback", "best_iteration", "iteration_cap", "hardware", "saved_at",
]


@pytest.fixture(autouse=True)
def fake_tfm(monkeypatch):
    """Atrapa estymatora, CPU, bez przekierowań wag i symulacji OOM; liczniki wywołań od zera."""
    monkeypatch.setenv("SOR_AI_TFM_FAKE", "1")
    monkeypatch.setenv("SOR_AI_DEVICE", "cpu")
    for var in ("SOR_AI_DEVICE_TABPFN", "SOR_AI_DEVICE_TABICL", "SOR_AI_TABPFN_MODEL_PATH",
                "SOR_AI_TABICL_MODEL_PATH", tfm_fake.FAKE_OOM_ENV, "TABPFN_MODEL_CACHE_SIZE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(tfm_fake, "FAKE_OOM", None)
    tfm_fake.reset_calls()
    yield
    tfm_fake.reset_calls()


@pytest.fixture(scope="module")
def frame():
    X, y, site = triage_frame(n=700, seed=3)
    return X, y, site


def fit_tfm(name: str, X: pd.DataFrame, y, strata=None, **params):
    model = get_model(name, params={"n_jobs": 1, **params})
    model.context_strata = strata
    model.fit_estimator(X, y)
    return model


@contextmanager
def capture_logs(logger_name: str):
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


# ─────────────────────────────────────────
# Rejestr, import bez torch, parametry
# ─────────────────────────────────────────
class TestRegistration:
    def test_import_does_not_load_torch(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import src.models; "
                "from src.models import get_model; get_model('pfn'); get_model('icl'); "
                "print('torch' in sys.modules, 'tabpfn' in sys.modules, 'tabicl' in sys.modules)")
        env = {**os.environ, "SOR_AI_DEVICE": "cpu"}
        out = subprocess.run([sys.executable, "-c", code, str(PROJECT_ROOT)], env=env, capture_output=True,
                             text=True, encoding="utf-8", timeout=120, check=True).stdout
        assert out.strip().splitlines()[-1] == "False False False"

    @pytest.mark.parametrize("name, cls", [("tabpfn", TabPFNTriageModel), ("tabicl", TabICLTriageModel)])
    def test_class_attributes_and_defaults(self, name, cls):
        assert MODEL_REGISTRY[name] is cls
        assert cls.pretrained is True and cls.supports_early_stopping is False
        model = get_model(name)
        defaults = getattr(config, f"{name.upper()}_DEFAULT_PARAMS")
        assert {k: v for k, v in model.params.items() if k != "device"} == defaults
        assert model.params["device"] == "cpu" and model.requested_device == "cpu"
        assert model.feature_importances() is None
        assert model.keep_fitted_bags == 1 and model.context_strata is None

    def test_profiles_have_tfm_budgets(self):
        for spec in config.TUNING_PROFILES.values():
            for name in TFM_NAMES:
                budget = spec[name]
                assert budget["n_trials"] == len(budget["candidates"]) >= 1
                assert budget["tune_rows"] == 0 and budget["n_folds"] == 1
                assert budget["timeout_s"] > 0 and budget["final_budget_s"] > 0 and budget["holdout_rows"] > 0
        smoke = config.TUNING_PROFILES["smoke"]
        assert all(c["n_ctx"] <= config.TFM_CPU_MAX_CONTEXT for m in TFM_NAMES for c in smoke[m]["candidates"])
        assert config.TFM_CPU_MAX_CONTEXT == 5000
        thorough = config.TUNING_PROFILES["thorough"]
        assert not any(c["n_ctx"] > 100_000 for c in thorough["tabpfn"]["candidates"])
        assert any(c.get("context_strategy") == "balanced" for c in thorough["tabicl"]["candidates"])
        assert any(c.get("max_features") == 100 for c in thorough["tabicl"]["candidates"])
        assert thorough["tabpfn"]["max_bags"] == 4 and thorough["tabpfn"]["bags_stage"]
        assert config.TUNING_PROFILES["quick"]["tabpfn"]["max_bags"] == 1


# ─────────────────────────────────────────
# Worki kontekstu
# ─────────────────────────────────────────
class TestBags:
    def test_partition_uses_all_rows_when_enough_bags(self):
        y = np.resize(np.arange(5), 1000)
        bags, events = build_bags(y, None, n_ctx=300, n_bags=4, bag_mode="partition", seed=1)
        assert [len(b) for b in bags] == [250] * 4
        union = np.concatenate(bags)
        assert len(union) == len(np.unique(union)) == 1000        # rozłączne, wszystkie wiersze
        assert events == []

    def test_partition_subset_is_disjoint_and_stratified(self):
        rng = np.random.default_rng(0)
        y = rng.choice(5, size=5000, p=[0.05, 0.25, 0.4, 0.2, 0.1])
        site = rng.choice(np.array(["A", "B"]), size=5000, p=[0.7, 0.3])
        bags, _ = build_bags(y, site_strata(y, site), n_ctx=800, n_bags=3, bag_mode="partition", seed=7)
        assert [len(b) for b in bags] == [800, 800, 800]
        union = np.concatenate(bags)
        assert len(np.unique(union)) == 2400
        overall = np.bincount(y, minlength=5) / len(y)
        for bag in bags:
            share = np.bincount(y[bag], minlength=5) / len(bag)
            np.testing.assert_allclose(share, overall, atol=0.01)
            assert abs((site[bag] == "B").mean() - (site == "B").mean()) < 0.02

    def test_resample_independent_and_deterministic(self):
        y = np.resize(np.arange(5), 2000)
        a, _ = build_bags(y, None, n_ctx=500, n_bags=3, bag_mode="resample", seed=3)
        b, _ = build_bags(y, None, n_ctx=500, n_bags=3, bag_mode="resample", seed=3)
        c, _ = build_bags(y, None, n_ctx=500, n_bags=3, bag_mode="resample", seed=4)
        assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))
        assert not np.array_equal(a[0], c[0])
        assert not np.array_equal(a[0], a[1])                       # worki niezależne
        assert all(len(bag) == 500 for bag in a)

    def test_every_bag_has_two_rows_of_each_class(self):
        y = np.array([0, 0, 0] + [1] * 200 + [2] * 300 + [3] * 100 + [4] * 2)
        bags, events = build_bags(y, None, n_ctx=200, n_bags=4, bag_mode="partition", seed=5)
        for bag in bags:
            counts = np.bincount(y[bag], minlength=5)
            assert counts[0] >= 2 and counts[4] >= 2
        assert {e["kind"] for e in events} == {"rare_class_reuse"}
        assert any(e["reused_from_other_bags"] > 0 for e in events)
        sizes_before = [len(b) for b in build_bags(np.resize(np.arange(1, 5), len(y)), None, n_ctx=200, n_bags=4,
                                                  seed=5)[0]]
        assert [len(b) for b in bags] == sizes_before               # dobór w miejsce wierszy klasy licznej

    def test_class_with_single_row_needs_only_one(self):
        y = np.array([0] + [2] * 50 + [3] * 50)
        bags, _ = build_bags(y, None, n_ctx=40, n_bags=2, seed=0)
        assert all((y[b] == 0).sum() == 1 for b in bags)

    def test_balanced_caps_each_class(self):
        y = np.repeat(np.arange(5), [10, 300, 500, 150, 40])
        bags, _ = build_bags(y, None, n_ctx=200, n_bags=1, context_strategy="balanced", seed=0)
        assert np.bincount(y[bags[0]], minlength=5).tolist() == [10, 50, 50, 50, 40]

    def test_n_bags_auto_and_validation(self):
        assert resolve_n_bags("auto", 446_421, 50_000, 4) == 4
        assert resolve_n_bags("auto", 60_000, 50_000, 4) == 2
        assert resolve_n_bags("auto", 1_000, 50_000, 4) == 1
        assert resolve_n_bags(3, 10, 5, 1) == 3
        with pytest.raises(ValueError):
            resolve_n_bags("dużo", 10, 5, 2)
        with pytest.raises(ValueError):
            resolve_n_bags(0, 10, 5, 2)
        with pytest.raises(ValueError, match="bag_mode"):
            build_bags(np.zeros(10, int), None, n_ctx=5, n_bags=1, bag_mode="bootstrap")
        with pytest.raises(ValueError, match="context_strata"):
            build_bags(np.zeros(10, int), np.zeros(3), n_ctx=5, n_bags=1)

    def test_model_stores_self_contained_context(self, frame):
        X, y, site = frame
        model = fit_tfm("tabpfn", X, y, site_strata(y, site), n_ctx=150, n_bags="auto", max_bags=3)
        p = model.model
        assert isinstance(p, TFMPredictor) and p.n_bags == 3
        assert p.context_sizes() == [150, 150, 150]
        assert p.X_ctx.dtype == np.float32 and len(p.y_ctx) == 450   # tylko wiersze worków
        assert p.n_fit == len(X)
        assert model.params["n_bags"] == "auto"                     # "auto" dosłownie (refit przeliczy)
        np.testing.assert_allclose(p.prior_fit, np.bincount(y, minlength=5) / len(y))

    def test_context_strata_length_checked(self, frame):
        X, y, _ = frame
        with pytest.raises(ValueError, match="context_strata"):
            fit_tfm("tabpfn", X, y, np.array(["a"] * 3), n_ctx=100)


# ─────────────────────────────────────────
# Korekta priorów, max_features
# ─────────────────────────────────────────
class TestPriorsAndFeatures:
    @pytest.mark.parametrize("strategy", ["balanced", "stratified"])
    def test_prior_correction_exact(self, frame, strategy):
        X, y, _ = frame
        model = fit_tfm("tabicl", X.iloc[:500], y.iloc[:500], n_ctx=60, context_strategy=strategy)
        proba = model.predict_proba(X.iloc[500:])
        est = model.model._fitted[0]                     # keep_fitted_bags=1 → estymator worka w pamięci
        raw = np.zeros((200, N_CLASSES))
        raw[:, est.classes_] = est.predict_proba(X.iloc[500:][model.model.kept_features].to_numpy(np.float32))
        if strategy == "balanced":
            pi_fit = np.bincount(y.iloc[:500], minlength=5) / 500
            pi_ctx = model.model.bag_prior(0)
            assert not np.allclose(pi_fit, pi_ctx)                    # kontekst faktycznie zrównoważony
            raw = raw * pi_fit / pi_ctx
        expected = raw / raw.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(proba, expected, atol=1e-12)
        assert model.model.correction is (strategy == "balanced")

    def test_select_features_rule(self, frame):
        X, _, _ = frame
        names = list(X.columns)
        protected = tfm_base.protected_features(names)
        assert {"cc_count", "cc_critical", "cc_group_cardiac", "cc_cardiacarrest", "cc_strokealert"} <= set(protected)
        assert {"hr_missing", "triage_vital_hr", "age", "arrival_ambulance", "news2_score"} <= set(protected)
        limit = len(protected) + 4
        kept, info = select_features(names, X, limit)
        assert len(kept) == limit and kept == [n for n in names if n in kept]       # kolejność wejścia
        assert [n for n in kept if n not in protected] == ["cc_flag00", "cc_flag01", "cc_flag02", "cc_flag03"]
        assert info["dropped"] == [f"cc_flag{i:02d}" for i in range(4, 12)]
        assert select_features(names, X, None) == (names, None)
        assert select_features(names, X, 10_000)[1] is None

    def test_select_features_protected_exceed_limit(self, frame):
        X, _, _ = frame
        with capture_logs("src.models.tfm_base") as records:
            kept, info = select_features(list(X.columns), X, 5)
        assert kept == tfm_base.protected_features(list(X.columns))
        assert info["n_kept"] == len(kept) > 5
        assert any("zostają tylko one" in r.getMessage() for r in records)

    def test_model_with_max_features_accepts_full_input(self, frame):
        X, y, _ = frame
        n_protected = len(tfm_base.protected_features(list(X.columns)))
        model = fit_tfm("tabicl", X.iloc[:500], y.iloc[:500], n_ctx=100, max_features=n_protected + 2)
        assert len(model.model.kept_features) == n_protected + 2
        assert model.feature_names == list(X.columns)                 # wejście = pełna lista cech
        proba = model.predict_proba(X.iloc[500:])
        np.testing.assert_allclose(model.model.predict_proba(X.iloc[500:].to_numpy()), proba, atol=1e-12)
        assert any(e["kind"] == "max_features" for e in model.tfm_events)
        with pytest.raises(KeyError, match="Brak 1 cech"):
            model.predict_proba(X.iloc[500:].drop(columns=["age"]))


# ─────────────────────────────────────────
# predict_proba_many, keep_fitted_bags
# ─────────────────────────────────────────
class TestPrediction:
    def test_many_equals_single_and_fits_each_bag_once(self, frame):
        X, y, _ = frame
        model = fit_tfm("tabpfn", X.iloc[:500], y.iloc[:500], n_ctx=120, n_bags=3)
        model.keep_fitted_bags = 0
        tfm_fake.reset_calls()
        a, b = model.predict_proba_many([X.iloc[500:600], X.iloc[600:]])
        assert tfm_fake.FAKE_CALLS["fit"] == 3                       # jeden fit na worek dla obu ramek
        single = model.predict_proba(X.iloc[500:])
        np.testing.assert_allclose(np.vstack([a, b]), single, atol=1e-12)
        assert a.shape == (100, 5) and a.dtype == np.float64
        np.testing.assert_allclose(a.sum(axis=1), 1.0, atol=1e-12)
        assert model.tfm_predict_s is not None

    def test_base_default_predict_proba_many(self, frame):
        X, y, _ = frame
        model = get_model("logreg", params={"max_iter": 300})
        model.fit_estimator(X.iloc[:500], y.iloc[:500])
        many = model.predict_proba_many([X.iloc[500:550], X.iloc[550:]])
        np.testing.assert_array_equal(many[0], model.predict_proba(X.iloc[500:550]))
        np.testing.assert_array_equal(many[1], model.predict_proba(X.iloc[550:]))
        assert BaseTriageModel.pretrained is False and model._artifact_extra() == {}

    def test_batches_do_not_change_predictions(self, frame):
        X, y, _ = frame
        big = fit_tfm("tabpfn", X.iloc[:500], y.iloc[:500], n_ctx=200, predict_batch=10_000)
        small = fit_tfm("tabpfn", X.iloc[:500], y.iloc[:500], n_ctx=200, predict_batch=7)
        tfm_fake.reset_calls()
        np.testing.assert_allclose(small.predict_proba(X.iloc[500:]), big.predict_proba(X.iloc[500:]), atol=1e-12)
        assert tfm_fake.FAKE_CALLS["predict"] == int(np.ceil(200 / 7)) + 1

    def test_keep_fitted_bags(self, frame):
        X, y, _ = frame
        one = fit_tfm("tabpfn", X.iloc[:500], y.iloc[:500], n_ctx=200)           # 1 worek, keep=1
        one.predict_proba(X.iloc[500:510])
        tfm_fake.reset_calls()
        one.predict_proba(X.iloc[510:520])
        assert tfm_fake.FAKE_CALLS["fit"] == 0                           # worek z pamięci

        many = fit_tfm("tabpfn", X.iloc[:500], y.iloc[:500], n_ctx=100, n_bags=3)
        many.keep_fitted_bags = "all"
        many.model.warm_up()
        assert len(many.model._fitted) == 3
        tfm_fake.reset_calls()
        many.predict_proba(X.iloc[500:])
        assert tfm_fake.FAKE_CALLS["fit"] == 0
        many.keep_fitted_bags = 0
        many.model.release()
        many.predict_proba(X.iloc[500:])
        assert tfm_fake.FAKE_CALLS["fit"] == 3 and many.model._fitted == {}
        many.keep_fitted_bags = "dwa"
        with pytest.raises(ValueError, match="keep_fitted_bags"):
            many.predict_proba(X.iloc[500:])

    def test_limit_bags_compacts_context(self, frame):
        X, y, _ = frame
        model = fit_tfm("tabicl", X.iloc[:500], y.iloc[:500], n_ctx=100, n_bags=4)
        before = model.predict_proba(X.iloc[500:])
        model.limit_bags(2, "test budżetu")
        assert model.n_bags == 2 and len(model.model.y_ctx) == 200
        assert model.tfm_events[-1]["kind"] == "bags_reduced"
        after = model.predict_proba(X.iloc[500:])
        assert after.shape == before.shape and not np.allclose(after, before)


# ─────────────────────────────────────────
# OOM: rozpoznanie i drabina awaryjna
# ─────────────────────────────────────────
class TabPFNCUDAOutOfMemoryError(Exception):
    """Jak tabpfn.errors.TabPFNCUDAOutOfMemoryError — NIE RuntimeError."""


class TestOOM:
    @pytest.mark.parametrize("exc, expected", [
        (tfm_fake.FakeOutOfMemoryError("x"), True),
        (TabPFNCUDAOutOfMemoryError("CUDA"), True),
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), True),
        (RuntimeError("CPU memory allocation failed (x) and disk offload is not available"), True),
        (RuntimeError("Failed to execute even with minimum batch size 1. Error: x"), True),
        (ValueError("Input contains NaN"), False),
        (RuntimeError("CUDA error: device-side assert triggered"), False),
        (KeyboardInterrupt(), False),
    ])
    def test_is_tfm_oom(self, exc, expected):
        assert is_tfm_oom(exc) is expected

    def test_torch_oom_class(self):
        torch = pytest.importorskip("torch")
        assert is_tfm_oom(torch.OutOfMemoryError("x")) and is_tfm_oom(torch.cuda.OutOfMemoryError("y"))

    def test_tabpfn_halves_predict_batch(self, frame, monkeypatch):
        X, y, _ = frame
        reference = fit_tfm("tabpfn", X.iloc[:400], y.iloc[:400], n_ctx=150).predict_proba(X.iloc[400:])
        model = fit_tfm("tabpfn", X.iloc[:400], y.iloc[:400], n_ctx=150, predict_batch=4096)
        monkeypatch.setattr(tfm_fake, "FAKE_OOM", {"predict_rows_gt": 1500})
        rows = pd.concat([X.iloc[400:]] * 12)                          # 3600 wierszy
        out = model.predict_proba(rows)
        np.testing.assert_allclose(out[:300], reference, atol=1e-12)    # wynik niezależny od porcji
        steps = [e["step"] for e in model.tfm_events if e["kind"] == "oom_downgrade"]
        assert steps == ["predict_batch 4096 → 2048 (max_batched_test_rows 2048)",
                         "predict_batch 2048 → 1024 (max_batched_test_rows 1024)"]
        assert model.model.runtime["predict_batch"] == 1024
        assert model.params["predict_batch"] == 4096                    # params bez zmian (refit dziedziczy)

    def test_tabpfn_fit_oom_moves_cache_off_device(self, frame, monkeypatch):
        X, y, _ = frame
        model = fit_tfm("tabpfn", X.iloc[:400], y.iloc[:400], n_ctx=150)
        monkeypatch.setattr(tfm_fake, "FAKE_OOM", {"fit_when": {"keep_cache_on_device": True}})
        seen = []
        original = tfm_fake.FakeTFMClassifier.fit

        def spy(self, X_, y_):
            seen.append(tfm_fake.FAKE_SETTINGS.tabpfn.max_batched_estimator_rows)
            return original(self, X_, y_)

        monkeypatch.setattr(tfm_fake.FakeTFMClassifier, "fit", spy)
        model.predict_proba(X.iloc[400:])
        event = [e for e in model.tfm_events if e["kind"] == "oom_downgrade"][0]
        assert event["phase"] == "fit" and event["rebuild"] and "keep_cache_on_device=False" in event["step"]
        assert seen == [None, 0]                                        # 2. próba: jeden bufor naraz
        assert tfm_fake.FAKE_SETTINGS.tabpfn.max_batched_estimator_rows is None   # przywrócone

    def test_tabpfn_ladder_exhausted(self, frame, monkeypatch):
        X, y, _ = frame
        model = fit_tfm("tabpfn", X.iloc[:400], y.iloc[:400], n_ctx=150, predict_batch=2048)
        monkeypatch.setattr(tfm_fake, "FAKE_OOM", {"always": True})
        with pytest.raises(TFMOutOfMemoryError, match="Zmniejsz: n_ctx") as err:
            model.predict_proba(X.iloc[400:])
        steps = [e["step"] for e in model.tfm_events if e["kind"] == "oom_downgrade"]
        assert steps[0].startswith("predict_batch 2048 → 1024")
        assert any("keep_cache_on_device=False" in s for s in steps)
        assert any(s.startswith("fit_mode=fit_preprocessors") for s in steps)
        assert steps[-1] == "predict_batch 2048 → 1024 (max_batched_test_rows 1024)"
        assert is_tfm_oom(err.value)                                    # wybór konfiguracji → status "oom"
        assert tfm_fake.FAKE_SETTINGS.tabpfn.max_batched_test_rows == 32_768

    def test_tabicl_ladder(self, frame, monkeypatch):
        X, y, _ = frame
        model = fit_tfm("tabicl", X.iloc[:400], y.iloc[:400], n_ctx=150, kv_cache="kv", batch_size=4,
                        n_estimators=4)
        assert model.model.runtime["kv_cache"] == "kv" and model.model.runtime["batch_size"] == 4
        monkeypatch.setattr(tfm_fake, "FAKE_OOM", {"predict_when": [{"batch_size": 4}, {"kv_cache": "kv"}]})
        model.predict_proba(X.iloc[400:])
        steps = [e["step"] for e in model.tfm_events if e["kind"] == "oom_downgrade"]
        assert steps == ["batch_size 4 → 1 (jeden estymator na przebieg)", "kv_cache 'kv' → 'repr'"]
        assert model.model.runtime == {**model.model.runtime, "batch_size": 1, "kv_cache": "repr"}

    def test_tabicl_fit_oom_on_last_rung_rebuilds_estimator(self, frame, monkeypatch):
        """
        TabICL na ostatnim szczeblu (batch_size=1, bez bufora): OOM w fazie FIT (np. model_.to(cuda)) →
        krok bez przebudowy (predict_batch / 2), ale estymator i tak budowany od nowa — nie predykcja na
        częściowo dopasowanym estymatorze.
        """
        X, y, _ = frame
        params = {"n_ctx": 150, "kv_cache": False, "batch_size": 1, "n_estimators": 2, "predict_batch": 4096}
        reference = fit_tfm("tabicl", X.iloc[:400], y.iloc[:400], **params).predict_proba(X.iloc[400:])
        model = fit_tfm("tabicl", X.iloc[:400], y.iloc[:400], **params)
        original = tfm_fake.FakeTFMClassifier.fit
        attempts: list = []                                          # referencje (id po GC bywa powtórzone)

        def flaky_fit(self, X_, y_):
            attempts.append(self)
            if len(attempts) == 1:                                    # jednorazowy OOM w fit
                raise tfm_fake.FakeOutOfMemoryError("CUDA out of memory (model_.to)")
            return original(self, X_, y_)

        monkeypatch.setattr(tfm_fake.FakeTFMClassifier, "fit", flaky_fit)
        out = model.predict_proba(X.iloc[400:])
        np.testing.assert_allclose(out, reference, atol=1e-12)
        event = [e for e in model.tfm_events if e["kind"] == "oom_downgrade"][0]
        assert event["phase"] == "fit" and event["step"] == "predict_batch 4096 → 2048"
        assert len(attempts) == 2 and attempts[0] is not attempts[1]  # drugi fit na NOWYM estymatorze

    def test_non_oom_error_propagates(self, frame, monkeypatch):
        X, y, _ = frame
        model = fit_tfm("tabicl", X.iloc[:400], y.iloc[:400], n_ctx=150)

        def broken(self, X_):
            raise ValueError("Input contains infinity")

        monkeypatch.setattr(tfm_fake.FakeTFMClassifier, "predict_proba", broken)
        with pytest.raises(ValueError, match="infinity"):
            model.predict_proba(X.iloc[400:])
        assert not [e for e in model.tfm_events if e["kind"] == "oom_downgrade"]

    def test_fake_oom_env_policy(self, monkeypatch):
        monkeypatch.setenv(tfm_fake.FAKE_OOM_ENV, '{"predict_rows_gt": 3}')
        assert tfm_fake.oom_policy() == {"predict_rows_gt": 3}
        monkeypatch.setenv(tfm_fake.FAKE_OOM_ENV, "{zły json")
        with pytest.raises(ValueError, match=tfm_fake.FAKE_OOM_ENV):
            tfm_fake.oom_policy()


# ─────────────────────────────────────────
# Strażnik CPU
# ─────────────────────────────────────────
class TestCpuGuard:
    def test_large_context_on_cpu_fails_at_fit(self, frame, monkeypatch):
        X, y, _ = frame
        monkeypatch.setattr(config, "TFM_CPU_MAX_CONTEXT", 100)
        with pytest.raises(TFMDeviceError, match="TFM_CPU_MAX_CONTEXT=100") as err:
            fit_tfm("tabpfn", X, y, n_ctx=300)
        assert "download.pytorch.org/whl/cu130" in str(err.value) and "--profile smoke" in str(err.value)
        fit_tfm("tabpfn", X.iloc[:90], y.iloc[:90], n_ctx=300)         # kontekst = 90 wierszy ≤ limit

    def test_switch_to_cpu(self, monkeypatch):
        monkeypatch.setattr(config, "TFM_CPU_MAX_CONTEXT", 1000)
        model = get_model("tabicl", params={"device": "cuda", "n_ctx": 5000})
        with pytest.raises(TFMDeviceError, match="NIE przełączam na CPU"):
            model._switch_to_cpu()
        small = get_model("tabicl", params={"device": "cuda", "n_ctx": 800})
        small._switch_to_cpu()
        assert small.params["device"] == "cpu"

    def test_gpu_device_skips_guard_and_reports_cuda(self, frame, monkeypatch):
        X, y, _ = frame
        monkeypatch.setattr(config, "TFM_CPU_MAX_CONTEXT", 10)
        model = fit_tfm("tabpfn", X, y, n_ctx=200, device="cuda")
        model._after_fit(False)
        assert model.actual_device == "cuda" and model.requested_device == "gpu"
        check_cpu_context("tabpfn", "gpu", 10**6)

    def test_planned_max_context(self):
        budget = dict(config.TUNING_PROFILES["quick"]["tabpfn"], train_sample=60_000)
        assert planned_max_context("tabpfn", budget, None, tune=True) == 10_000
        assert planned_max_context("tabpfn", {**budget, "n_trials": 1}, None, tune=True) == 5_000
        assert planned_max_context("tabpfn", budget, None, tune=False) == 50_000
        assert planned_max_context("tabicl", {**budget, "train_sample": 6_000}, None, tune=False) == 6_000
        assert planned_max_context("tabicl", budget, {"n_ctx": 3_000}, tune=False) == 3_000


# ─────────────────────────────────────────
# Zapis / odczyt artefaktu
# ─────────────────────────────────────────
class TestPersistence:
    @pytest.mark.parametrize("name", TFM_NAMES)
    def test_roundtrip_with_hooks(self, frame, name, tmp_path):
        X, y, site = frame
        model = fit_tfm(name, X.iloc[:500], y.iloc[:500], site_strata(y.iloc[:500], site[:500]), n_ctx=100,
                        n_bags=2)
        proba = model.predict_proba(X.iloc[500:])
        assert model.model._fitted                                     # estymator w pamięci (keep=1)
        path = model.save(tmp_path / f"{name}.joblib")
        artifact = joblib.load(path)
        assert set(ARTIFACT_KEYS) <= set(artifact)
        assert artifact["pretrained"] is True and artifact["tfm"]["library"] == name
        assert artifact["tfm"]["weights_override"] == "SOR_AI_TFM_FAKE=1"
        assert artifact["tfm"]["licence"] == config.TFM_LICENSES[name]
        assert isinstance(artifact["model"], TFMPredictor)
        assert artifact["model"]._fitted == {}                         # estymatory nigdy w pickle
        assert "_adapter" not in artifact["model"].__getstate__()
        for loaded in (load_model(path), BaseTriageModel.load(path), MODEL_REGISTRY[name].load(path)):
            assert type(loaded) is type(model) and loaded.pretrained
            assert loaded.context_strata is None and loaded.keep_fitted_bags == 1
            assert loaded.tfm_events == model.tfm_events and loaded.model.events is loaded.tfm_events
            assert loaded.tfm_predict_s is None
            np.testing.assert_allclose(loaded.predict_proba(X.iloc[500:]), proba, atol=1e-12)
            # backend: bundle["model"].predict_proba na pełnej liście cech
            np.testing.assert_allclose(artifact["model"].predict_proba(X.iloc[500:]), proba, atol=1e-12)

    def test_classical_artifact_unchanged(self, frame, tmp_path):
        X, y, _ = frame
        model = get_model("logreg", params={"max_iter": 200})
        model.fit_estimator(X.iloc[:300], y.iloc[:300])
        artifact = joblib.load(model.save(tmp_path / "lr.joblib"))
        assert sorted(artifact) == sorted(ARTIFACT_KEYS)

    def test_pickle_drops_fitted_estimators(self, frame):
        X, y, _ = frame
        model = fit_tfm("tabicl", X.iloc[:300], y.iloc[:300], n_ctx=100)
        model.keep_fitted_bags = "all"
        model.model.warm_up()
        clone = pickle.loads(pickle.dumps(model.model))
        assert clone._fitted == {} and model.model._fitted
        assert clone.keep_fitted_bags == "all"

    def test_fit_logs_context(self, frame):
        X, y, _ = frame
        model = get_model("tabpfn", params={"n_jobs": 1, "n_ctx": 100, "n_bags": 2})
        model.fit(X.iloc[:300], y.iloc[:300], run_id="tfm_log_test")
        text = model.train_log_path.read_text(encoding="utf-8")
        assert "Worki: 2" in text and "worek 0: klasy ESI 1–5" in text and "SOR_AI_TFM_FAKE=1" in text
        assert model.tracker.json_path.exists()


# ─────────────────────────────────────────
# Ustawienia procesu TabPFN, haki wag
# ─────────────────────────────────────────
class TestSettingsAndHooks:
    def test_tabpfn_process_settings_restored(self, frame, monkeypatch):
        X, y, _ = frame
        model = fit_tfm("tabpfn", X.iloc[:300], y.iloc[:300], n_ctx=100, predict_batch=5000)
        seen = []
        original = tfm_fake.FakeTFMClassifier.predict_proba

        def spy(self, X_):
            seen.append((os.environ.get("TABPFN_MODEL_CACHE_SIZE"), tfm_fake.FAKE_SETTINGS.tabpfn.max_batched_test_rows))
            return original(self, X_)

        monkeypatch.setattr(tfm_fake.FakeTFMClassifier, "predict_proba", spy)
        model.predict_proba(X.iloc[300:])
        assert seen and set(seen) == {("1", 5000)}
        assert "TABPFN_MODEL_CACHE_SIZE" not in os.environ
        assert tfm_fake.FAKE_SETTINGS.tabpfn.max_batched_test_rows == 32_768
        monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "3")                 # ustawienie użytkownika zostaje
        seen.clear()
        model.predict_proba(X.iloc[300:])
        assert {s[0] for s in seen} == {"3"} and os.environ["TABPFN_MODEL_CACHE_SIZE"] == "3"

    def test_weights_override_hooks(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SOR_AI_TFM_FAKE")
        assert tfm_base.weights_override_active("tabpfn") is None
        weights = tmp_path / "tabpfn-v3.5-random.safetensors"
        weights.write_bytes(b"0123456789")
        monkeypatch.setenv("SOR_AI_TABPFN_MODEL_PATH", str(weights))
        adapter = TabPFNAdapter(fake=False)
        ident = adapter.resolve_weights(config.TABPFN_DEFAULT_PARAMS)
        assert ident["file"] == weights.name and ident["size"] == 10 and len(ident["sha256"]) == 64
        assert adapter.weights_override() == f"SOR_AI_TABPFN_MODEL_PATH={weights}"
        assert tfm_base.weights_override_active("tabpfn") == f"SOR_AI_TABPFN_MODEL_PATH={weights}"
        monkeypatch.setenv("SOR_AI_TABICL_MODEL_PATH", str(tmp_path / "brak.ckpt"))
        with pytest.raises(TFMWeightsError, match="nie istnieje"):
            TabICLAdapter(fake=False).resolve_weights(config.TABICL_DEFAULT_PARAMS)
        monkeypatch.setenv("SOR_AI_TFM_FAKE", "1")
        assert tfm_base.weights_override_active("tabicl") == "SOR_AI_TFM_FAKE=1"

    def test_same_weights(self):
        a = {"file": "w", "size": 1, "sha256": "x"}
        assert tfm_base.same_weights(a, dict(a))
        assert not tfm_base.same_weights(a, {**a, "sha256": "y"})
        assert tfm_base.same_weights({"file": "w", "size": 1}, {"file": "w", "size": 1, "sha256": None})

    def test_tabicl_runtime_policy(self, monkeypatch):
        adapter = TabICLAdapter(fake=True)
        params = dict(config.TABICL_DEFAULT_PARAMS)
        small = adapter.initial_runtime(params, n_ctx=1_000, n_features=271, device="cpu")
        assert small["batch_size"] == 4 and small["kv_cache"] == "kv"
        large = adapter.initial_runtime(params, n_ctx=50_000, n_features=271, device="cuda")
        assert large["batch_size"] == 1 and large["kv_cache"] == "repr"           # VRAM nieznana (atrapa)
        assert 1_024 <= large["predict_batch"] <= 65_536
        monkeypatch.setattr(config, "TFM_HOST_RAM_BUDGET_GB", 4.0)
        tight = adapter.initial_runtime({**params, "kv_cache": False}, n_ctx=5_000, n_features=271, device="cpu")
        assert tight["kv_cache"] is False and tight["predict_batch"] == 2_626   # 4 GiB / (4·275·512) − 5000
        # 50k × 8 estymatorów "kv" na 16 GB: 9,56 GiB > 0,35 × 16 GiB → "repr"; 25k × 8 → "kv"
        assert TabICLAdapter.kv_cache_bytes(8, 50_000, 271, "kv", "cuda") > 0.35 * 16 * 2**30
        assert TabICLAdapter.kv_cache_bytes(8, 25_000, 271, "kv", "cuda") < 0.35 * 16 * 2**30
        with pytest.raises(ValueError, match="kv_cache"):
            adapter.initial_runtime({**params, "kv_cache": "wszystko"}, n_ctx=10, n_features=5, device="cpu")

    def test_estimated_cache_bytes(self, frame):
        X, y, _ = frame
        model = fit_tfm("tabpfn", X.iloc[:300], y.iloc[:300], n_ctx=100, n_bags=2, n_estimators=2)
        assert model.model.estimated_cache_bytes() > 0
        assert model.model.estimated_cache_bytes(on_device=True) == model.model.estimated_cache_bytes()

    def test_reencodes_context_and_cache_on_host(self, frame):
        """Bufor kodowany przy każdym wywołaniu (repr / bez bufora / fit_preprocessors) i przeniesienie buforów."""
        X, y, _ = frame
        icl = fit_tfm("tabicl", X.iloc[:300], y.iloc[:300], n_ctx=100, kv_cache="kv", n_estimators=2)
        assert icl.model.reencodes_context() is False
        before = icl.model.estimated_cache_bytes()
        assert icl.model.cache_on_host("test") == ("kv_cache 'kv' → 'repr' (mniejszy bufor; kontekst ICL liczony "
                                                   "przy każdym wywołaniu)")
        assert icl.model.runtime["kv_cache"] == "repr" and icl.model.reencodes_context() is True
        assert icl.model.estimated_cache_bytes() < before and icl.params.get("kv_cache") == "kv"   # params bez zmian
        assert icl.model.cache_on_host("test") is None                         # nic więcej do zrobienia
        assert [e["kind"] for e in icl.tfm_events].count("cache_on_host") == 1
        for kv in ("repr", False):
            assert fit_tfm("tabicl", X.iloc[:300], y.iloc[:300], n_ctx=100, kv_cache=kv,
                           n_estimators=2).model.reencodes_context() is True
        pfn = fit_tfm("tabpfn", X.iloc[:300], y.iloc[:300], n_ctx=100, n_bags=2, n_estimators=2)
        assert pfn.model.reencodes_context() is False
        assert pfn.model.cache_on_host("test").startswith("bufory KV w RAM")
        assert pfn.model.runtime["keep_cache_on_device"] is False and pfn.model.runtime["one_cache_at_a_time"]
        assert pfn.model.estimated_cache_bytes(on_device=True) == 0 and pfn.model.estimated_cache_bytes() > 0
        assert pfn.model.reencodes_context() is False                          # bufor w RAM, nie kodowanie
        np.testing.assert_allclose(pfn.predict_proba(X.iloc[300:350]).sum(axis=1), 1.0)
        slow = fit_tfm("tabpfn", X.iloc[:300], y.iloc[:300], n_ctx=100, fit_mode="fit_preprocessors")
        assert slow.model.reencodes_context() is True and slow.model.cache_on_host("x") is None
        assert slow.model.estimated_cache_bytes() == 0

    def test_probe_latency(self, frame):
        X, y, _ = frame
        model = fit_tfm("tabpfn", X.iloc[:300], y.iloc[:300], n_ctx=100, n_bags=2)
        model.keep_fitted_bags = "all"
        model.model.warm_up()
        assert model.model.probe_latency(1) >= 0.0 and len(model.model._fitted) == 2

    def test_throughput_from_unrounded_times(self, frame, monkeypatch):
        """Predykcja < 0,5 ms: suma zaokrąglonych czasów dawała 0 → rows_per_s=None (niestabilny test wyboru)."""
        X, y, _ = frame
        model = fit_tfm("tabpfn", X.iloc[:300], y.iloc[:300], n_ctx=100)
        ticks = iter(np.arange(0.0, 10.0, 0.0001))                       # każdy pomiar = 0,1 ms
        monkeypatch.setattr(tfm_base, "time", types.SimpleNamespace(perf_counter=lambda: float(next(ticks))))
        model.model.predict_proba(X.iloc[300:550])
        stats = model.model.throughput()
        assert stats["rows_per_s"] == pytest.approx(250 / 0.0001, rel=1e-6)
        assert stats["per_bag"]["0"]["predict_s"] == 0.0                   # opis: zaokrąglony do ms
        assert model.model.stats["bags"][0]["predict_s"] == pytest.approx(0.0001)   # surowy pomiar


# ─────────────────────────────────────────
# hardware.py: próba "torch" i urządzenie TFM
# ─────────────────────────────────────────
class TestHardwareTorch:
    @pytest.fixture
    def clean_env(self, monkeypatch, tmp_path):
        for var in ("SOR_AI_DEVICE", "SOR_AI_DEVICE_TABPFN", "SOR_AI_DEVICE_TABICL"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(hardware, "_probe_cache_path", lambda: tmp_path / "probe.json")
        hardware.probe_gpu.cache_clear()
        hardware._announced.clear()
        yield monkeypatch
        hardware.probe_gpu.cache_clear()
        hardware._announced.clear()

    def test_device_for_tfm_uses_torch_probe(self, clean_env):
        calls = []
        clean_env.setitem(hardware._PROBES, "torch", lambda: (calls.append("torch"), (True, "torch ok"))[1])
        assert hardware.device_for("tabpfn") == "gpu"
        assert hardware.device_for("icl") == "gpu"
        assert calls == ["torch"]                                  # wynik próby z pamięci procesu
        assert hardware.probe_library("tabicl") == "torch" and hardware.probe_library("xgb") == "xgboost"
        clean_env.setenv("SOR_AI_DEVICE_TABPFN", "cpu")
        assert hardware.device_for("tabpfn") == "cpu"
        assert hardware.device_preference("tabpfn") == ("cpu", "SOR_AI_DEVICE_TABPFN")

    def test_forced_gpu_without_cuda_explains_wheel(self, clean_env):
        clean_env.setitem(hardware._PROBES, "torch", lambda: (False, "torch 2.14.1+cpu zbudowany BEZ CUDA"))
        clean_env.setenv("SOR_AI_DEVICE_TABICL", "gpu")
        with pytest.raises(RuntimeError, match="SOR_AI_DEVICE_TABICL=gpu") as err:
            hardware.device_for("tabicl")
        assert "download.pytorch.org/whl/cu130" in str(err.value)
        clean_env.delenv("SOR_AI_DEVICE_TABICL")
        assert hardware.device_for("tabicl") == "cpu"              # auto + porażka → CPU (z ostrzeżeniem)

    @pytest.mark.parametrize("info, ok, fragment", [
        ({"installed": False, "version": None}, False, "nie jest zainstalowany"),
        ({"installed": True, "version": "2.14.1+cpu", "cuda_build": None, "cuda_available": False,
          "error": None}, False, "BEZ CUDA"),
        ({"installed": True, "version": "2.14.1+cu130", "cuda_build": "13.0", "cuda_available": False,
          "error": None}, False, "is_available() == False"),
    ])
    def test_probe_torch_failures(self, clean_env, info, ok, fragment):
        base = {"cuda_build": None, "cuda_available": False, "device_name": None, "total_vram_mb": None,
                "free_vram_mb": None, "error": None}
        clean_env.setattr(hardware, "torch_info", lambda: {**base, **info})
        result, reason = hardware._probe_torch()
        assert result is ok and fragment in reason

    def test_torch_info_never_raises(self):
        info = hardware.torch_info()
        assert set(info) == {"installed", "version", "cuda_build", "cuda_available", "device_name",
                             "total_vram_mb", "free_vram_mb", "error"}

    def test_summary_unchanged_for_classical(self, monkeypatch):
        assert "torch" not in hardware._LIBRARY_DISTS
        monkeypatch.delenv("SOR_AI_DEVICE_TABPFN", raising=False)
        assert "SOR_AI_DEVICE_TABPFN" not in hardware.hardware_summary()["env"]
        monkeypatch.setenv("SOR_AI_DEVICE_TABPFN", "gpu")
        assert hardware.hardware_summary()["env"]["SOR_AI_DEVICE_TABPFN"] == "gpu"
        assert "TabPFN" in hardware._GPU_HINTS["torch"] and "cu130" in hardware._GPU_HINTS["torch"]


def test_tfm_never_chosen_for_shap(frame):
    """SHAP (05_explain, przypadki ESI dla LLM) wybiera tylko modele drzewiaste / liniowe / EBM — nigdy TFM."""
    from src.explain.shap_explainer import detect_method
    from src.llm_eval.esi_cases import SHAP_METHOD_RANK

    X, y, _ = frame
    model = fit_tfm("tabpfn", X.iloc[:300], y.iloc[:300], n_ctx=100)
    assert detect_method(model) == "permutation" and "permutation" not in SHAP_METHOD_RANK
