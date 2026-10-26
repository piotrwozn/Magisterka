"""
Testy z PRAWDZIWYMI bibliotekami TabPFN 9.2 / TabICL 2.2 na maleńkich sieciach z LOSOWYMI wagami
(`tests/tfm_testkit.py`; checkpointy budowane w katalogu tymczasowym, bez sieci) — pomijane, gdy torch /
tabpfn / tabicl nie są zainstalowane (główne środowisko bez requirements-tfm.txt).

Predykcje losowych sieci są bezwartościowe — sprawdzana jest mechanika: wrappery (worki, predict_proba_many,
porcje), zapis → odczyt → identyczne predykcje, przywracanie ustawień procesu TabPFN, drabina OOM z prawdziwym
`torch.OutOfMemoryError`, brak wag → komunikat z poleceniem 13_tfm_setup, mały wybór konfiguracji i run_training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.tfm_testkit import (
    HAS_TABICL,
    HAS_TABPFN,
    TABICL_TINY_NAME,
    TABPFN_TINY_NAME,
    requires_tabicl,
    requires_tabpfn,
    site_strata,
    triage_frame,
    write_tiny_tabicl,
    write_tiny_tabpfn,
)

pytestmark = pytest.mark.skipif(not (HAS_TABPFN or HAS_TABICL), reason="brak torch / tabpfn / tabicl")
LIBS = [pytest.param("tabpfn", marks=requires_tabpfn), pytest.param("tabicl", marks=requires_tabicl)]


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    import torch

    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("tfm_ckpt")
    return {
        "tabpfn": write_tiny_tabpfn(root) if HAS_TABPFN else None,
        "tabicl": write_tiny_tabicl(root) if HAS_TABICL else None,
    }


@pytest.fixture(autouse=True)
def real_tfm(monkeypatch, checkpoints):
    monkeypatch.delenv("SOR_AI_TFM_FAKE", raising=False)
    monkeypatch.setenv("SOR_AI_DEVICE", "cpu")
    for var in ("SOR_AI_DEVICE_TABPFN", "SOR_AI_DEVICE_TABICL", "TABPFN_MODEL_CACHE_SIZE"):
        monkeypatch.delenv(var, raising=False)
    if checkpoints["tabpfn"] is not None:
        monkeypatch.setenv("SOR_AI_TABPFN_MODEL_PATH", str(checkpoints["tabpfn"]))
    if checkpoints["tabicl"] is not None:
        monkeypatch.setenv("SOR_AI_TABICL_MODEL_PATH", str(checkpoints["tabicl"]))
    from src.utils import config

    for name in ("TABPFN_DEFAULT_PARAMS", "TABICL_DEFAULT_PARAMS"):
        monkeypatch.setitem(getattr(config, name), "n_jobs", 2)


@pytest.fixture(scope="module")
def frame():
    return triage_frame(n=420, seed=5)


def fit(name, X, y, strata=None, **params):
    from src.models import get_model

    model = get_model(name, params={"n_estimators": 1, "n_jobs": 2, **params})
    model.context_strata = strata
    model.fit_estimator(X, y)
    return model


@pytest.mark.parametrize("name", LIBS)
def test_fit_predict_many_through_wrapper(frame, name):
    X, y, site = frame
    model = fit(name, X.iloc[:320], y.iloc[:320], site_strata(y.iloc[:320], site[:320]), n_ctx=150, n_bags=2)
    a, b = model.predict_proba_many([X.iloc[320:370], X.iloc[370:]])
    assert a.shape == (50, 5) and b.shape == (50, 5)
    np.testing.assert_allclose(np.vstack([a, b]).sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.vstack([a, b]), model.predict_proba(X.iloc[320:]), atol=1e-6)
    info = model.tfm_info()
    expected = TABPFN_TINY_NAME if name == "tabpfn" else TABICL_TINY_NAME
    assert info["weights"]["file"] == expected and len(info["weights"]["sha256"]) == 64
    assert info["weights_override"].startswith(f"SOR_AI_{name.upper()}_MODEL_PATH=")
    assert info["library_version"] == ("9.2.0" if name == "tabpfn" else "2.2.0")
    assert info["fake"] is False and info["throughput"]["rows_per_s"] > 0
    assert info["context_sizes"] == [150, 150]


@pytest.mark.parametrize("name", LIBS)
def test_save_load_predict_equal(frame, name, tmp_path):
    import joblib

    from src.models import load_model

    X, y, _ = frame
    model = fit(name, X.iloc[:320], y.iloc[:320], n_ctx=200, context_strategy="balanced")
    proba = model.predict_proba(X.iloc[320:])
    path = model.save(tmp_path / f"{name}.joblib")
    assert joblib.load(path)["model"]._fitted == {}                    # bez obiektów torch w pickle
    loaded = load_model(path)
    np.testing.assert_allclose(loaded.predict_proba(X.iloc[320:]), proba, atol=1e-6)
    assert loaded.tfm_events == model.tfm_events


@requires_tabicl
def test_tabicl_max_features_and_cache_modes(frame):
    X, y, _ = frame
    reference = fit("tabicl", X.iloc[:320], y.iloc[:320], n_ctx=200, kv_cache=False).predict_proba(X.iloc[320:])
    for mode in ("kv", "repr"):
        cached = fit("tabicl", X.iloc[:320], y.iloc[:320], n_ctx=200, kv_cache=mode, batch_size=1)
        assert cached.model.runtime["kv_cache"] == mode
        np.testing.assert_allclose(cached.predict_proba(X.iloc[320:]), reference, atol=1e-5)
    reduced = fit("tabicl", X.iloc[:320], y.iloc[:320], n_ctx=200, max_features=20)
    assert len(reduced.model.kept_features) == 20
    assert reduced.predict_proba(X.iloc[320:]).shape == (100, 5)


@requires_tabpfn
def test_tabpfn_settings_restored_and_oom_ladder(frame, monkeypatch):
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.settings import settings

    X, y, _ = frame
    before = (settings.tabpfn.max_batched_test_rows, settings.tabpfn.max_batched_estimator_rows)
    model = fit("tabpfn", X.iloc[:320], y.iloc[:320], n_ctx=150, predict_batch=2048)
    reference = model.predict_proba(X.iloc[320:])
    assert (settings.tabpfn.max_batched_test_rows, settings.tabpfn.max_batched_estimator_rows) == before
    assert "TABPFN_MODEL_CACHE_SIZE" not in os.environ
    original = TabPFNClassifier.predict_proba

    def flaky(self, X_):
        if len(X_) > 1024:
            raise torch.OutOfMemoryError("CUDA out of memory (test)")
        return original(self, X_)

    monkeypatch.setattr(TabPFNClassifier, "predict_proba", flaky)
    rows = X.iloc[320:].sample(n=1500, replace=True, random_state=0)
    out = model.predict_proba(rows)
    steps = [e["step"] for e in model.tfm_events if e["kind"] == "oom_downgrade"]
    assert steps == ["predict_batch 2048 → 1024 (max_batched_test_rows 1024)"]
    np.testing.assert_allclose(out[:5], model.predict_proba(rows.iloc[:5]), atol=1e-6)
    assert reference.shape == (100, 5)


@requires_tabpfn
def test_tabpfn_missing_weights_message(monkeypatch, tmp_path):
    from tabpfn.settings import settings

    from src.models.tabpfn_model import TabPFNAdapter
    from src.models.tfm_base import TFMWeightsError
    from src.utils import config

    monkeypatch.delenv("SOR_AI_TABPFN_MODEL_PATH")
    monkeypatch.setattr(settings.tabpfn, "model_cache_dir", tmp_path / "pusto")
    with pytest.raises(TFMWeightsError, match="13_tfm_setup.py"):
        TabPFNAdapter().resolve_weights(config.TABPFN_DEFAULT_PARAMS)


@requires_tabicl
def test_tabicl_missing_checkpoint_message(monkeypatch, tmp_path):
    import huggingface_hub.constants as hf_constants

    from src.models.tabicl_model import TabICLAdapter
    from src.models.tfm_base import TFMWeightsError
    from src.utils import config

    monkeypatch.delenv("SOR_AI_TABICL_MODEL_PATH")
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    with pytest.raises(TFMWeightsError, match="13_tfm_setup.py"):
        TabICLAdapter().resolve_weights(config.TABICL_DEFAULT_PARAMS)


@pytest.mark.parametrize("name", LIBS)
def test_tiny_config_selection(frame, name, tmp_path):
    from src.models.tfm_tuning import select_config

    X, y, site = frame
    budget = {"n_trials": 2, "tune_rows": 0, "n_folds": 1, "timeout_s": 600, "holdout_rows": 100,
              "candidates": [{"n_ctx": 120, "n_estimators": 1}, {"n_ctx": 200, "n_estimators": 1}],
              "bags_stage": False, "final_budget_s": 600, "max_bags": 1}
    result = select_config(name, X, y.to_numpy(), strata=site_strata(y, site), budget=budget, profile="smoke",
                           feature_set="triage", storage_dir=tmp_path, data_fp="fpreal")
    assert result["n_complete"] == 2 and result["kind"] == "tfm_config_selection"
    assert result["throughput"]["rows_per_s"] > 0
    assert result["weights"]["file"] in (TABPFN_TINY_NAME, TABICL_TINY_NAME)
    cached = [json.loads(p.read_text(encoding="utf-8")) for p in Path(tmp_path).glob(f"tfm_{name}_*.json")]
    assert sum(1 for c in cached if c["kind"] == "tfm_candidate") == 2


@pytest.mark.parametrize("name", LIBS)
def test_run_training_real_library(prepared_paths, name):
    from src.models.train import run_training

    result = run_training(name, tune=True, profile="smoke")
    info = json.loads((Path(result["run_dir"]) / "run_info.json").read_text(encoding="utf-8"))
    assert info["pretrained"] is True and info["tag"] == "smoke" and info["tfm"]["fake"] is False
    assert info["tfm"]["weights_override"].startswith(f"SOR_AI_{name.upper()}_MODEL_PATH=")
    assert info["tuning"]["kind"] == "tfm_config_selection"
    assert np.isfinite(result["metrics"]["val_argmax_log_loss"])


@requires_tabicl
def test_weights_mismatch_at_predict_is_reported(frame, monkeypatch, tmp_path):
    from src.models import load_model

    X, y, _ = frame
    model = fit("tabicl", X.iloc[:320], y.iloc[:320], n_ctx=150)
    path = model.save(tmp_path / "icl.joblib")
    other = write_tiny_tabicl(tmp_path / "other", seed=1)              # inne losowe wagi
    monkeypatch.setenv("SOR_AI_TABICL_MODEL_PATH", str(other))
    loaded = load_model(path)
    loaded.predict_proba(X.iloc[320:330])
    assert [e["kind"] for e in loaded.tfm_events].count("weights_mismatch") == 1
    loaded.predict_proba(X.iloc[330:340])                              # zgłoszone raz na proces
    assert [e["kind"] for e in loaded.tfm_events].count("weights_mismatch") == 1


@pytest.mark.parametrize("name", LIBS)
def test_cache_on_host_keeps_predictions_and_bags(frame, name):
    """Bufory poza VRAM (bake-off): TabPFN keep_cache_on_device=False, TabICL kv → repr — te same predykcje,
    wszystkie worki dopasowane raz (warm_up), pomiar 1 wiersza działa."""
    X, y, _ = frame
    params = {"n_ctx": 150, "n_bags": 2, **({"kv_cache": "kv", "batch_size": 1} if name == "tabicl" else {})}
    reference = fit(name, X.iloc[:320], y.iloc[:320], **params).predict_proba(X.iloc[320:])
    model = fit(name, X.iloc[:320], y.iloc[:320], **params)
    assert model.model.reencodes_context() is False
    assert model.model.cache_on_host("test") is not None
    model.keep_fitted_bags = "all"
    model.model.warm_up()
    assert len(model.model._fitted) == 2
    assert model.model.probe_latency(1) > 0
    np.testing.assert_allclose(model.predict_proba(X.iloc[320:]), reference, atol=1e-5)
    assert len(model.model._fitted) == 2                                    # bez ponownego dopasowania
    assert model.model.reencodes_context() is (name == "tabicl")           # repr koduje kontekst przy wywołaniu
    model.release()


@requires_tabicl
def test_tabicl_fit_oom_on_last_rung_retries_fit(frame, monkeypatch):
    """Prawdziwy TabICL: OOM przy `model_.to(device)` w fit na ostatnim szczeblu → ponowne dopasowanie
    (wcześniej: predykcja na częściowo dopasowanym estymatorze → RuntimeError „Cannot predict…”)."""
    import torch

    X, y, _ = frame
    params = {"n_ctx": 150, "kv_cache": False, "batch_size": 1, "predict_batch": 4096}
    reference = fit("tabicl", X.iloc[:320], y.iloc[:320], **params).predict_proba(X.iloc[320:])
    model = fit("tabicl", X.iloc[:320], y.iloc[:320], **params)
    original = torch.nn.Module.to
    state = {"raised": False}

    def flaky_to(self, *args, **kwargs):
        if not state["raised"] and type(self).__name__ == "TabICL":
            state["raised"] = True
            raise torch.OutOfMemoryError("CUDA out of memory (test: model_.to)")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "to", flaky_to)
    out = model.predict_proba(X.iloc[320:])
    assert state["raised"]
    np.testing.assert_allclose(out, reference, atol=1e-6)
    event = [e for e in model.tfm_events if e["kind"] == "oom_downgrade"][0]
    assert event["phase"] == "fit" and event["step"] == "predict_batch 4096 → 2048"
