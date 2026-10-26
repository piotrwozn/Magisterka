"""
Testy wyboru konfiguracji TFM na holdoucie z TRAIN (`src.models.tfm_tuning`) — atrapa estymatora.

Sprawdzane: kształt wyniku (klucze czytane przez train.py / orkiestrator), holdout rozłączny z kontekstem,
reguła wyboru (najniższy log loss; w tolerancji 0,002 tańszy), etap worków, limit czasu (pominięci
kandydaci zalogowani, nie zapisani), błędy / OOM kandydatów, wznawianie z plików kandydatów, klucz
z wersją biblioteki i wagami, blokada, budżet finalnej predykcji, postęp dla --status.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
from contextlib import contextmanager

import numpy as np
import pytest

from src.models import tfm_fake, tfm_tuning
from src.models.tfm_tuning import (
    candidate_cost,
    final_bag_budget,
    holdout_split,
    select_config,
    selection_progress,
)
from src.utils import config
from tests.tfm_testkit import site_strata, triage_frame

REQUIRED_KEYS = {"n_trials", "n_complete", "metric", "kind", "n_fold_train", "best_params", "study_name",
                 "best_trial", "candidates", "throughput"}


@pytest.fixture(autouse=True)
def fake_tfm(monkeypatch):
    monkeypatch.setenv("SOR_AI_TFM_FAKE", "1")
    monkeypatch.setenv("SOR_AI_DEVICE", "cpu")
    monkeypatch.delenv(tfm_fake.FAKE_OOM_ENV, raising=False)
    monkeypatch.setattr(tfm_fake, "FAKE_OOM", None)
    for name in ("TABPFN_DEFAULT_PARAMS", "TABICL_DEFAULT_PARAMS"):
        monkeypatch.setitem(getattr(config, name), "n_jobs", 1)


@pytest.fixture(scope="module")
def data():
    X, y, site = triage_frame(n=900, seed=11)
    return X, y.to_numpy(), site_strata(y, site)


def make_budget(candidates, **extra):
    return {"n_trials": len(candidates), "tune_rows": 0, "n_folds": 1, "timeout_s": 600,
            "candidates": candidates, "holdout_rows": 250, "bags_stage": False, "final_budget_s": 600,
            "max_bags": 1, **extra}


def run_select(data, tmp_path, candidates, model="tabpfn", **extra):
    X, y, strata = data
    return select_config(model, X, y, strata=strata, budget=make_budget(candidates, **extra), profile="smoke",
                         feature_set="triage", storage_dir=tmp_path / "optuna", data_fp="fp123")


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


def scripted(losses: dict[int, float | Exception], calls: list | None = None):
    """Zastępstwo `_evaluate`: log loss wg n_ctx kandydata (albo wyjątek)."""

    def fake_evaluate(model_name, params, X_pool, y_pool, strata_pool, X_ho, y_ho):
        if calls is not None:
            calls.append({"params": dict(params), "pool": X_pool.index, "holdout": X_ho.index})
        value = losses[int(params["n_ctx"])] if params.get("n_bags") == 1 else losses.get(-1, 0.9)
        if isinstance(value, Exception):
            raise value
        return {"status": "complete", "holdout_logloss": float(value), "qwk_argmax": 0.5, "wall_s": 1.0,
                "fit_s_per_bag": 0.5, "predict_s": 0.5, "rows_per_s": 500.0, "peak_vram_mb": None,
                "n_ctx_used": int(params["n_ctx"]), "n_bags_used": 1, "runtime": {}, "events": []}

    return fake_evaluate


# ─────────────────────────────────────────
# Wynik i protokół
# ─────────────────────────────────────────
class TestSelection:
    def test_result_shape_and_cache_files(self, data, tmp_path):
        result = run_select(data, tmp_path, [{"n_ctx": 150, "n_estimators": 1}, {"n_ctx": 80, "n_estimators": 1}])
        assert REQUIRED_KEYS <= set(result)
        assert result["kind"] == "tfm_config_selection" and result["metric"] == "holdout_logloss"
        assert result["n_fold_train"] is None and "n_completed" not in result
        assert result["n_complete"] == result["n_trials"] == 2 and result["n_fail"] == result["n_pruned"] == 0
        assert result["study_name"].startswith("tfm_tabpfn__triage__smoke__data-fp123__")
        assert [c["config"]["n_ctx"] for c in result["candidates"]] == [80, 150]       # najtańszy pierwszy
        assert result["best_params"]["n_bags"] == 1 and result["best_trial"] in (0, 1)
        chosen = result["candidates"][result["best_trial"]]
        assert result["best_value"] == chosen["holdout_logloss"]
        assert result["throughput"]["rows_per_s"] > 0 and result["throughput"]["n_ctx_used"] == chosen["n_ctx_used"]
        assert result["holdout"]["rows"] == 250 and result["holdout"]["stratified_by"] == "esi_x_site"
        assert result["weights_override"] == "SOR_AI_TFM_FAKE=1" and result["library_version"] == "fake"
        files = sorted((tmp_path / "optuna").glob("tfm_tabpfn_*.json"))
        kinds = [json.loads(p.read_text(encoding="utf-8"))["kind"] for p in files]
        assert sorted(kinds) == ["tfm_candidate", "tfm_candidate", "tfm_selection"]
        assert not list((tmp_path / "optuna").glob("*.lock"))                     # blokada zwolniona
        assert json.loads(open(result["best_json"], encoding="utf-8").read())["best_value"] == result["best_value"]

    def test_holdout_never_in_context(self, data, tmp_path, monkeypatch):
        calls: list = []
        monkeypatch.setattr(tfm_tuning, "_evaluate", scripted({80: 0.8}, calls))
        run_select(data, tmp_path, [{"n_ctx": 80, "n_estimators": 1}])
        pool, hold = calls[0]["pool"], calls[0]["holdout"]
        assert len(hold) == 250 and len(pool) == 650 and not set(pool) & set(hold)

    def test_holdout_split_stratified(self, data):
        _, y, strata = data
        pool, hold, how = holdout_split(y, strata, 300)
        assert how == "esi_x_site" and len(hold) == 300 and len(np.intersect1d(pool, hold)) == 0
        np.testing.assert_allclose(np.bincount(y[hold], minlength=5) / 300, np.bincount(y, minlength=5) / len(y),
                                   atol=0.02)
        assert len(holdout_split(y, None, 10_000)[1]) == len(y) // 2               # najwyżej połowa
        assert holdout_split(y, None, 50)[2] == "esi"

    def test_lowest_loss_and_tolerance_prefers_cheaper(self, data, tmp_path, monkeypatch):
        monkeypatch.setattr(tfm_tuning, "_evaluate", scripted({80: 0.800, 150: 0.7990, 300: 0.70}))
        result = run_select(data, tmp_path / "a", [{"n_ctx": 80, "n_estimators": 1},
                                                   {"n_ctx": 150, "n_estimators": 1}])
        assert result["best_params"]["n_ctx"] == 80                 # 0,7990 w tolerancji 0,002 → tańszy
        assert result["best_overall_value"] == pytest.approx(0.799) and result["best_value"] == pytest.approx(0.8)
        result = run_select(data, tmp_path / "b", [{"n_ctx": 80, "n_estimators": 1},
                                                   {"n_ctx": 300, "n_estimators": 1}])
        assert result["best_params"]["n_ctx"] == 300

    def test_n_trials_limits_to_cheapest(self, data, tmp_path, monkeypatch):
        calls: list = []
        monkeypatch.setattr(tfm_tuning, "_evaluate", scripted({80: 0.8, 150: 0.7, 300: 0.6}, calls))
        result = run_select(data, tmp_path, [{"n_ctx": 300, "n_estimators": 1}, {"n_ctx": 80, "n_estimators": 1},
                                             {"n_ctx": 150, "n_estimators": 1}], n_trials=2)
        assert [c["params"]["n_ctx"] for c in calls] == [80, 150]
        assert result["n_trials_target"] == 2 and result["n_candidates_profile"] == 3

    def test_bags_stage(self, data, tmp_path, monkeypatch):
        monkeypatch.setattr(tfm_tuning, "_evaluate", scripted({80: 0.80, -1: 0.75}))
        result = run_select(data, tmp_path / "a", [{"n_ctx": 80, "n_estimators": 1}], bags_stage=True, max_bags=3)
        assert result["best_params"] == {"n_ctx": 80, "n_estimators": 1, "n_bags": "auto", "max_bags": 3,
                                         "bag_mode": "partition"}
        assert result["stage2"]["n_bags_auto"] == 3 and result["candidates"][1]["stage"] == 2
        monkeypatch.setattr(tfm_tuning, "_evaluate", scripted({80: 0.80, -1: 0.799}))
        result = run_select(data, tmp_path / "b", [{"n_ctx": 80, "n_estimators": 1}], bags_stage=True, max_bags=3)
        assert result["best_params"]["n_bags"] == 1                  # poprawa w tolerancji → jeden worek

    def test_bags_stage_with_real_fake_estimator(self, data, tmp_path):
        result = run_select(data, tmp_path, [{"n_ctx": 200, "n_estimators": 1}], bags_stage=True, max_bags=2)
        stage2 = result["candidates"][1]
        assert stage2["status"] == "complete" and stage2["n_bags_used"] == 2

    def test_failures_and_oom_recorded(self, data, tmp_path, monkeypatch):
        losses = {80: ValueError("zepsute dane"), 150: tfm_fake.FakeOutOfMemoryError("CUDA out of memory"),
                  300: 0.7}
        monkeypatch.setattr(tfm_tuning, "_evaluate", scripted(losses))
        with capture_logs("src.models.tfm_tuning") as records:
            result = run_select(data, tmp_path, [{"n_ctx": n, "n_estimators": 1} for n in (80, 150, 300)])
        statuses = [c["status"] for c in result["candidates"]]
        assert statuses == ["failed", "oom", "complete"] and result["n_fail"] == 2 and result["n_oom"] == 1
        assert result["best_params"]["n_ctx"] == 300
        assert any("FAILED" in r.getMessage() for r in records) and any("OOM" in r.getMessage() for r in records)
        monkeypatch.setattr(tfm_tuning, "_evaluate", scripted({80: ValueError("x")}))
        with pytest.raises(RuntimeError, match="żaden kandydat"):
            run_select(data, tmp_path / "all_failed", [{"n_ctx": 80, "n_estimators": 1}])

    def test_empty_candidates_raise(self, data, tmp_path):
        with pytest.raises(ValueError, match="candidates"):
            run_select(data, tmp_path, [])


# ─────────────────────────────────────────
# Limit czasu, wznawianie, klucz, blokada
# ─────────────────────────────────────────
class TestResume:
    def test_timeout_skips_remaining(self, data, tmp_path, monkeypatch):
        calls: list = []
        monkeypatch.setattr(tfm_tuning, "_evaluate", scripted({80: 0.8, 150: 0.7, 300: 0.6}, calls))
        cands = [{"n_ctx": n, "n_estimators": 1} for n in (80, 150, 300)]
        with capture_logs("src.models.tfm_tuning") as records:
            result = run_select(data, tmp_path, cands, timeout_s=1e-9)
        assert [c["status"] for c in result["candidates"]] == ["complete", "skipped", "skipped"]
        assert result["n_pruned"] == 2 and result["stopped_reason"] == "timeout" and result["budget_exhausted"]
        assert any("pominięci" in r.getMessage() and "n_ctx=300" in r.getMessage() for r in records)
        assert len(list((tmp_path / "optuna").glob("tfm_tabpfn_*.json"))) == 2   # 1 kandydat + podsumowanie
        # Ponowne uruchomienie z większym limitem: gotowy kandydat z pamięci, reszta liczona
        calls.clear()
        result = run_select(data, tmp_path, cands, timeout_s=3600)
        assert [c["params"]["n_ctx"] for c in calls] == [150, 300]
        assert result["n_cached"] == 1 and result["best_params"]["n_ctx"] == 300

    def test_resume_reuses_complete_and_oom_retries_failed(self, data, tmp_path, monkeypatch):
        calls: list = []
        losses = {80: 0.8, 150: tfm_fake.FakeOutOfMemoryError("out of memory"), 300: ValueError("chwilowy błąd")}
        monkeypatch.setattr(tfm_tuning, "_evaluate", scripted(losses, calls))
        cands = [{"n_ctx": n, "n_estimators": 1} for n in (80, 150, 300)]
        first = run_select(data, tmp_path, cands)
        assert len(calls) == 3
        calls.clear()
        second = run_select(data, tmp_path, cands)
        assert [c["params"]["n_ctx"] for c in calls] == [300]              # tylko "failed" od nowa
        assert [c["cached"] for c in second["candidates"]] == [True, True, False]
        assert second["study_name"] == first["study_name"] and second["n_cached"] == 2

    def test_cache_key_includes_library_version_and_weights(self, data, tmp_path, monkeypatch):
        calls: list = []
        monkeypatch.setattr(tfm_tuning, "_evaluate", scripted({80: 0.8}, calls))
        cands = [{"n_ctx": 80, "n_estimators": 1}]
        run_select(data, tmp_path, cands)
        run_select(data, tmp_path, cands)
        assert len(calls) == 1
        monkeypatch.setattr(tfm_tuning, "_weights_key",
                            lambda model, params: ({"file": "inne", "size": 1, "sha256": "abc"}, "9.2.0", None))
        run_select(data, tmp_path, cands)
        assert len(calls) == 2                                              # inne wagi → nowy klucz

    def test_lock_refuses_second_process(self, data, tmp_path, monkeypatch):
        monkeypatch.setattr(tfm_tuning, "_evaluate", scripted({80: 0.8}))
        cands = [{"n_ctx": 80, "n_estimators": 1}]
        name = run_select(data, tmp_path, cands)["study_name"]
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            lock = tmp_path / "optuna" / f"{name}.lock"
            lock.write_text(json.dumps({"pid": sleeper.pid, "host": socket.gethostname(), "proc_start": None,
                                        "boot_time": None, "study": name}), encoding="utf-8")
            with pytest.raises(RuntimeError, match="trwa w innym procesie"):
                run_select(data, tmp_path, cands)
        finally:
            sleeper.kill()
            sleeper.wait()

    def test_selection_progress(self, data, tmp_path, monkeypatch):
        monkeypatch.setattr(tfm_tuning, "_evaluate", scripted({80: 0.8, 150: ValueError("x")}))
        run_select(data, tmp_path, [{"n_ctx": 80, "n_estimators": 1}, {"n_ctx": 150, "n_estimators": 1}])
        storage = tmp_path / "optuna"
        assert selection_progress(storage, "tabpfn", profile="smoke", data_fp="fp123") == \
            {"complete": 1, "oom": 0, "failed": 1}
        assert selection_progress(storage, "tabpfn", profile="quick") is None
        assert selection_progress(storage, "tabicl") is None


# ─────────────────────────────────────────
# Koszt kandydata, budżet finalnej predykcji
# ─────────────────────────────────────────
def test_candidate_cost():
    assert candidate_cost({"n_ctx": 1000, "n_estimators": 4}, 271) == 4000
    assert candidate_cost({"n_ctx": 1000, "n_estimators": "auto"}, 271) == 8000
    assert candidate_cost({"n_ctx": 1000, "n_estimators": 2, "max_features": 100}, 200) == 1000
    assert candidate_cost({"n_ctx": 1000, "n_estimators": 1, "n_bags": "auto", "max_bags": 4}, 10, 2500) == 3000
    assert candidate_cost({"n_ctx": 50_000, "n_estimators": 1}, 10, n_rows_pool=800) == 800


def test_final_bag_budget():
    thr = {"rows_per_s": 100.0, "fit_s_per_bag": 10.0}
    ok = final_bag_budget(thr, 4, 1_000, 200)               # 4 × (10 + 10) = 80 s ≤ 200 s
    assert ok == {"n_bags": 4, "reduce": False, "estimate_s": 80.0, "per_bag_s": 20.0, "final_budget_s": 200}
    cut = final_bag_budget(thr, 4, 1_000, 50)               # 80 s > 50 s → 2 worki
    assert cut["reduce"] and cut["n_bags"] == 2
    assert final_bag_budget(thr, 4, 10_000, 5)["n_bags"] == 1
    assert final_bag_budget({}, 4, 1_000, 50)["reduce"] is False
    assert final_bag_budget(thr, 4, 1_000, None)["reduce"] is False
    assert final_bag_budget(thr, 1, 10**6, 1)["reduce"] is False


def test_module_does_not_bind_storage_dir():
    assert not hasattr(tfm_tuning, "OPTUNA_STORAGE_DIR")
    assert "OPTUNA_STORAGE_DIR" not in open(tfm_tuning.__file__, encoding="utf-8").read().split('"""')[2]
    assert os.path.basename(tfm_tuning.__file__) == "tfm_tuning.py"
