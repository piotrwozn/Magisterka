"""
Testy tuningu Optuną (`src.models.tuning`) na małych danych.

Pełny tuning trwa godziny — tu: przestrzenie, fold CV (early stopping na wycinku z części
treningowej, nigdy na ocenianym foldzie), klucz badania, wznawianie (tylko brakujące próby
i czas), odporność (błędne próby, osierocone próby po awarii, blokada), best.json, EBM równolegle.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.trial import TrialState

from src.models import MODEL_REGISTRY, tuning
from src.models.base import BaseTriageModel, compute_sample_weights
from src.models.tuning import (
    SUGGEST_FUNCTIONS,
    StudyLock,
    filter_features_by_importance_1fold,
    fit_predict_fold,
    n_startup_trials,
    tune_model,
)
from src.utils.config import (
    EARLY_STOPPING_FRACTION,
    N_CLASSES,
    SEARCH_SPACE_VERSION,
    TUNE_FIXED_PARAMS,
)

# Parametry przyspieszające pojedynczy fold (reszta = domyślne z config.py)
FOLD_PARAMS: dict[str, dict] = {
    "logreg": {},
    "lightgbm": {"n_estimators": 40, "num_leaves": 7, "min_child_samples": 5},
    "xgboost": {"n_estimators": 40, "max_depth": 3, "early_stopping_rounds": 10},
    "catboost": {"iterations": 40, "depth": 4, "od_wait": 10},
    "random_forest": {"n_estimators": 30},
    "ebm": {"max_rounds": 200, "outer_bags": 2, "early_stopping_rounds": 20},
}
# Szybkie próby w testach tune_model (stałe parametry każdej próby)
FAST_FIXED = {"lightgbm": {"n_estimators": 30}, "logreg": {"max_iter": 300}}


@pytest.fixture(scope="module")
def folds(make_xy):
    X, y = make_xy(n=300, seed=3)
    return X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:]


def _tune(paths, name="logreg", n=200, seed=5, **kwargs):
    """tune_model na małych danych z magazynem SQLite w katalogu testu."""
    from tests.conftest import make_numeric_xy

    X, y = make_numeric_xy(n=n, seed=seed)
    kwargs.setdefault("n_trials", 1)
    kwargs.setdefault("n_folds", 2)
    kwargs.setdefault("storage_path", paths.OPTUNA_STORAGE_DIR / f"optuna_{name}.db")
    kwargs.setdefault("fixed_params", FAST_FIXED.get(name))
    return tune_model(name, X, y, **kwargs)


def _load(result: dict) -> optuna.Study:
    return optuna.load_study(study_name=result["study_name"], storage=f"sqlite:///{result['storage']}")


# ─────────────────────────────────────────
# Przestrzenie
# ─────────────────────────────────────────
def test_search_space_for_every_model():
    assert set(SUGGEST_FUNCTIONS) == set(MODEL_REGISTRY)
    assert set(SEARCH_SPACE_VERSION) == set(MODEL_REGISTRY)


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_search_space_samples(name):
    study = optuna.create_study()
    params = SUGGEST_FUNCTIONS[name](study.ask(), "triage")
    assert params
    if name == "random_forest":
        assert "n_estimators" not in params            # stałe: 300 w CV, 500 w finale
        assert 2 <= params["min_samples_leaf"] <= 40
    if name == "ebm":
        assert "interactions" not in params            # stałe 0 (glassbox)
        assert params["max_bins"] in (256, 512, 1024)
    assert TUNE_FIXED_PARAMS["ebm"]["interactions"] == 0
    assert TUNE_FIXED_PARAMS["random_forest"]["n_estimators"] == 300


def test_random_forest_full_set_max_features():
    choices = set()
    for i in range(40):
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=i))
        choices.add(SUGGEST_FUNCTIONS["random_forest"](study.ask(), "full")["max_features"])
    assert choices <= {"sqrt", "log2", 0.05, 0.1}


def test_n_startup_trials_scale_with_dimensions():
    assert n_startup_trials("logreg") == 10
    assert n_startup_trials("xgboost") == 18
    assert n_startup_trials("lightgbm", "qwk") == 26


# ─────────────────────────────────────────
# Fold CV
# ─────────────────────────────────────────
@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_fit_predict_fold(folds, name):
    X_tr, y_tr, X_va, _ = folds
    proba, info = fit_predict_fold(name, FOLD_PARAMS[name], X_tr, y_tr, X_va)
    assert proba.shape == (len(X_va), N_CLASSES)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert info["actual_device"] == "cpu"
    uses_es = MODEL_REGISTRY[name].supports_early_stopping
    assert (info["n_es"] > 0) == uses_es
    assert info["n_train"] + info["n_es"] == len(X_tr)
    if name == "random_forest":
        assert info["nodes_per_tree"] > 1


def test_fit_predict_fold_with_class_weights(folds):
    X_tr, y_tr, X_va, _ = folds
    weights = compute_sample_weights(y_tr, class_weights={0: 10.0, 1: 2.0, 2: 1.0, 3: 1.0, 4: 2.0})
    proba, _ = fit_predict_fold("logreg", {}, X_tr, y_tr, X_va, weights)
    assert proba.shape == (len(X_va), N_CLASSES)


@pytest.mark.parametrize("name", ["lightgbm", "logreg"])
def test_early_stopping_slice_comes_from_fold_train_part(folds, monkeypatch, name):
    """Early stopping na wycinku z treningowej części foldu — nigdy na ocenianym foldzie."""
    X_tr, y_tr, X_va, _ = folds
    calls = []
    original = BaseTriageModel.fit_estimator

    def spy(self, X, y, X_es=None, y_es=None, sample_weight=None):
        calls.append((X, X_es, sample_weight))
        return original(self, X, y, X_es, y_es, sample_weight=sample_weight)

    monkeypatch.setattr(BaseTriageModel, "fit_estimator", spy)
    weights = np.arange(len(X_tr), dtype=float) + 1.0
    fit_predict_fold(name, FOLD_PARAMS[name], X_tr, y_tr, X_va, weights)

    (X_fit, X_es, w_fit), = calls
    if name == "logreg":
        assert X_es is None and len(X_fit) == len(X_tr)
        return
    fit_idx, es_idx, va_idx = set(X_fit.index), set(X_es.index), set(X_va.index)
    assert es_idx and es_idx <= set(X_tr.index)
    assert not es_idx & va_idx and not es_idx & fit_idx
    assert fit_idx | es_idx == set(X_tr.index)
    assert len(X_es) == pytest.approx(EARLY_STOPPING_FRACTION * len(X_tr), abs=2)
    # Wagi przycięte do części fit (te same wiersze)
    np.testing.assert_array_equal(w_fit, weights[X_tr.index.get_indexer(X_fit.index)])


def test_weighted_logloss_score():
    y = np.array([0, 1, 2, 3, 4, 2, 2, 2])
    proba = np.full((len(y), N_CLASSES), 0.1)
    proba[np.arange(len(y)), 2] = 0.6
    plain = tuning._score("logloss", y, proba)
    weighted = tuning._score("logloss", y, proba, compute_sample_weights(y, strategy="balanced"))
    assert weighted > plain          # rzadkie klasy (źle przewidziane) ważą więcej


# ─────────────────────────────────────────
# tune_model
# ─────────────────────────────────────────
@pytest.mark.parametrize("name, metric", [
    ("logreg", "logloss"),
    ("logreg", "qwk"),
    ("lightgbm", "logloss"),
])
def test_tune_model(isolated_paths, make_xy, name, metric):
    X, y = make_xy(n=300, seed=4)
    storage = isolated_paths.OPTUNA_STORAGE_DIR / f"optuna_{name}.db"
    result = tune_model(name, X, y, n_trials=2, metric=metric, n_folds=2, storage_path=storage,
                        fixed_params=FAST_FIXED[name], feature_set="triage", profile="smoke")
    params = result["best_params"]
    assert result["n_trials"] == 2 and result["n_complete"] + result["n_pruned"] == 2
    assert result["n_fail"] == 0
    assert result["n_folds"] == 2 and result["metric"] == metric
    assert np.isfinite(result["best_value"])
    assert result["search_space_version"] == SEARCH_SPACE_VERSION[name]
    assert result["config"]["profile"] == "smoke" and result["config"]["n_rows"] == 300
    assert name in result["study_name"] and "smoke" in result["study_name"]
    if name == "logreg":
        assert set(params) >= {"C"}
    else:
        assert {"learning_rate", "num_leaves"} <= set(params)
        assert result["best_iter"] and len(result["best_iter"]) == 2
    if metric == "qwk":
        assert {"cw_esi1", "cw_esi2", "cw_esi4", "cw_esi5"} <= set(params)
        assert -1.0 <= result["best_value"] <= 1.0
    else:
        assert not [k for k in params if k.startswith("cw_")]
        assert result["best_value"] > 0
    assert storage.exists()
    assert str(isolated_paths.TRAINING_LOGS_DIR) in result["history_csv"]
    assert len(pd.read_csv(result["history_csv"])) == 2

    # Atrybuty próby: wyniki foldów, metryki bezpieczeństwa OOF, urządzenie
    best = _load(result).best_trial
    assert len(best.user_attrs["fold_scores"]) == 2
    for key in ("qwk_argmax", "qwk_cost", "expected_cost_cost", "undertriage_cost", "cv_std"):
        assert key in best.user_attrs
    assert best.user_attrs["actual_device"] == "cpu"

    # best.json — atomowo, czytelny bez Optuny
    best_json = json.loads(Path(result["best_json"]).read_text(encoding="utf-8"))
    assert best_json["best_params"] == params
    assert best_json["study_name"] == result["study_name"]
    assert best_json["budget_exhausted"] is True
    assert not list(isolated_paths.OPTUNA_STORAGE_DIR.glob("*.tmp"))
    assert not list(isolated_paths.OPTUNA_STORAGE_DIR.glob("*.lock"))   # blokada zwolniona


def test_resume_runs_only_remaining_trials(isolated_paths):
    first = _tune(isolated_paths, n_trials=2)
    assert first["n_trials_session"] == 2
    second = _tune(isolated_paths, n_trials=3)
    assert second["study_name"] == first["study_name"]
    assert second["n_trials_session"] == 1 and second["n_trials"] == 3
    third = _tune(isolated_paths, n_trials=3)
    assert third["n_trials_session"] == 0 and third["stopped_reason"] == "n_trials"
    assert len(_load(third).trials) == 3


def test_resume_respects_cumulative_timeout(isolated_paths):
    first = _tune(isolated_paths, n_trials=1, timeout=600)
    study = _load(first)
    study.set_user_attr("elapsed_s", 600.0)          # budżet czasu zużyty w poprzednich sesjach
    second = _tune(isolated_paths, n_trials=5, timeout=600)
    assert second["n_trials_session"] == 0
    assert second["stopped_reason"] == "timeout" and second["budget_exhausted"] is True


def test_resume_reseeds_sampler(isolated_paths):
    """Wznowienie nie powtarza parametrów pierwszych (losowych) prób."""
    _tune(isolated_paths, n_trials=2)
    result = _tune(isolated_paths, n_trials=4)
    values = [t.params["C"] for t in _load(result).trials]
    assert len(set(values)) == 4


def test_different_config_creates_new_study(isolated_paths):
    a = _tune(isolated_paths, n_folds=2)
    b = _tune(isolated_paths, n_folds=3)
    c = _tune(isolated_paths, n_folds=2, sample_weight_strategy="balanced")
    d = _tune(isolated_paths, n_folds=2, profile="quick")
    assert len({a["study_name"], b["study_name"], c["study_name"], d["study_name"]}) == 4
    assert c["config"]["weights"] == "balanced"
    for r in (a, b, c, d):
        assert r["n_trials_session"] == 1                # żadne próby nie zostały wymieszane


def test_explicit_study_name_with_other_config_is_an_error(isolated_paths):
    _tune(isolated_paths, study_name="moje_badanie", n_folds=2)
    with pytest.raises(RuntimeError, match="inną konfigurację"):
        _tune(isolated_paths, study_name="moje_badanie", n_folds=3)


def test_failing_trial_does_not_kill_study(isolated_paths, monkeypatch):
    original = tuning.fit_predict_fold
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA error: symulowany błąd")
        return original(*args, **kwargs)

    monkeypatch.setattr(tuning, "fit_predict_fold", flaky)
    result = _tune(isolated_paths, n_trials=2)
    assert result["n_fail"] == 1 and result["n_complete"] == 2
    failed = [t for t in _load(result).trials if t.state == TrialState.FAIL]
    assert "symulowany" in failed[0].user_attrs["error"]


def test_consecutive_failures_stop_and_clear_error(isolated_paths, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("zawsze źle")

    monkeypatch.setattr(tuning, "fit_predict_fold", broken)
    with pytest.raises(RuntimeError, match="brak ukończonej próby.*zawsze źle"):
        _tune(isolated_paths, n_trials=20)
    study = optuna.load_study(
        study_name=optuna.study.get_all_study_names(
            f"sqlite:///{isolated_paths.OPTUNA_STORAGE_DIR / 'optuna_logreg.db'}")[0],
        storage=f"sqlite:///{isolated_paths.OPTUNA_STORAGE_DIR / 'optuna_logreg.db'}",
    )
    assert len(study.trials) == tuning.OPTUNA_MAX_CONSECUTIVE_FAILS
    assert not list(isolated_paths.OPTUNA_STORAGE_DIR.glob("*.lock"))


def test_orphaned_running_trial_is_failed_and_retried(isolated_paths):
    """Próba RUNNING po awarii procesu → FAIL + jedno ponowienie z tymi samymi parametrami."""
    first = _tune(isolated_paths, n_trials=1)
    study = _load(first)
    orphan = study.ask()                              # symulacja: proces padł w trakcie próby
    orphan.suggest_float("C", 1e-4, 10.0, log=True)
    result = _tune(isolated_paths, n_trials=2)
    trials = _load(result).trials
    assert trials[orphan.number].state == TrialState.FAIL
    retried = [t for t in trials if t.system_attrs.get("failed_trial") == orphan.number]
    assert len(retried) == 1 and retried[0].state == TrialState.COMPLETE
    assert retried[0].params == trials[orphan.number].params
    assert result["n_complete"] == 2 and result["n_trials_session"] == 1


def test_rf_size_guard_prunes(isolated_paths, monkeypatch):
    monkeypatch.setattr(tuning, "TUNE_RF_MAX_MODEL_BYTES", 1.0)
    with pytest.raises(RuntimeError, match="brak ukończonej próby"):
        _tune(isolated_paths, name="random_forest", n_trials=1,
              fixed_params={"n_estimators": 5, "n_jobs": 1})
    study = optuna.load_study(
        study_name=optuna.study.get_all_study_names(
            f"sqlite:///{isolated_paths.OPTUNA_STORAGE_DIR / 'optuna_random_forest.db'}")[0],
        storage=f"sqlite:///{isolated_paths.OPTUNA_STORAGE_DIR / 'optuna_random_forest.db'}",
    )
    (trial,) = study.trials
    assert trial.state == TrialState.PRUNED and trial.user_attrs["pruned_size"] > 1.0


def test_no_pruning_after_last_fold(isolated_paths, monkeypatch):
    """Pruner pytany tylko między foldami — w pełni oceniona próba zawsze zachowuje wynik."""
    asked = []

    def should_prune(self):
        asked.append(self.number)
        return False

    monkeypatch.setattr(optuna.trial.Trial, "should_prune", should_prune)
    result = _tune(isolated_paths, n_trials=1, n_folds=3)
    assert result["n_complete"] == 1
    assert len(asked) == 2                              # po foldach 1 i 2, nie po 3.


def test_pruning_between_folds(isolated_paths, monkeypatch):
    monkeypatch.setattr(optuna.trial.Trial, "should_prune", lambda self: self.number == 1)
    result = _tune(isolated_paths, n_trials=2, n_folds=2)
    trial = _load(result).trials[1]
    assert trial.state == TrialState.PRUNED and len(trial.user_attrs["fold_scores"]) == 1
    assert result["n_pruned"] == 1 and result["n_complete"] == 1


def test_strata_and_max_rows(isolated_paths, make_xy):
    X, y = make_xy(n=400, seed=6)
    strata = np.array([f"{t}_{s}" for t, s in zip(y, np.resize(["A", "B", "C"], len(y)), strict=True)])
    result = tune_model("logreg", X, y, n_trials=1, n_folds=2, max_rows=150, strata=strata)
    assert result["n_rows"] == 150
    assert result["config"]["stratified_by"] == "esi_site"
    assert result["best_json"] is None                # bez magazynu: bez best.json


def test_tune_model_unknown_metric(isolated_paths, make_xy):
    X, y = make_xy(n=100, seed=7)
    with pytest.raises(ValueError, match="Nieznana metryka"):
        tune_model("logreg", X, y, n_trials=1, metric="accuracy")


# ─────────────────────────────────────────
# EBM: foldy równolegle
# ─────────────────────────────────────────
def test_ebm_fold_workers():
    assert tuning._ebm_fold_workers({"n_jobs": 24}, 3) == (3, 8)
    assert tuning._ebm_fold_workers({"n_jobs": 1}, 3) == (1, 1)
    assert tuning._ebm_fold_workers({"n_jobs": 2}, 3) == (2, 1)


def test_ebm_parallel_folds(isolated_paths, monkeypatch):
    """Foldy EBM w procesach loky (2 foldy × 1 wątek), bez przerywania po foldzie."""
    original = tuning._run_folds_parallel
    calls = []

    def spy(model_name, params, jobs, n_workers):
        out = original(model_name, params, jobs, n_workers)
        calls.append((params["n_jobs"], n_workers, len(out)))
        return out

    monkeypatch.setattr(tuning, "_run_folds_parallel", spy)
    fixed = {"max_rounds": 60, "outer_bags": 2, "n_jobs": 2, "early_stopping_rounds": 10}
    result = _tune(isolated_paths, name="ebm", n=160, n_trials=1, fixed_params=fixed)
    assert calls == [(1, 2, 2)]
    trial = _load(result).trials[0]
    assert trial.state == TrialState.COMPLETE and len(trial.user_attrs["fold_scores"]) == 2
    assert "interactions" not in result["best_params"]


def test_ebm_parallel_failure_falls_back_to_sequential(isolated_paths, monkeypatch):
    def broken_pool(*args, **kwargs):
        raise OSError("symulowana awaria puli procesów")

    monkeypatch.setattr(tuning, "_run_folds_parallel", broken_pool)
    fixed = {"max_rounds": 60, "outer_bags": 2, "n_jobs": 2, "early_stopping_rounds": 10}
    result = _tune(isolated_paths, name="ebm", n=160, n_trials=1, fixed_params=fixed)
    assert result["n_complete"] == 1


def test_ebm_fold_worker_runs_bags_as_processes(monkeypatch):
    """
    W procesie roboczym loky joblib ustawia zagnieżdżony backend na wątki — worki EBM (pętla
    w Pythonie, GIL) byłyby wtedy 2× wolniejsze. Opakowanie foldu przywraca procesy loky.
    """
    from joblib import parallel_config
    from joblib.parallel import get_active_backend

    seen: list[str] = []

    def fake_fold(model_name, params, X_tr, y_tr, X_va, sample_weight):
        seen.append(type(get_active_backend()[0]).__name__)
        return np.zeros((1, N_CLASSES)), {}

    monkeypatch.setattr(tuning, "fit_predict_fold", fake_fold)
    with parallel_config(backend="threading"):          # jak BatchedCalls w procesie roboczym loky
        tuning._fit_predict_fold_worker("ebm", {}, None, None, None, None)
    assert seen == ["LokyBackend"]


# ─────────────────────────────────────────
# Blokada badania
# ─────────────────────────────────────────
def test_lock_held_by_live_process_blocks(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        path = tmp_path / "s.lock"
        path.write_text(json.dumps({"pid": proc.pid, "host": __import__("socket").gethostname()}),
                        encoding="utf-8")
        with pytest.raises(RuntimeError, match=f"PID {proc.pid}"):
            StudyLock(path, "s").acquire()
    finally:
        proc.kill()
        proc.wait()


def test_lock_from_before_reboot_or_reused_pid_is_stale(tmp_path, monkeypatch):
    """Żywy PID, ale blokada sprzed restartu albo PID należy do innego programu → przejęcie."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        host = __import__("socket").gethostname()
        path = tmp_path / "s.lock"
        boot_now = tuning._boot_time()
        if boot_now is not None:
            path.write_text(json.dumps({"pid": proc.pid, "host": host, "boot_time": boot_now - 86400}),
                            encoding="utf-8")
            lock = StudyLock(path, "s")
            lock.acquire()
            lock.release()

        path.write_text(json.dumps({"pid": proc.pid, "host": host}), encoding="utf-8")
        monkeypatch.setattr(tuning, "_process_name", lambda pid: "/usr/sbin/sshd")
        lock = StudyLock(path, "s")
        lock.acquire()
        lock.release()
    finally:
        proc.kill()
        proc.wait()


def test_stale_lock_is_taken_over(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()                                        # PID martwego procesu
    path = tmp_path / "s.lock"
    path.write_text(json.dumps({"pid": proc.pid, "host": __import__("socket").gethostname()}),
                    encoding="utf-8")
    lock = StudyLock(path, "s")
    lock.acquire()
    assert json.loads(path.read_text(encoding="utf-8"))["pid"] == os.getpid()
    lock.release()
    assert not path.exists()


def test_corrupted_lock_is_taken_over(tmp_path):
    path = tmp_path / "s.lock"
    path.write_text("", encoding="utf-8")              # awaria w trakcie zapisu blokady
    lock = StudyLock(path, "s")
    lock.acquire()
    lock.release()


# ─────────────────────────────────────────
# Filtr cech
# ─────────────────────────────────────────
def test_filter_features_by_importance(make_xy, monkeypatch):
    import lightgbm as lgb

    # Przy obciążonym CPU OpenMP potrafi spowolnić LightGBM ~100× → 1 wątek
    original = lgb.LGBMClassifier
    monkeypatch.setattr(lgb, "LGBMClassifier", lambda **kw: original(**{**kw, "n_jobs": 1}))

    X, y = make_xy(n=400, seed=8)
    X = X.assign(constant=0.0)
    kept = filter_features_by_importance_1fold(X, y, list(X.columns))
    assert "constant" not in kept
    assert "triage_vital_hr" in kept
    assert set(kept) <= set(X.columns)
