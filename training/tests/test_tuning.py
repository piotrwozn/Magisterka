"""
Testy tuningu Optuną (`src.models.tuning`) na małych danych.

Pełny tuning trwa godziny — tu tylko: każdy model przechodzi przez fold CV,
a `tune_model` z 2 próbami × 2 foldy zwraca sensowny wynik i zapisuje historię.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import MODEL_REGISTRY
from src.models.base import compute_sample_weights
from src.models.tuning import (
    SUGGEST_FUNCTIONS,
    filter_features_by_importance_1fold,
    fit_predict_fold,
    tune_model,
)
from src.utils.config import N_CLASSES

# Parametry przyspieszające pojedynczy fold (reszta = domyślne z config.py)
FOLD_PARAMS: dict[str, dict] = {
    "logreg": {},
    "lightgbm": {"n_estimators": 40, "num_leaves": 7, "min_child_samples": 5},
    "xgboost": {"n_estimators": 40, "max_depth": 3, "early_stopping_rounds": 10},
    "catboost": {"iterations": 40, "depth": 4, "od_wait": 10},
    "random_forest": {"n_estimators": 30},
    "ebm": {"max_rounds": 200, "outer_bags": 2, "early_stopping_rounds": 20},
}


@pytest.fixture(scope="module")
def folds(make_xy):
    X, y = make_xy(n=300, seed=3)
    return X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:]


def test_search_space_for_every_model():
    assert set(SUGGEST_FUNCTIONS) == set(MODEL_REGISTRY)


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_fit_predict_fold(folds, name):
    X_tr, y_tr, X_va, y_va = folds
    weights = compute_sample_weights(y_tr, strategy="none")
    proba = fit_predict_fold(name, FOLD_PARAMS[name], X_tr, y_tr, X_va, y_va, weights)
    assert proba.shape == (len(X_va), N_CLASSES)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_fit_predict_fold_with_class_weights(folds):
    X_tr, y_tr, X_va, y_va = folds
    weights = compute_sample_weights(y_tr, class_weights={0: 10.0, 1: 2.0, 2: 1.0, 3: 1.0, 4: 2.0})
    proba = fit_predict_fold("logreg", {}, X_tr, y_tr, X_va, y_va, weights)
    assert proba.shape == (len(X_va), N_CLASSES)


@pytest.mark.parametrize("name, metric", [
    ("logreg", "logloss"),
    ("logreg", "qwk"),
    ("lightgbm", "logloss"),
    ("lightgbm", "qwk"),
])
def test_tune_model(isolated_paths, make_xy, name, metric):
    X, y = make_xy(n=300, seed=4)
    storage = isolated_paths.OPTUNA_STORAGE_DIR / f"optuna_{name}.db"
    result = tune_model(
        name, X, y, n_trials=2, metric=metric, n_folds=2,
        storage_path=storage, study_name=f"test_{name}",
    )
    params = result["best_params"]
    assert result["n_trials"] == 2
    assert result["n_folds"] == 2
    assert result["metric"] == metric
    assert np.isfinite(result["best_value"])
    if name == "logreg":
        assert set(params) >= {"C"}
    else:
        assert {"learning_rate", "num_leaves"} <= set(params)
    if metric == "qwk":
        assert {"cw_esi1", "cw_esi2", "cw_esi4", "cw_esi5"} <= set(params)
        assert -1.0 <= result["best_value"] <= 1.0
    else:
        assert not [k for k in params if k.startswith("cw_")]
        assert result["best_value"] > 0
    assert storage.exists()
    assert str(isolated_paths.TRAINING_LOGS_DIR) in result["history_csv"]
    history = pd.read_csv(result["history_csv"])
    assert len(history) == 2


def test_tune_model_resumes_study(isolated_paths, make_xy):
    """Ta sama baza SQLite + nazwa badania → próby się sumują (tuning można wznowić)."""
    X, y = make_xy(n=200, seed=5)
    storage = isolated_paths.OPTUNA_STORAGE_DIR / "resume.db"
    kwargs = dict(n_trials=1, n_folds=2, storage_path=storage, study_name="resume")
    tune_model("logreg", X, y, **kwargs)
    second = tune_model("logreg", X, y, **kwargs)
    assert second["n_trials"] == 2


def test_tune_model_max_rows(isolated_paths, make_xy):
    X, y = make_xy(n=400, seed=6)
    result = tune_model("logreg", X, y, n_trials=1, n_folds=2, max_rows=150)
    assert result["n_rows"] == 150


def test_tune_model_unknown_metric(isolated_paths, make_xy):
    X, y = make_xy(n=100, seed=7)
    with pytest.raises(ValueError, match="Nieznana metryka"):
        tune_model("logreg", X, y, n_trials=1, metric="accuracy")


def test_filter_features_by_importance(make_xy, monkeypatch):
    import lightgbm as lgb

    # Filtr ma na sztywno n_jobs=-1; przy obciążonym CPU OpenMP spowalnia go ~100× → 1 wątek
    original = lgb.LGBMClassifier
    monkeypatch.setattr(lgb, "LGBMClassifier", lambda **kw: original(**{**kw, "n_jobs": 1}))

    X, y = make_xy(n=400, seed=8)
    X = X.assign(constant=0.0)
    kept = filter_features_by_importance_1fold(X, y, list(X.columns))
    assert "constant" not in kept
    assert "triage_vital_hr" in kept
    assert set(kept) <= set(X.columns)
