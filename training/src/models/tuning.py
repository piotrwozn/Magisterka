"""
Tuning hiperparametrów Optuną (k-fold CV na części treningowej).

Zasady:
    - Estymatory budowane przez te same klasy modeli co finalny trening
      (te same parametry domyślne, ten sam wybór CPU/GPU) — dostrojone
      parametry zawsze pasują do `run_training`.
    - Domyślny cel: wieloklasowy LOG LOSS (minimalizacja). Daje najlepsze
      prawdopodobieństwa, a te są podstawą obu reguł decyzyjnych i ręcznego
      łączenia modeli. Alternatywa: --tune-metric qwk (wtedy Optuna dobiera
      też wagi klas, bo QWK liczony na argmax premiuje rzadkie klasy).
    - Stałe foldy (seed) — uczciwe porównanie triali; MedianPruner przerywa
      wyraźnie słabe próby po pierwszych foldach.
    - Historia w SQLite (logs/optuna/) — tuning można przerwać i wznowić.

Zbiory `val` i `test` nie są używane — tuning widzi wyłącznie TRAIN.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import cohen_kappa_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.models import get_model, resolve_model_name
from src.models.base import compute_sample_weights
from src.utils.config import (
    CLASS_NAMES,
    CV_N_SPLITS,
    N_CLASSES,
    OPTUNA_N_TRIALS,
    OPTUNA_STORAGE_DIR,
    OPTUNA_TIMEOUT_SECONDS,
    RANDOM_SEED,
    TRAINING_LOGS_DIR,
)
from src.utils.logger import get_logger, get_training_logger

log = get_logger(__name__)

TUNE_METRICS = ("logloss", "qwk")


# ─────────────────────────────────────────
# Przestrzenie hiperparametrów
# ─────────────────────────────────────────
def _suggest_xgboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
    }


def _suggest_lightgbm(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 511, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 500, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 1.0, log=True),
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
    }


def _suggest_catboost(trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bernoulli", "Bayesian"]),
    }
    if params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    else:
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 5.0)
    return params


def _suggest_random_forest(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.1, 0.2, 0.3]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, log=True),
        "max_samples": trial.suggest_float("max_samples", 0.2, 0.8),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }


def _suggest_ebm(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_bins": trial.suggest_categorical("max_bins", [128, 256, 512]),
        "interactions": trial.suggest_int("interactions", 0, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 100, log=True),
        "max_leaves": trial.suggest_int("max_leaves", 2, 4),
        "smoothing_rounds": trial.suggest_int("smoothing_rounds", 0, 500, step=50),
    }


def _suggest_logreg(trial: optuna.Trial) -> dict[str, Any]:
    return {"C": trial.suggest_float("C", 1e-4, 10.0, log=True)}


SUGGEST_FUNCTIONS: dict[str, callable] = {
    "xgboost": _suggest_xgboost,
    "lightgbm": _suggest_lightgbm,
    "catboost": _suggest_catboost,
    "random_forest": _suggest_random_forest,
    "ebm": _suggest_ebm,
    "logreg": _suggest_logreg,
}


def _suggest_class_weights(trial: optuna.Trial) -> dict[int, float]:
    """Wagi klas jako hiperparametry (tylko przy celu 'qwk')."""
    return {
        0: trial.suggest_float("cw_esi1", 1.0, 30.0, log=True),
        1: trial.suggest_float("cw_esi2", 0.5, 5.0, log=True),
        2: 1.0,  # ESI 3 — klasa referencyjna
        3: trial.suggest_float("cw_esi4", 0.5, 5.0, log=True),
        4: trial.suggest_float("cw_esi5", 0.5, 10.0, log=True),
    }


# ─────────────────────────────────────────
# Trening jednego foldu
# ─────────────────────────────────────────
def fit_predict_fold(
    model_name: str,
    params: dict[str, Any],
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    sample_weight: np.ndarray,
) -> np.ndarray:
    """
    Trenuje surowy estymator (z early stopping na foldzie walidacyjnym, jeśli model
    go wspiera) i zwraca prawdopodobieństwa dla X_va, shape (n, 5).
    """
    wrapper = get_model(model_name, params=params)

    if model_name == "xgboost":
        est = wrapper._build_model(with_eval_set=True)
        est.fit(X_tr, y_tr, sample_weight=sample_weight, eval_set=[(X_va, y_va)], verbose=False)
    elif model_name == "lightgbm":
        import lightgbm as lgb

        from src.utils.config import LGBM_EARLY_STOPPING_ROUNDS

        est = wrapper._build_model()
        est.fit(
            X_tr, y_tr, sample_weight=sample_weight,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )
    elif model_name == "catboost":
        from catboost import CatBoostClassifier

        from src.models.catboost_model import _normalize_catboost_params

        cb_params = _normalize_catboost_params({**wrapper.params, "verbose": 0})
        est = CatBoostClassifier(**cb_params)
        est.fit(X_tr, y_tr, sample_weight=sample_weight, eval_set=(X_va, y_va), use_best_model=True)
    elif model_name == "logreg":
        est = wrapper._build_model()
        est.fit(X_tr, y_tr, clf__sample_weight=sample_weight)
    else:  # random_forest, ebm
        est = wrapper._build_model()
        est.fit(X_tr, y_tr, sample_weight=sample_weight)

    proba = np.asarray(est.predict_proba(X_va), dtype=float)
    if proba.shape[1] != N_CLASSES:
        raise RuntimeError(f"Model zwrócił {proba.shape[1]} kolumn prawdopodobieństw (oczekiwano {N_CLASSES})")
    return proba


def _score(metric: str, y_true: pd.Series, proba: np.ndarray) -> float:
    if metric == "logloss":
        return float(log_loss(y_true, proba, labels=list(range(N_CLASSES))))
    return float(cohen_kappa_score(y_true, proba.argmax(axis=1), weights="quadratic"))


# ─────────────────────────────────────────
# Szybki filtr cech
# ─────────────────────────────────────────
def filter_features_by_importance_1fold(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    model_name: str | None = None,
    min_importance: float = 0.0,
) -> list[str]:
    """
    Odrzuca cechy, których LightGBM (na połowie TRAIN) nie użył ani razu.

    LightGBM służy jako szybki, uniwersalny filtr niezależnie od modelu docelowego.
    """
    import lightgbm as lgb

    X_half, _, y_half, _ = train_test_split(
        X[feature_names], y, train_size=0.5, random_state=RANDOM_SEED, stratify=y
    )
    proxy = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=500, learning_rate=0.1, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=RANDOM_SEED, verbosity=-1, n_jobs=-1,
    )
    proxy.fit(X_half, y_half)
    importances = proxy.booster_.feature_importance(importance_type="split")
    kept = [f for f, imp in zip(feature_names, importances) if imp > min_importance]
    dropped = [f for f in feature_names if f not in set(kept)]
    log.info(f"Filtr cech (LightGBM, połowa TRAIN): zostaje {len(kept)} / {len(feature_names)}")
    if dropped:
        log.info(f"Odrzucone ({len(dropped)}): {dropped[:30]}{' …' if len(dropped) > 30 else ''}")
    return kept


# ─────────────────────────────────────────
# Główna funkcja tuningu
# ─────────────────────────────────────────
def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    n_trials: int = OPTUNA_N_TRIALS,
    metric: str = "logloss",
    n_folds: int = CV_N_SPLITS,
    max_rows: int | None = None,
    sample_weight_strategy: str = "none",
    timeout: int | None = OPTUNA_TIMEOUT_SECONDS,
    storage_path: Path | str | None = None,
    study_name: str | None = None,
) -> dict[str, Any]:
    """
    Optuna: k-fold CV na TRAIN.

    Parameters
    ----------
    metric : 'logloss' (minimalizacja, domyślnie) | 'qwk' (maksymalizacja + wagi klas)
    max_rows : int, optional
        Stratyfikowana próbka TRAIN do tuningu (przyspiesza; finalny trening i tak
        odbywa się na całości).
    sample_weight_strategy : str
        Wagi przy celu 'logloss' (przy 'qwk' wagi dobiera Optuna).
    """
    model_name = resolve_model_name(model_name)
    if metric not in TUNE_METRICS:
        raise ValueError(f"Nieznana metryka tuningu '{metric}'. Dostępne: {TUNE_METRICS}")

    X = X_train.reset_index(drop=True)
    y = pd.Series(np.asarray(y_train)).reset_index(drop=True)
    if max_rows is not None and max_rows < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, random_state=RANDOM_SEED, stratify=y)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tune_logger, log_path = get_training_logger(model_name=f"{model_name}_tuning", run_id=run_id)
    tune_logger.info(f"Optuna: {model_name} | cel={metric} | {n_folds}-fold CV | "
                     f"n={len(X):,} | trials={n_trials}")

    folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED).split(X, y))
    suggest = SUGGEST_FUNCTIONS[model_name]
    direction = "minimize" if metric == "logloss" else "maximize"

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        class_weights = _suggest_class_weights(trial) if metric == "qwk" else None
        scores: list[float] = []
        for i, (tr_idx, va_idx) in enumerate(folds):
            y_tr = y.iloc[tr_idx]
            weights = compute_sample_weights(y_tr, strategy=sample_weight_strategy, class_weights=class_weights)
            proba = fit_predict_fold(
                model_name, params, X.iloc[tr_idx], y_tr, X.iloc[va_idx], y.iloc[va_idx], weights,
            )
            scores.append(_score(metric, y.iloc[va_idx], proba))
            trial.report(float(np.mean(scores)), step=i)
            if trial.should_prune():
                tune_logger.info(f"Trial {trial.number:>4}: przerwany po foldzie {i + 1} "
                                 f"({metric}={np.mean(scores):.5f})")
                raise optuna.TrialPruned()
        value = float(np.mean(scores))
        tune_logger.info(f"Trial {trial.number:>4}: {metric}={value:.5f} (±{np.std(scores):.5f}) {params}")
        return value

    storage_url = None
    if storage_path is not None:
        Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{Path(storage_path).resolve().as_posix()}"
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED, multivariate=True, group=True,
                                           n_startup_trials=min(20, max(5, n_trials // 5))),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        storage=storage_url,
        study_name=(study_name or f"{model_name}_{run_id}") + f"_{metric}",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)

    tune_logger.info(f"Najlepszy {metric}: {study.best_value:.5f}")
    tune_logger.info(f"Najlepsze parametry: {study.best_params}")

    history_csv = TRAINING_LOGS_DIR / f"{model_name}_tuning" / f"{run_id}_trials.csv"
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(history_csv, index=False)

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "metric": metric,
        "n_trials": len(study.trials),
        "n_rows": len(X),
        "n_folds": n_folds,
        "study_name": study.study_name,
        "log_path": str(log_path),
        "history_csv": str(history_csv),
        "class_weight_keys": [f"cw_{c.lower()}" for c in CLASS_NAMES],
    }


__all__ = [
    "OPTUNA_STORAGE_DIR",
    "SUGGEST_FUNCTIONS",
    "filter_features_by_importance_1fold",
    "fit_predict_fold",
    "tune_model",
]
