"""
Hyperparameter tuning z Optuną.

Optuna używa TPE (Tree-structured Parzen Estimator) — bayesowskie wyszukiwanie
hiperparametrów, znacznie efektywniejsze niż grid/random search.

Optymalizujemy QUADRATIC WEIGHTED KAPPA (QWK) — najlepsza metryka dla
ordinalnej klasyfikacji triażu (penalizuje większe błędy ordinalne mocniej).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.models.base import compute_sample_weights
from src.utils.config import (
    EXPERIMENTS_DIR,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT_SECONDS,
    RANDOM_SEED,
    TRAINING_LOGS_DIR,
)
from src.utils.logger import get_logger, get_training_logger

log = get_logger(__name__)


# ─────────────────────────────────────────
# Search spaces per model
# ─────────────────────────────────────────
def _suggest_xgboost_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
    }


def _suggest_lightgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
    }


def _suggest_rf_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


SUGGEST_FUNCTIONS: dict[str, callable] = {
    "xgboost": _suggest_xgboost_params,
    "lightgbm": _suggest_lightgbm_params,
    "random_forest": _suggest_rf_params,
}


# ─────────────────────────────────────────
# Główna funkcja tuningu
# ─────────────────────────────────────────
def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_val: pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout: int | None = OPTUNA_TIMEOUT_SECONDS,
    sample_weight_strategy: str = "custom",
    storage_path: Path | str | None = None,
    study_name: str | None = None,
) -> dict[str, Any]:
    """
    Optymalizuje hiperparametry modelu maksymalizując QWK na zbiorze walidacyjnym.

    Parameters
    ----------
    model_name : str
        'xgboost' | 'lightgbm' | 'random_forest'
    X_train, y_train, X_val, y_val : pd.DataFrame, pd.Series
    n_trials : int
        Liczba prób Optuny.
    timeout : int, optional
        Limit czasu w sekundach.
    sample_weight_strategy : str
    storage_path : Path
        SQLite dla persistent study (do wznowienia po crashu).
    study_name : str, optional

    Returns
    -------
    dict z kluczami: best_params, best_value, n_trials, study_name, log_path
    """
    if model_name not in SUGGEST_FUNCTIONS:
        raise ValueError(
            f"Nieznany model: '{model_name}'. Dostępne: {sorted(SUGGEST_FUNCTIONS.keys())}"
        )

    # Logger dedykowany dla tuningu
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tune_logger, log_path = get_training_logger(
        model_name=f"{model_name}_tuning",
        run_id=run_id,
    )

    tune_logger.info(f"=== Optuna tuning: {model_name} ===")
    tune_logger.info(f"n_trials={n_trials}, timeout={timeout}, sample_weights='{sample_weight_strategy}'")

    sample_weights = compute_sample_weights(y_train, strategy=sample_weight_strategy)
    suggest_fn = SUGGEST_FUNCTIONS[model_name]

    # ─── Funkcja celu ───
    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial)
        params["random_state"] = RANDOM_SEED
        params["n_jobs"] = -1

        try:
            if model_name == "xgboost":
                import xgboost as xgb
                params.update({
                    "objective": "multi:softprob",
                    "num_class": 5,
                    "tree_method": "hist",
                    "verbosity": 0,
                })
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    sample_weight=sample_weights,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
            elif model_name == "lightgbm":
                import lightgbm as lgb
                params.update({
                    "objective": "multiclass",
                    "num_class": 5,
                    "verbosity": -1,
                })
                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_train, y_train,
                    sample_weight=sample_weights,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
                )
            elif model_name == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                params["class_weight"] = "balanced"
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                raise ValueError(f"Brak obsługi tuningu dla: {model_name}")

            y_pred = model.predict(X_val)
            qwk = cohen_kappa_score(y_val, y_pred, weights="quadratic")

            tune_logger.info(f"Trial {trial.number:>3}: QWK={qwk:.4f}, params={params}")
            return qwk

        except Exception as e:
            tune_logger.error(f"Trial {trial.number} błąd: {e}")
            return -1.0

    # ─── Konfiguracja Optuny ───
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    if storage_path is not None:
        storage_url = f"sqlite:///{Path(storage_path).resolve().as_posix()}"
    else:
        storage_url = None

    if study_name is None:
        study_name = f"{model_name}_{run_id}"

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        study_name=study_name,
        load_if_exists=True,
    )

    # ─── Optymalizacja ───
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
        gc_after_trial=True,
    )

    tune_logger.info("=== Tuning zakończony ===")
    tune_logger.info(f"Best QWK:    {study.best_value:.4f}")
    tune_logger.info(f"Best params: {study.best_params}")
    tune_logger.info(f"Trials:      {len(study.trials)}")

    # Zapisz historię tuningu jako CSV
    history_csv = TRAINING_LOGS_DIR / f"{model_name}_tuning" / f"{run_id}_trials.csv"
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(history_csv, index=False)
    tune_logger.info(f"Historia trials zapisana: {history_csv}")

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "study_name": study_name,
        "log_path": str(log_path),
        "history_csv": str(history_csv),
    }
