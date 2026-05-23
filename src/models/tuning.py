"""
Hyperparameter tuning z Optuną — maksymalna jakość (czas nieistotny).

Strategia maksymalnej jakości:
1. 5-fold StratifiedKFold CV (stabilna ocena generalizacji).
2. Stały seed dla CV — wszystkie triale na tych samych splitach (fair comparison).
3. Cost-sensitive wagi klas jako hiperparametry Optuny.
4. MultivariateTPESampler (modeluje korelacje między hiperparametrami).
5. NopPruner — żaden trial nie jest zabijany przedwcześnie.
6. n_startup_trials=50 — więcej losowej eksploracji przed bayesowskim ułożeniem TPE.
7. n_ei_candidates=48 — więcej kandydatów per iteracja TPE.
8. Storage SQLite domyślnie — wznowienie po crashu / wyłączeniu.
9. GPU dla XGBoost (CUDA), CPU dla reszty.
10. Rozszerzone przestrzenie hiperparametrów (DART, GOSS, log-scale regularyzacja).

Optymalizujemy QUADRATIC WEIGHTED KAPPA (QWK) — najlepsza metryka dla
ordinalnej klasyfikacji triażu (penalizuje większe błędy ordinalne mocniej).
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

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
# Konfiguracja CV (maksymalna jakość)
# ─────────────────────────────────────────
CV_N_SPLITS = 5  # 5-fold zamiast 3-fold dla stabilniejszej oceny


# ─────────────────────────────────────────
# Search spaces per model (rozszerzone)
# ─────────────────────────────────────────
def _suggest_xgboost_params(trial: optuna.Trial) -> dict[str, Any]:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 4000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 14),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-6, 50.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "max_bin": trial.suggest_int("max_bin", 128, 2048, step=64),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
    }

    return params


def _suggest_lightgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    params = {
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 500, 2500, step=100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 5.0, log=True),
        "max_bin": trial.suggest_int("max_bin", 128, 1024, step=64),
        "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.5, 1.0),
        "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 1.0),
    }

    if params["boosting_type"] == "dart":
        params["drop_rate"] = trial.suggest_float("drop_rate", 0.01, 0.4)
        params["max_drop"] = trial.suggest_int("max_drop", 5, 60)
        params["uniform_drop"] = trial.suggest_categorical("uniform_drop", [True, False])

    return params


def _suggest_rf_params(trial: optuna.Trial) -> dict[str, Any]:
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
        "max_depth": trial.suggest_int("max_depth", 8, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.2, 0.4, 0.6]),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 1e-8, 1e-3, log=True),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.005),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "bootstrap": bootstrap,
        "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-8, 1e-3, log=True),
        "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [None, 500, 1000, 5000, 10000]),
    }
    if bootstrap:
        params["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)
    return params


def _suggest_ebm_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.5, log=True),
        "max_bins": trial.suggest_int("max_bins", 128, 2048, step=64),
        "max_interaction_bins": trial.suggest_int("max_interaction_bins", 16, 128, step=8),
        "interactions": trial.suggest_int("interactions", 0, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100),
        "max_rounds": trial.suggest_int("max_rounds", 3000, 20000, step=500),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 50, 300, step=10),
        "outer_bags": trial.suggest_int("outer_bags", 1, 8),
        "inner_bags": trial.suggest_int("inner_bags", 0, 4),
        "validation_size": trial.suggest_float("validation_size", 0.1, 0.4),
        "smoothing_rounds": trial.suggest_int("smoothing_rounds", 10, 200, step=10),
        "greedy_ratio": trial.suggest_float("greedy_ratio", 1.0, 50.0, log=True),
    }


def _suggest_catboost_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    Pełna przestrzeń CatBoost — wszystkie wymiary regularyzacji i struktury drzew.

    Konwencja nazw CatBoost (NIE sklearn):
      - depth  (max_depth)        - l2_leaf_reg (reg_lambda)
      - rsm    (colsample_bylevel) - border_count (max_bin)
      - iterations (n_estimators)  - random_strength (Bayesian regul.)
    """
    bootstrap_type = trial.suggest_categorical(
        "bootstrap_type", ["Bernoulli", "MVS", "Bayesian"]
    )
    grow_policy = trial.suggest_categorical(
        "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
    )

    params: dict[str, Any] = {
        "iterations": trial.suggest_int("iterations", 500, 5000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 254, step=2),
        "rsm": trial.suggest_float("rsm", 0.4, 1.0),
        "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
        "leaf_estimation_method": trial.suggest_categorical(
            "leaf_estimation_method", ["Newton", "Gradient"]
        ),
        "bootstrap_type": bootstrap_type,
        "grow_policy": grow_policy,
    }

    # Conditional params — zależą od bootstrap_type
    if bootstrap_type in ("Bernoulli", "MVS"):
        params["subsample"] = trial.suggest_float("subsample", 0.4, 1.0)
    elif bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)

    # Conditional — grow_policy
    if grow_policy == "Lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 16, 256)
    if grow_policy in ("Depthwise", "Lossguide"):
        params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 100)

    return params


def _suggest_extra_trees_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    Pełna przestrzeń ExtraTrees — 14+ hiperparametrów.

    ExtraTrees różni się od RF:
      - Domyślnie bootstrap=False (cały dataset)
      - Splits są losowe, nie best-split — szybszy, większa wariancja
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 6, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5, 0.7]
        ),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 1e-8, 1e-3, log=True),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.001),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-8, 1e-3, log=True),
        "max_leaf_nodes": trial.suggest_categorical(
            "max_leaf_nodes", [None, 500, 1000, 2500, 5000, 10000]
        ),
        "bootstrap": False,
        "class_weight": None,
    }


def _suggest_hist_gbt_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    Pełna przestrzeń HistGradientBoosting — 14+ hiperparametrów.

    HistGBT używa binowania histogramowego (inny inductive bias niż
    XGB/LGBM/CatBoost). Kluczowe parametry:
      - max_depth / max_leaf_nodes: kontrola głębokości
      - l2_regularization: L2 na liściach
      - max_bins: jakość histogramów (im więcej tym lepiej, ale wolniej)
      - early_stopping: natywny, z własną walidacją
    """
    early_stopping = trial.suggest_categorical("early_stopping", [True, False])

    params: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 200, 2000, step=50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255),
        "max_depth": trial.suggest_categorical("max_depth", [None, 4, 6, 8, 10, 12, 15]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        "max_bins": trial.suggest_int("max_bins", 128, 255),
        "early_stopping": early_stopping,
        "tol": trial.suggest_float("tol", 1e-7, 1e-3, log=True),
        "scoring": trial.suggest_categorical("scoring", ["loss", "neg_log_loss"]),
    }

    if early_stopping:
        params["validation_fraction"] = trial.suggest_float("validation_fraction", 0.05, 0.4)
        params["n_iter_no_change"] = trial.suggest_int("n_iter_no_change", 5, 50)
    else:
        params["validation_fraction"] = None
        params["n_iter_no_change"] = None

    return params


SUGGEST_FUNCTIONS: dict[str, callable] = {
    "xgboost": _suggest_xgboost_params,
    "lightgbm": _suggest_lightgbm_params,
    "random_forest": _suggest_rf_params,
    "extra_trees": _suggest_extra_trees_params,
    "hist_gbt": _suggest_hist_gbt_params,
    "ebm": _suggest_ebm_params,
    "catboost": _suggest_catboost_params,
}


# ─────────────────────────────────────────
# Cost-sensitive wagi klas (hiperparametry)
# ─────────────────────────────────────────
def _suggest_class_weights(trial: optuna.Trial) -> dict[int, float]:
    """Wagi klas jako hiperparametry — Yellow ma wagę 1.0 (referencyjna)."""
    return {
        0: trial.suggest_float("cw_red", 2.0, 25.0),       # Red
        1: trial.suggest_float("cw_orange", 1.0, 15.0),    # Orange
        2: 1.0,                                              # Yellow
        3: trial.suggest_float("cw_green", 0.5, 5.0),      # Green
        4: trial.suggest_float("cw_blue", 0.5, 10.0),      # Blue
    }


# ─────────────────────────────────────────
# Trening + ewaluacja per model (single fold)
# ─────────────────────────────────────────
# Globalny singleton redirectora CatBoost — CatBoost pozwala tylko na
# jeden log_cout w całym procesie, więc foldy współdzielą ten sam obiekt.
_CATBOOST_LOG_REDIRECTOR: _CatBoostLogRedirector | None = None

class _CatBoostLogRedirector:
    """Redirectuje CatBoost verbose output przez Python logging (plik + konsola).

    CatBoost przyjmuje `log_cout=` jako file-like z write()/flush().
    Zamiast iść na stdout, przekazujemy przez tune_logger — trafia
    do pliku .log i na konsolę (RichHandler).

    Uwaga: CatBoost wspiera tylko JEDEN log_cout w całym procesie.
    Dlatego foldy współdzielą ten sam singleton _CATBOOST_LOG_REDIRECTOR.
    """
    def __init__(self, logger, fold_idx: int = 0):
        self.logger = logger
        self.fold_idx = fold_idx
        self._buf = ""

    def write(self, msg: str) -> None:
        self._buf += msg
        if "\n" in self._buf:
            for line in self._buf.split("\n"):
                stripped = line.strip()
                if stripped:
                    self.logger.info(f"[Fold {self.fold_idx}] {stripped}")
            self._buf = ""

    def flush(self) -> None:
        if self._buf.strip():
            self.logger.info(f"[Fold {self.fold_idx}] {self._buf.strip()}")
            self._buf = ""


def _get_catboost_log_redirector(logger, fold_idx: int = 0) -> _CatBoostLogRedirector:
    """Zwraca singleton redirectora dla CatBoost.

    CatBoost pozwala na ustawienie log_cout tylko raz globalnie.
    Wszystkie foldy współdzielą ten sam obiekt, ale z różnym fold_idx.
    """
    global _CATBOOST_LOG_REDIRECTOR
    if _CATBOOST_LOG_REDIRECTOR is None:
        _CATBOOST_LOG_REDIRECTOR = _CatBoostLogRedirector(logger, fold_idx)
    else:
        # Tylko zaktualizuj fold_idx dla kolejnych wywołań
        _CATBOOST_LOG_REDIRECTOR.fold_idx = fold_idx
    return _CATBOOST_LOG_REDIRECTOR


def _train_eval_fold(
    model_name: str,
    params: dict,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    sample_weights: np.ndarray,
    tune_logger: logging.Logger | None = None,
    fold_idx: int = 0,
) -> float:
    """Trenuje model na foldzie i zwraca QWK.

    tune_logger — opcjonalny (dla CatBoost: redirect per-iteracyjnych logów
    przez Python logging zamiast stdout)."""
    if model_name == "xgboost":
        import xgboost as xgb
        import time as _time

        # Build train params (NOT XGBClassifier — używamy xgb.train + QuantileDMatrix dla GPU speedup)
        train_params = dict(params)
        # n_estimators → num_boost_round (xgb.train style)
        n_boost_round = int(train_params.pop("n_estimators", 1000))
        early_stop = int(train_params.pop("early_stopping_rounds", 100))
        # random_state → seed
        if "random_state" in train_params and "seed" not in train_params:
            train_params["seed"] = train_params.pop("random_state")

        train_params.setdefault("objective", "multi:softprob")
        train_params.setdefault("num_class", 5)
        train_params.setdefault("tree_method", "hist")
        train_params.setdefault("eval_metric", "mlogloss")
        train_params.setdefault("verbosity", 1)
        if "device" not in train_params:
            train_params["device"] = "cuda"

        # QuantileDMatrix — optymalny format dla tree_method=hist + GPU (2-3× szybciej)
        # max_bin musi być spójny: dtrain, dval i Booster (XGBoost 3.x ref= nie propaguje max_bin)
        qm_max_bin = int(train_params.get("max_bin", 256))
        dtrain = xgb.QuantileDMatrix(X_tr, label=np.asarray(y_tr), weight=sample_weights, max_bin=qm_max_bin)
        dval = xgb.QuantileDMatrix(X_cv, label=np.asarray(y_cv), max_bin=qm_max_bin, ref=dtrain)

        print(f"[Fold {fold_idx}] XGBoost start: {n_boost_round} rounds, device={train_params.get('device')}", flush=True)
        t0 = _time.time()
        booster = xgb.train(
            train_params,
            dtrain,
            num_boost_round=n_boost_round,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stop,
            verbose_eval=50,
        )
        print(f"[Fold {fold_idx}] XGBoost done: {_time.time()-t0:.1f}s", flush=True)

        y_pred_proba = booster.predict(dval)
        y_pred = np.argmax(y_pred_proba, axis=1).ravel().astype(int)
        qwk = cohen_kappa_score(y_cv, y_pred, weights="quadratic")
        print(f"[Fold {fold_idx}] QWK={qwk:.4f}", flush=True)
        return qwk

    elif model_name == "lightgbm":
        import lightgbm as lgb

        if params.get("boosting_type") == "goss":
            params["boosting_type"] = "gbdt"
            params["data_sample_strategy"] = "goss"

        params.update({
            "objective": "multiclass",
            "num_class": 5,
            "metric": "multi_logloss",
            "verbosity": -1,
            "random_state": RANDOM_SEED,
        })
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            sample_weight=sample_weights,
            eval_set=[(X_cv, y_cv)],
            callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(10)],
        )

    elif model_name == "random_forest":
        import time as _time

        cuml_gpu_id = params.pop("__cuml_gpu_id", None)

        if cuml_gpu_id is not None:
            # cuML GPU path
            import cupy as _cp
            from cuml.ensemble import RandomForestClassifier as _CuRF

            # Map sklearn-style → cuML-style
            crit_map = {"gini": 0, "entropy": 1, "log_loss": 1}
            mf = params.get("max_features", "sqrt")
            if isinstance(mf, str):
                cuml_mf = mf
            else:
                cuml_mf = float(mf)

            cuml_params = {
                "n_estimators": int(params.get("n_estimators", 100)),
                "max_depth": int(params.get("max_depth", 16)),
                "min_samples_split": int(params.get("min_samples_split", 2)),
                "min_samples_leaf": int(params.get("min_samples_leaf", 1)),
                "max_features": cuml_mf,
                "bootstrap": bool(params.get("bootstrap", True)),
                "split_criterion": crit_map.get(params.get("criterion", "gini"), 0),
                "random_state": int(params.get("random_state", RANDOM_SEED)),
                "n_streams": 1,
                "n_bins": 128,
            }
            if cuml_params["bootstrap"] and "max_samples" in params:
                cuml_params["max_samples"] = float(params["max_samples"])

            # cuML nie wspiera sample_weight → weighted bootstrap
            if sample_weights is not None and not np.allclose(sample_weights, sample_weights[0]):
                probs = sample_weights / sample_weights.sum()
                rng = np.random.RandomState(RANDOM_SEED + fold_idx)
                idx = rng.choice(len(X_tr), size=len(X_tr), replace=True, p=probs)
                X_tr_eff = X_tr.iloc[idx].reset_index(drop=True)
                y_tr_eff = pd.Series(np.asarray(y_tr)[idx]).reset_index(drop=True)
            else:
                X_tr_eff = X_tr
                y_tr_eff = y_tr

            X_tr_np = X_tr_eff.values.astype("float32")
            X_cv_np = X_cv.values.astype("float32")
            y_tr_np = np.asarray(y_tr_eff).astype("int32")

            print(f"[Fold {fold_idx}] cuML RF GPU{cuml_gpu_id} start: {cuml_params['n_estimators']} trees", flush=True)
            t0 = _time.time()
            with _cp.cuda.Device(cuml_gpu_id):
                model = _CuRF(**cuml_params)
                model.fit(X_tr_np, y_tr_np)
                y_pred = np.asarray(model.predict(X_cv_np)).ravel().astype(int)
            print(f"[Fold {fold_idx}] cuML RF done: {_time.time()-t0:.1f}s", flush=True)

            qwk = cohen_kappa_score(y_cv, y_pred, weights="quadratic")
            print(f"[Fold {fold_idx}] QWK={qwk:.4f}", flush=True)
            return qwk

        # sklearn CPU path
        from sklearn.ensemble import RandomForestClassifier

        n_est = params.get("n_estimators", 500)
        n_j = params.get("n_jobs", 1)
        print(f"[Fold {fold_idx}] RF start: {n_est} trees, n_jobs={n_j}", flush=True)
        t0 = _time.time()
        model = RandomForestClassifier(class_weight=None, **params)
        model.fit(X_tr, y_tr, sample_weight=sample_weights)
        print(f"[Fold {fold_idx}] RF done: {_time.time()-t0:.1f}s", flush=True)

    elif model_name == "ebm":
        import time as _time
        from interpret.glassbox import ExplainableBoostingClassifier

        params["random_state"] = RANDOM_SEED
        print(f"[Fold {fold_idx}] EBM start", flush=True)
        t0 = _time.time()
        model = ExplainableBoostingClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=sample_weights)
        print(f"[Fold {fold_idx}] EBM done: {_time.time()-t0:.1f}s", flush=True)

    elif model_name == "extra_trees":
        import time as _time
        from sklearn.ensemble import ExtraTreesClassifier

        params.pop("class_weight", None)
        n_est = params.get("n_estimators", 500)
        n_j = params.get("n_jobs", 1)
        print(f"[Fold {fold_idx}] ET start: {n_est} trees, n_jobs={n_j}", flush=True)
        t0 = _time.time()
        model = ExtraTreesClassifier(class_weight=None, **params)
        model.fit(X_tr, y_tr, sample_weight=sample_weights)
        print(f"[Fold {fold_idx}] ET done: {_time.time()-t0:.1f}s", flush=True)

    elif model_name == "hist_gbt":
        import time as _time
        from sklearn.ensemble import HistGradientBoostingClassifier

        # Usuń puste parametry które HistGBT by odrzucił
        for key in ["validation_fraction", "n_iter_no_change"]:
            if params.get(key) is None:
                params.pop(key, None)
        params["verbose"] = 1   # per-iteration logi
        max_it = params.get("max_iter", 1000)
        print(f"[Fold {fold_idx}] HistGBT start: max_iter={max_it}", flush=True)
        t0 = _time.time()
        model = HistGradientBoostingClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=sample_weights)
        print(f"[Fold {fold_idx}] HistGBT done: {_time.time()-t0:.1f}s", flush=True)

    elif model_name == "catboost":
        from catboost import CatBoostClassifier, Pool

        from src.models.catboost_model import _normalize_catboost_params

        # _fold_idx to nasz prywatny znacznik — nie idzie do CatBoost
        fold_idx = params.pop("_fold_idx", 0)

        params = _normalize_catboost_params(params)
        # GPU restrictions: MVS + rsm nie wspierane dla multiclass na GPU
        if params.get("bootstrap_type") == "MVS":
            params["bootstrap_type"] = "Bernoulli"
        params.pop("rsm", None)
        params.update({
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "classes_count": 5,
            "task_type": "GPU",
            "allow_writing_files": False,
            "od_type": "Iter",
            "od_wait": 100,
            "boosting_type": "Plain",
            "verbose": 10,
            "gpu_ram_part": 0.85,
        })
        # devices może być ustawione per-fold (multi-GPU), nie nadpisuj
        params.setdefault("devices", "0")

        train_pool = Pool(data=X_tr, label=y_tr.values, weight=sample_weights)
        eval_pool = Pool(data=X_cv, label=y_cv.values)

        model = CatBoostClassifier(**params)
        fit_kwargs = {"eval_set": eval_pool, "use_best_model": True}
        # CatBoost nie wspiera ustawiania log_cout z wielu wątków —
        # verbose=10 idzie na stdout, nohup redirectuje do pliku.
        model.fit(train_pool, **fit_kwargs)

    else:
        raise ValueError(f"Brak obsługi tuningu dla: {model_name}")

    y_pred = np.asarray(model.predict(X_cv)).ravel()
    qwk = cohen_kappa_score(y_cv, y_pred, weights="quadratic")
    print(f"[Fold {fold_idx}] QWK={qwk:.4f}", flush=True)
    return qwk


# ─────────────────────────────────────────
# 1-fold feature importance filter (per-model)
# ─────────────────────────────────────────
def filter_features_by_importance_1fold(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    model_name: str = "lightgbm",
    min_importance: float = 0.0,
    n_estimators: int | None = None,
) -> list[str]:
    """Szybki 1-fold trening modelu do odrzucenia cech z zerową importance.

    Trenuje ten SAM model co target, na 1 foldzie.
    Dla modeli bez feature_importances_ (EBM) używa LightGBM jako proxy.
    """
    # Maksymalne n_estimators per model (szybki filter, nie full tuning)
    model_max_estimators = {
        "xgboost": 2000,
        "lightgbm": 2000,
        "catboost": 1000,
        "random_forest": 500,
        "extra_trees": 500,
        "hist_gbt": 1000,
    }
    alias_map = {"cb": "catboost", "rf": "random_forest", "et": "extra_trees", "hgbt": "hist_gbt"}
    model_key = alias_map.get(model_name, model_name)
    if n_estimators is None:
        n_estimators = model_max_estimators.get(model_key, 1000)
    log.info(f"n_estimators dla filtra: {n_estimators}")
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
    train_idx, _ = next(skf.split(X, y))

    X_fold = X.iloc[train_idx]
    y_fold = y.iloc[train_idx]

    if model_name == "xgboost":
        import xgboost as xgb
        log.info(f"=== 1-fold feature importance filter (XGBoost, {n_estimators} trees) ===")
        model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", device="cuda",
            random_state=RANDOM_SEED, verbosity=0,
        )
        model.fit(X_fold, y_fold)
        importances = model.feature_importances_

    elif model_name == "lightgbm":
        import lightgbm as lgb
        log.info(f"=== 1-fold feature importance filter (LightGBM, {n_estimators} trees) ===")
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators, max_depth=8, num_leaves=63,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, verbosity=-1, n_jobs=1,
        )
        model.fit(X_fold, y_fold)
        importances = model.feature_importances_

    elif model_name in ("catboost", "cb"):
        from catboost import CatBoostClassifier
        log.info(f"=== 1-fold feature importance filter (CatBoost, {n_estimators} trees) ===")
        model = CatBoostClassifier(
            n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
            task_type="GPU", devices="0", verbose=False, random_seed=RANDOM_SEED,
        )
        model.fit(X_fold, y_fold, verbose=False)
        importances = model.feature_importances_

    elif model_name in ("random_forest", "rf"):
        from sklearn.ensemble import RandomForestClassifier
        log.info(f"=== 1-fold feature importance filter (RF, {n_estimators} trees) ===")
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=10,
            random_state=RANDOM_SEED, n_jobs=1,
        )
        model.fit(X_fold, y_fold)
        importances = model.feature_importances_

    elif model_name in ("extra_trees", "et"):
        from sklearn.ensemble import ExtraTreesClassifier
        log.info(f"=== 1-fold feature importance filter (ExtraTrees, {n_estimators} trees) ===")
        model = ExtraTreesClassifier(
            n_estimators=n_estimators, max_depth=10,
            random_state=RANDOM_SEED, n_jobs=1,
        )
        model.fit(X_fold, y_fold)
        importances = model.feature_importances_

    elif model_name in ("hist_gbt", "hgbt"):
        from sklearn.ensemble import HistGradientBoostingClassifier
        log.info(f"=== 1-fold feature importance filter (HistGBT, {n_estimators} iterations) ===")
        model = HistGradientBoostingClassifier(
            max_iter=n_estimators, max_depth=6,
            random_state=RANDOM_SEED,
        )
        model.fit(X_fold, y_fold)
        importances = model.feature_importances_

    else:
        import lightgbm as lgb
        log.info(f"=== 1-fold feature importance filter (LightGBM proxy dla {model_name}, {n_estimators} trees) ===")
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators, max_depth=8, num_leaves=63,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, verbosity=-1, n_jobs=1,
        )
        model.fit(X_fold, y_fold)
        importances = model.feature_importances_

    kept = [f for f, imp in zip(feature_names, importances) if imp > min_importance]
    dropped = [f for f, imp in zip(feature_names, importances) if imp <= min_importance]

    log.info(f"Kept: {len(kept)} / {len(feature_names)} features")
    if dropped:
        log.info(f"Dropped ({len(dropped)}): {dropped}")

    return kept


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
    storage_path: Path | str | None = EXPERIMENTS_DIR / "optuna_studies.db",
    study_name: str | None = None,
) -> dict[str, Any]:
    """
    Optymalizuje hiperparametry modelu maksymalizując QWK przez 5-fold CV.

    Strategia maksymalnej jakości:
    - 5-fold StratifiedKFold ze STAŁYM seedem (fair comparison między trialami)
    - Cost-sensitive wagi klas dobierane przez Optunę
    - MultivariateTPE z 50 trial-ami startowymi
    - Bez prunera (NopPruner) — żaden trial nie jest zabijany
    - SQLite storage — wznowienie po przerwie

    Parameters
    ----------
    model_name : str
        'xgboost' | 'lightgbm' | 'random_forest' | 'ebm'
    X_train, y_train, X_val, y_val : pd.DataFrame, pd.Series
        X_val/y_val nieużywane (CV liczone wewnątrz na X_train).
    n_trials : int
    timeout : int, optional
    sample_weight_strategy : str
        Fallback gdy nie używamy cost-sensitive (np. dla finalnego retreningu).
    storage_path : Path
        SQLite dla persistent study.
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

    tune_logger.info(f"=== Optuna tuning: {model_name} (MAX QUALITY) ===")
    tune_logger.info(f"n_trials={n_trials}, timeout={timeout}, CV={CV_N_SPLITS}-fold")
    tune_logger.info(f"sample_weight_strategy='{sample_weight_strategy}' (fallback only)")

    suggest_fn = SUGGEST_FUNCTIONS[model_name]

    # STAŁY split CV — wszystkie triale ewaluowane na tych samych foldach
    skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    cv_splits = list(skf.split(X_train, y_train))
    tune_logger.info(f"Wygenerowano {len(cv_splits)} stałych splitów CV (seed={RANDOM_SEED})")

    # Pandas Series dla .iloc / .map
    y_train_series = pd.Series(y_train).reset_index(drop=True)
    X_train_reset = X_train.reset_index(drop=True)

    # ─── Funkcja celu ───
    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial)
        params["random_state"] = RANDOM_SEED

        # Cost-sensitive wagi klas (hiperparametry Optuny)
        class_weight_map = _suggest_class_weights(trial)

        try:
            if model_name == "xgboost":
                # Multi-GPU: foldy równolegle, każdy na innym GPU
                import os
                from joblib import Parallel, delayed

                # Wykryj liczbę GPU
                n_gpus = 0
                try:
                    import subprocess
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=5,
                    )
                    n_gpus = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
                except Exception:
                    pass

                if n_gpus >= CV_N_SPLITS:
                    available_cpus = len(os.sched_getaffinity(0))
                    n_jobs_per_fold = max(1, available_cpus // (CV_N_SPLITS * 2))

                    fold_args = []
                    for fold_idx, (tr_idx, cv_idx) in enumerate(cv_splits):
                        X_tr = X_train_reset.iloc[tr_idx]
                        X_cv = X_train_reset.iloc[cv_idx]
                        y_tr = y_train_series.iloc[tr_idx]
                        y_cv = y_train_series.iloc[cv_idx]
                        sw = y_tr.map(class_weight_map).values
                        fold_params = dict(params)
                        fold_params["n_jobs"] = n_jobs_per_fold
                        fold_params["device"] = f"cuda:{fold_idx}"
                        fold_args.append((model_name, fold_params, X_tr, y_tr, X_cv, y_cv, sw, None, fold_idx))

                    qwk_scores = Parallel(n_jobs=CV_N_SPLITS, backend="threading")(
                        delayed(_train_eval_fold)(*args) for args in fold_args
                    )
                else:
                    # Pojedyncze GPU — foldy sekwencyjnie
                    qwk_scores = []
                    for fold_idx, (tr_idx, cv_idx) in enumerate(cv_splits):
                        X_tr = X_train_reset.iloc[tr_idx]
                        X_cv = X_train_reset.iloc[cv_idx]
                        y_tr = y_train_series.iloc[tr_idx]
                        y_cv = y_train_series.iloc[cv_idx]
                        sw = y_tr.map(class_weight_map).values
                        fold_params = dict(params)
                        fold_params["n_jobs"] = max(1, os.cpu_count() // 2)
                        qwk = _train_eval_fold(
                            model_name, fold_params, X_tr, y_tr, X_cv, y_cv, sw,
                            tune_logger, fold_idx,
                        )
                        qwk_scores.append(qwk)
            elif model_name == "catboost":
                # Multi-GPU: foldy równolegle, każdy na innym GPU
                import os
                from joblib import Parallel, delayed

                n_gpus = 0
                try:
                    import subprocess
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=5,
                    )
                    n_gpus = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
                except Exception:
                    pass

                if n_gpus >= CV_N_SPLITS:
                    available_cpus = len(os.sched_getaffinity(0))
                    n_jobs_per_fold = max(1, available_cpus // (CV_N_SPLITS * 2))

                    fold_args = []
                    for fold_idx, (tr_idx, cv_idx) in enumerate(cv_splits):
                        X_tr = X_train_reset.iloc[tr_idx]
                        X_cv = X_train_reset.iloc[cv_idx]
                        y_tr = y_train_series.iloc[tr_idx]
                        y_cv = y_train_series.iloc[cv_idx]
                        sw = y_tr.map(class_weight_map).values
                        fold_params = dict(params)
                        fold_params["task_type"] = "GPU"
                        fold_params["devices"] = str(fold_idx)
                        fold_params["_fold_idx"] = fold_idx
                        fold_args.append((model_name, fold_params, X_tr, y_tr, X_cv, y_cv, sw, None, fold_idx))

                    qwk_scores = Parallel(n_jobs=CV_N_SPLITS, backend="threading")(
                        delayed(_train_eval_fold)(*args) for args in fold_args
                    )
                else:
                    # Pojedyncze GPU — foldy sekwencyjnie
                    qwk_scores = []
                    for fold_idx, (tr_idx, cv_idx) in enumerate(cv_splits):
                        X_tr = X_train_reset.iloc[tr_idx]
                        X_cv = X_train_reset.iloc[cv_idx]
                        y_tr = y_train_series.iloc[tr_idx]
                        y_cv = y_train_series.iloc[cv_idx]
                        sw = y_tr.map(class_weight_map).values
                        fold_params = dict(params)
                        fold_params["task_type"] = "GPU"
                        fold_params["_fold_idx"] = fold_idx
                        qwk = _train_eval_fold(
                            model_name, fold_params, X_tr, y_tr, X_cv, y_cv, sw,
                            tune_logger,
                        )
                        qwk_scores.append(qwk)

            elif model_name == "lightgbm":
                import os
                from joblib import Parallel, delayed

                # LightGBM GPU (OpenCL) nie działa na WSL i jest wolny na NVIDIA.
                # Zawsze CPU — foldy równoległe z OMP capped.
                available_cpus = len(os.sched_getaffinity(0))
                omp_per_fold = max(1, min(available_cpus // CV_N_SPLITS, 64))
                os.environ["OMP_NUM_THREADS"] = str(omp_per_fold)
                for var in ["MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
                    os.environ[var] = "1"
                fold_args = []
                for fold_idx, (tr_idx, cv_idx) in enumerate(cv_splits):
                    X_tr = X_train_reset.iloc[tr_idx]
                    X_cv = X_train_reset.iloc[cv_idx]
                    y_tr = y_train_series.iloc[tr_idx]
                    y_cv = y_train_series.iloc[cv_idx]
                    sw = y_tr.map(class_weight_map).values
                    fold_params = dict(params)
                    fold_params["n_jobs"] = omp_per_fold
                    fold_args.append((model_name, fold_params, X_tr, y_tr, X_cv, y_cv, sw, tune_logger, fold_idx))
                qwk_scores = Parallel(n_jobs=CV_N_SPLITS, backend="threading")(
                    delayed(_train_eval_fold)(*args) for args in fold_args
                )

            elif model_name == "random_forest":
                import os
                from joblib import Parallel, delayed

                n_gpus = 0
                try:
                    import subprocess
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=5,
                    )
                    n_gpus = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
                except Exception:
                    pass

                cuml_available = False
                try:
                    import cuml  # noqa: F401
                    cuml_available = True
                except ImportError:
                    pass

                use_gpu = n_gpus > 0 and cuml_available

                if use_gpu:
                    fold_args = []
                    for fold_idx, (tr_idx, cv_idx) in enumerate(cv_splits):
                        X_tr = X_train_reset.iloc[tr_idx]
                        X_cv = X_train_reset.iloc[cv_idx]
                        y_tr = y_train_series.iloc[tr_idx]
                        y_cv = y_train_series.iloc[cv_idx]
                        sw = y_tr.map(class_weight_map).values
                        fold_params = dict(params)
                        fold_params["__cuml_gpu_id"] = fold_idx if n_gpus >= CV_N_SPLITS else 0
                        fold_args.append((model_name, fold_params, X_tr, y_tr, X_cv, y_cv, sw, None, fold_idx))

                    if n_gpus >= CV_N_SPLITS:
                        qwk_scores = Parallel(n_jobs=CV_N_SPLITS, backend="threading")(
                            delayed(_train_eval_fold)(*args) for args in fold_args
                        )
                    else:
                        qwk_scores = [_train_eval_fold(*args) for args in fold_args]
                else:
                    available_cpus = len(os.sched_getaffinity(0))
                    omp_per_fold = max(1, min(available_cpus // CV_N_SPLITS, 64))
                    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
                        os.environ[var] = "1"
                    fold_args = []
                    for fold_idx, (tr_idx, cv_idx) in enumerate(cv_splits):
                        X_tr = X_train_reset.iloc[tr_idx]
                        X_cv = X_train_reset.iloc[cv_idx]
                        y_tr = y_train_series.iloc[tr_idx]
                        y_cv = y_train_series.iloc[cv_idx]
                        sw = y_tr.map(class_weight_map).values
                        fold_params = dict(params)
                        fold_params["n_jobs"] = omp_per_fold
                        fold_args.append((model_name, fold_params, X_tr, y_tr, X_cv, y_cv, sw, None, fold_idx))
                    qwk_scores = Parallel(n_jobs=CV_N_SPLITS, backend="loky")(
                        delayed(_train_eval_fold)(*args) for args in fold_args
                    )

            else:
                import os
                from joblib import Parallel, delayed
                available_cpus = len(os.sched_getaffinity(0))
                omp_per_fold = max(1, min(available_cpus // CV_N_SPLITS, 64))
                for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
                    os.environ[var] = "1"
                fold_args = []
                for fold_idx, (tr_idx, cv_idx) in enumerate(cv_splits):
                    X_tr = X_train_reset.iloc[tr_idx]
                    X_cv = X_train_reset.iloc[cv_idx]
                    y_tr = y_train_series.iloc[tr_idx]
                    y_cv = y_train_series.iloc[cv_idx]
                    sw = y_tr.map(class_weight_map).values
                    fold_params = dict(params)
                    fold_params["n_jobs"] = omp_per_fold
                    fold_args.append((model_name, fold_params, X_tr, y_tr, X_cv, y_cv, sw, None, fold_idx))
                qwk_scores = Parallel(n_jobs=CV_N_SPLITS, backend="loky")(
                    delayed(_train_eval_fold)(*args) for args in fold_args
                )

            qwk_mean = float(np.mean(qwk_scores))
            qwk_std = float(np.std(qwk_scores))

            tune_logger.info(
                f"Trial {trial.number:>4}: QWK={qwk_mean:.4f} (±{qwk_std:.4f}), "
                f"folds={[f'{s:.3f}' for s in qwk_scores]}"
            )
            return qwk_mean

        except Exception as e:
            tune_logger.error(f"Trial {trial.number} błąd: {type(e).__name__}: {e}")
            return -1.0

    # ─── Konfiguracja Optuny (MAX QUALITY) ───
    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_SEED,
        multivariate=True,          # modeluje korelacje między hiperparametrami
        group=True,                  # grupuje conditional params (np. DART-only)
        n_startup_trials=50,         # 50 losowych prób przed bayesowskim TPE
        n_ei_candidates=48,          # 2x więcej kandydatów per iteracja (default=24)
        warn_independent_sampling=False,
    )
    pruner = optuna.pruners.NopPruner()  # żaden trial nie jest zabijany przedwcześnie

    if storage_path is not None:
        storage_url = f"sqlite:///{Path(storage_path).resolve().as_posix()}"
    else:
        storage_url = None

    if study_name is None:
        study_name = f"{model_name}_max_quality_{run_id}"

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
