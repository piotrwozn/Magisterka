"""
Tuning hiperparametrów Optuną (k-fold CV na zbiorze dopasowania z TRAIN).

Zasady:
    - Każdy fold trenuje ten sam wrapper modelu co finalny trening
      (`get_model(...).fit_estimator`): te same parametry domyślne, ten sam wybór CPU/GPU
      i ten sam powrót GPU → CPU — dostrojone parametry zawsze pasują do `run_training`.
    - Early stopping (XGBoost / LightGBM / CatBoost) na wycinku EARLY_STOPPING_FRACTION
      z TRENINGOWEJ części foldu — nigdy na foldzie ocenianym (ten sam protokół co finał).
    - Domyślny cel: wieloklasowy LOG LOSS (minimalizacja; ważony, gdy trening używa wag).
      Alternatywa: --tune-metric qwk (wtedy Optuna dobiera też wagi klas).
    - Foldy stratyfikowane po ESI × oddział (`strata` z run_training), stałe ziarno.
    - MedianPruner przerywa słabe próby po pierwszych foldach (nigdy po ostatnim);
      EBM: foldy równolegle (joblib/loky), bez przerywania.

Wznawianie i odporność (nocny przebieg bez nadzoru):
    - Badanie identyfikuje KONFIGURACJA (model, cechy, cel, wagi, liczba wierszy, foldy, filtr,
      wersja przestrzeni, odcisk danych, profil) — zapisana w badaniu i sprawdzana przy wczytaniu.
    - `n_trials` i `timeout` to budżet CAŁEGO badania: wznowienie dokańcza tylko brakujące
      próby i brakujący czas.
    - SQLite (RDBStorage) z pulsem: próba przerwana awarią procesu → FAIL + jedno ponowienie.
    - Błąd próby nie przerywa badania (catch); 5 błędów z rzędu → koniec badania.
    - Plik blokady (PID) — jedno badanie = jeden proces; nieaktualna blokada jest przejmowana.
    - Wynik w `logs/optuna/<badanie>.best.json` (zapis atomowy) — czytelny bez Optuny.

Zbiory `val` i `test` nie są używane — tuning widzi wyłącznie TRAIN.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import re
import socket
import time
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState
from sklearn.metrics import cohen_kappa_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.models import MODEL_REGISTRY, get_model, resolve_model_name
from src.models.base import ITERATION_CAP_WARN_FRACTION, compute_sample_weights
from src.utils.config import (
    CLASS_NAMES,
    DATA_MANIFEST_JSON,
    EARLY_STOPPING_FRACTION,
    EBM_DEFAULT_PARAMS,
    LGBM_DEFAULT_PARAMS,
    N_CLASSES,
    OPTUNA_GRACE_PERIOD_S,
    OPTUNA_HEARTBEAT_INTERVAL_S,
    OPTUNA_MAX_CONSECUTIVE_FAILS,
    OPTUNA_MAX_RETRY,
    OPTUNA_PRUNER_STARTUP_TRIALS,
    OPTUNA_STORAGE_DIR,
    RANDOM_SEED,
    RF_DEFAULT_PARAMS,
    SEARCH_SPACE_VERSION,
    TRAINING_LOGS_DIR,
    TUNE_CV_FOLDS,
    TUNE_FIXED_PARAMS,
    TUNE_RF_MAX_MODEL_BYTES,
)
from src.utils.logger import close_training_logger, get_logger, get_training_logger

log = get_logger(__name__)

TUNE_METRICS = ("logloss", "qwk")
FINISHED_STATES = (TrialState.COMPLETE, TrialState.PRUNED)

# Liczba strojonych wymiarów (start TPE: n_startup = max(10, 2·d))
SEARCH_DIMENSIONS: dict[str, int] = {
    "xgboost": 9,
    "lightgbm": 9,
    "catboost": 7,
    "random_forest": 4,
    "ebm": 5,
    "logreg": 1,
}
CLASS_WEIGHT_DIMENSIONS = 4     # cw_esi1, cw_esi2, cw_esi4, cw_esi5 (tylko cel 'qwk')

# Szacunek pamięci jednego węzła drzewa sklearn (Node 64 B + 5 wartości klas × 8 B)
_RF_BYTES_PER_NODE = 104


# ─────────────────────────────────────────
# Przestrzenie hiperparametrów (zmiana → podbij SEARCH_SPACE_VERSION w config.py)
# ─────────────────────────────────────────
def _suggest_xgboost(trial: optuna.Trial, feature_set: str) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        # Jednostki hesjanu: przy softmax hesjan ≤ 0.25, więc 200 ≈ ≥ 800 wierszy w liściu
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 200.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
    }


def _suggest_lightgbm(trial: optuna.Trial, feature_set: str) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 511, log=True),
        # Małe liście wygrywają (ESI 1 = 0,9%); optimum ~ √n — skalowane przed finałem
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 1000, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 1.0, log=True),
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
    }


def _suggest_catboost(trial: optuna.Trial, feature_set: str) -> dict[str, Any]:
    # grow_policy zostaje SymmetricTree (z config) — to inny inductive bias niż XGB/LGBM
    params: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bernoulli", "Bayesian"]),
    }
    if params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    else:
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 5.0)
    return params


def _suggest_random_forest(trial: optuna.Trial, feature_set: str) -> dict[str, Any]:
    # n_estimators NIE jest strojone (więcej drzew nie pogarsza log loss): 300 w CV, 500 w finale.
    # Zestaw 'full' (~1000 cech): ułamek 0.3 cech na 400 tys. wierszy to godziny na jeden model.
    if feature_set == "full":
        max_features = ["sqrt", "log2", 0.05, 0.1]
    else:
        max_features = ["sqrt", "log2", 0.1, 0.2, 0.3]
    return {
        "max_features": trial.suggest_categorical("max_features", max_features),
        # Bezwzględne optimum ≈ 3 (nie skaluje się z n); górna granica 40
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 40, log=True),
        "max_samples": trial.suggest_float("max_samples", 0.2, 0.8),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }


def _suggest_ebm(trial: optuna.Trial, feature_set: str) -> dict[str, Any]:
    # interactions = 0 na stałe: wieloklasowy EBM nie pokazuje par cech w wyjaśnieniach,
    # więc model z parami przewidywałby termami, których wyjaśnienie nie zawiera.
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "max_bins": trial.suggest_categorical("max_bins", [256, 512, 1024]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 100, log=True),
        "max_leaves": trial.suggest_int("max_leaves", 2, 4),
        "smoothing_rounds": trial.suggest_int("smoothing_rounds", 0, 500, step=50),
    }


def _suggest_logreg(trial: optuna.Trial, feature_set: str) -> dict[str, Any]:
    return {"C": trial.suggest_float("C", 1e-4, 10.0, log=True)}


SUGGEST_FUNCTIONS: dict[str, Callable[[optuna.Trial, str], dict[str, Any]]] = {
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


def n_startup_trials(model_name: str, metric: str = "logloss") -> int:
    """Liczba losowych prób na start TPE: max(10, 2·d), d = liczba strojonych wymiarów."""
    d = SEARCH_DIMENSIONS[resolve_model_name(model_name)]
    if metric == "qwk":
        d += CLASS_WEIGHT_DIMENSIONS
    return max(10, 2 * d)


# ─────────────────────────────────────────
# Trening jednego foldu
# ─────────────────────────────────────────
def _carve_es_slice(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    sample_weight: np.ndarray | None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray | None]:
    """Wycinek early stopping (stratyfikowany po klasie) z TRENINGOWEJ części foldu."""
    idx = np.arange(len(X_tr))
    try:
        fit_idx, es_idx = train_test_split(
            idx, test_size=EARLY_STOPPING_FRACTION, random_state=RANDOM_SEED, stratify=y_tr
        )
    except ValueError:  # klasa z 1 wierszem (mikroskopijna próbka) — bez stratyfikacji
        fit_idx, es_idx = train_test_split(idx, test_size=EARLY_STOPPING_FRACTION, random_state=RANDOM_SEED)
    fit_idx, es_idx = np.sort(fit_idx), np.sort(es_idx)
    w_fit = None if sample_weight is None else np.asarray(sample_weight)[fit_idx]
    return X_tr.iloc[fit_idx], y_tr[fit_idx], X_tr.iloc[es_idx], y_tr[es_idx], w_fit


def fit_predict_fold(
    model_name: str,
    params: dict[str, Any],
    X_tr: pd.DataFrame,
    y_tr: np.ndarray | pd.Series,
    X_va: pd.DataFrame,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Trenuje wrapper modelu na treningowej części foldu i zwraca prawdopodobieństwa dla X_va.

    Modele z early stopping dostają wycinek ES z (X_tr, y_tr) — fold X_va służy wyłącznie
    do oceny. Błąd GPU → ponowienie na CPU (BaseTriageModel.fit_estimator).

    Returns
    -------
    proba : np.ndarray (n_va, 5), float64
    info : dict — n_train, n_es, best_iteration, iteration_cap, actual_device, device_fallback
           (+ nodes_per_tree dla Random Forest)
    """
    wrapper = get_model(model_name, params=params)
    y_arr = np.asarray(y_tr).astype(int).ravel()
    X_es, y_es = None, None
    X_f, y_f, w_f = X_tr, y_arr, sample_weight
    if wrapper.supports_early_stopping:
        X_f, y_f, X_es, y_es, w_f = _carve_es_slice(X_tr, y_arr, sample_weight)

    with warnings.catch_warnings():
        # interpret ostrzega przy każdym fit o brakach (NaN) — w tuningu to szum
        warnings.filterwarnings("ignore", message="Missing values detected")
        wrapper.fit_estimator(X_f, y_f, X_es, y_es, sample_weight=w_f)
    proba = wrapper.predict_proba(X_va)

    info: dict[str, Any] = {
        "n_train": int(len(X_f)),
        "n_es": 0 if X_es is None else int(len(X_es)),
        "best_iteration": wrapper.best_iteration,
        "iteration_cap": wrapper.iteration_cap,
        "actual_device": wrapper.actual_device,
        "device_fallback": wrapper.device_fallback,
    }
    if model_name == "random_forest":
        trees = getattr(wrapper.model, "estimators_", [])
        info["nodes_per_tree"] = float(np.mean([t.tree_.node_count for t in trees])) if trees else 0.0
    return proba, info


def _score(
    metric: str,
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """logloss (ważony, gdy podano wagi — ten sam cel co trening z wagami) albo QWK na argmax."""
    if metric == "logloss":
        return float(log_loss(y_true, proba, labels=list(range(N_CLASSES)), sample_weight=sample_weight))
    return float(cohen_kappa_score(y_true, proba.argmax(axis=1), weights="quadratic"))


def _oof_summary(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Metryki bezpieczeństwa na predykcjach out-of-fold (zapisywane w próbie Optuny)."""
    from src.evaluation.metrics import decide, expected_cost, undertriage_rate

    pred_argmax = decide(proba, "argmax")
    pred_cost = decide(proba, "cost")
    esi1 = y_true == 0
    out = {
        "qwk_argmax": float(cohen_kappa_score(y_true, pred_argmax, weights="quadratic")),
        "qwk_cost": float(cohen_kappa_score(y_true, pred_cost, weights="quadratic")),
        "expected_cost_cost": expected_cost(y_true, pred_cost),
        "undertriage_cost": undertriage_rate(y_true, pred_cost),
        "esi1_recall_cost": float((pred_cost[esi1] == 0).mean()) if esi1.any() else float("nan"),
    }
    return {k: (None if isinstance(v, float) and math.isnan(v) else round(float(v), 6)) for k, v in out.items()}


# ─────────────────────────────────────────
# EBM: foldy równolegle
# ─────────────────────────────────────────
def _ebm_fold_workers(params: dict[str, Any], n_folds: int) -> tuple[int, int]:
    """
    (liczba równoległych foldów, wątki EBM na fold).

    EBM zrównolegla się tylko po workach (outer_bags) — przy 8 workach i 24 rdzeniach
    jeden fold zajmuje 8 rdzeni, więc 3 foldy liczone razem wykorzystują cały CPU.
    Budżet wątków = n_jobs EBM (config.N_JOBS); < 2 → foldy sekwencyjnie.
    """
    total = int(params.get("n_jobs") or 1)
    if total < 2 or n_folds < 2:
        return 1, max(total, 1)
    workers = min(n_folds, total)
    return workers, max(1, total // workers)


def _parallel_infra_errors() -> tuple[type[BaseException], ...]:
    """Błędy infrastruktury joblib/loky (nie modelu) — po nich foldy liczone sekwencyjnie."""
    errors: list[type[BaseException]] = [OSError, ImportError]
    try:
        from joblib.externals.loky.process_executor import BrokenProcessPool, TerminatedWorkerError

        errors += [BrokenProcessPool, TerminatedWorkerError]
    except ImportError:
        pass
    try:
        from pickle import PicklingError

        errors.append(PicklingError)
    except ImportError:
        pass
    return tuple(errors)


def _run_folds_parallel(
    model_name: str,
    params: dict[str, Any],
    jobs: list[tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray | None]],
    n_workers: int,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """
    Foldy w osobnych procesach (loky — działa tak samo na Windows i Linux).
    W procesie roboczym EBM zrównolegla worki wątkami (joblib przełącza zagnieżdżony
    Parallel na wątki; natywny boosting zwalnia GIL).
    """
    from joblib import Parallel, delayed

    return Parallel(n_jobs=n_workers, backend="loky", max_nbytes=None)(
        delayed(fit_predict_fold)(model_name, params, X_tr, y_tr, X_va, w) for X_tr, y_tr, X_va, w in jobs
    )


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
    Odrzuca cechy, których LightGBM (na połowie zbioru) nie użył ani razu.

    LightGBM służy jako szybki, uniwersalny filtr niezależnie od modelu docelowego.
    """
    import lightgbm as lgb

    X_half, _, y_half, _ = train_test_split(
        X[feature_names], y, train_size=0.5, random_state=RANDOM_SEED, stratify=y
    )
    proxy = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=500, learning_rate=0.1, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=RANDOM_SEED, verbosity=-1, n_jobs=LGBM_DEFAULT_PARAMS.get("n_jobs"),
    )
    proxy.fit(X_half, y_half)
    importances = proxy.booster_.feature_importance(importance_type="split")
    kept = [f for f, imp in zip(feature_names, importances, strict=True) if imp > min_importance]
    dropped = [f for f in feature_names if f not in set(kept)]
    log.info(f"Filtr cech (LightGBM, połowa zbioru): zostaje {len(kept)} / {len(feature_names)}")
    if dropped:
        log.info(f"Odrzucone ({len(dropped)}): {dropped[:30]}{' …' if len(dropped) > 30 else ''}")
    return kept


# ─────────────────────────────────────────
# Klucz badania, odcisk danych, zapis atomowy
# ─────────────────────────────────────────
def data_fingerprint(manifest_path: Path | None = None) -> str:
    """sha1[:10] pliku manifest.json splitów (zmienia się przy każdym przygotowaniu danych)."""
    path = Path(manifest_path) if manifest_path is not None else DATA_MANIFEST_JSON
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    except OSError:
        return "brak-danych"


def _json_ready(value: Any) -> Any:
    """Wartość zgodna z JSON (porównanie konfiguracji po odczycie z bazy Optuny)."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:8]


def build_study_name(config: dict[str, Any]) -> str:
    """Czytelna nazwa badania z konfiguracji + skrót całej konfiguracji (unikalność)."""
    readable = (
        f"{config['model']}__{config['feature_set']}__{config['metric']}__w-{config['weights']}"
        f"__n{config['n_rows']}__k{config['n_folds']}__filter{int(bool(config['filter']))}"
        f"__ss{config['search_space']}__data-{config['data']}__{config['profile']}"
    )
    return re.sub(r"[^\w.-]", "_", f"{readable}__{_config_hash(config)}")


def _safe_file_stem(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name)


def write_json_atomic(path: Path, payload: Any, retries: int = 5) -> Path:
    """
    JSON przez plik tymczasowy + os.replace (czytelnik nigdy nie widzi połowy pliku).

    Windows: os.replace zawodzi, gdy plik docelowy jest akurat otwarty (antywirus, Excel) —
    kilka ponowień z krótką przerwą.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return path
        except PermissionError:
            if attempt == retries - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.2 * (attempt + 1))
    return path


# ─────────────────────────────────────────
# Blokada badania (jeden proces na badanie)
# ─────────────────────────────────────────
def _process_start_time(pid: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:  # noqa: BLE001 — brak psutil albo procesu
        return None


def _boot_time() -> float | None:
    """Czas startu systemu (epoch s) — blokada sprzed restartu komputera jest zawsze nieaktualna."""
    try:
        import psutil

        return float(psutil.boot_time())
    except Exception:  # noqa: BLE001 — brak psutil
        pass
    if os.name == "nt":
        try:
            import ctypes

            ticks = ctypes.windll.kernel32.GetTickCount64
            ticks.restype = ctypes.c_ulonglong
            return time.time() - ticks() / 1000.0
        except Exception:  # noqa: BLE001
            return None
    try:
        for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
            if line.startswith("btime"):
                return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _process_name(pid: int) -> str | None:
    """Nazwa/ścieżka programu procesu (do odróżnienia Pythona od procesu z ponownie użytym PID)."""
    try:
        import psutil

        return str(psutil.Process(pid).name())
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 — proces zniknął / brak dostępu
        return None
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel32.QueryFullProcessImageNameW.argtypes = (
                wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD))
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return None
            try:
                size = wintypes.DWORD(1024)
                buffer = ctypes.create_unicode_buffer(size.value)
                if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                    return buffer.value
                return None
            finally:
                kernel32.CloseHandle(handle)
        except Exception:  # noqa: BLE001
            return None
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[0].decode("utf-8", "replace") or None
    except OSError:
        return None


def _pid_alive(pid: int) -> bool:
    """Czy proces o danym PID działa (bez psutil: Windows — OpenProcess, POSIX — kill(pid, 0))."""
    if pid <= 0:
        return False
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except ImportError:
        pass
    if os.name == "nt":
        # UWAGA: os.kill(pid, 0) na Windows ZABIJA proces — tylko WinAPI
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5             # ACCESS_DENIED = proces istnieje
        try:
            code = wintypes.DWORD()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == 259                    # STILL_ACTIVE
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class StudyLock:
    """
    Plik `<badanie>.lock` z PID właściciela. Żywy właściciel → RuntimeError;
    martwy (awaria, restart komputera) → blokada przejmowana z ostrzeżeniem.
    """

    def __init__(self, path: Path, study_name: str):
        self.path = Path(path)
        self.study_name = study_name
        self.acquired = False

    def _holder(self) -> dict | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def _holder_alive(self, holder: dict | None) -> bool:
        if holder is None:                                   # pusty/uszkodzony plik po awarii
            return False
        try:
            pid = int(holder.get("pid", -1))
        except (TypeError, ValueError):
            return False
        if holder.get("host") not in (None, socket.gethostname()):
            return True                                      # inna maszyna — nie da się sprawdzić
        if pid == os.getpid():
            return False                                     # pozostałość w tym samym procesie
        if not _pid_alive(pid):
            return False
        # PID bywa ponownie użyty (zwłaszcza po restarcie Windows) — sprawdź, czy to ten sam proces
        boot_then, boot_now = holder.get("boot_time"), _boot_time()
        if boot_then is not None and boot_now is not None and abs(float(boot_then) - boot_now) > 300:
            return False                                     # blokada sprzed restartu komputera
        name = _process_name(pid)
        if name and "python" not in Path(name).name.lower():
            return False                                     # inny program dostał ten PID
        recorded, current = holder.get("proc_start"), _process_start_time(pid)
        if recorded is not None and current is not None and abs(float(recorded) - current) > 1.0:
            return False                                     # inny proces Pythona z tym PID
        return True

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "proc_start": _process_start_time(os.getpid()),
            "boot_time": _boot_time(),
            "study": self.study_name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                holder = self._holder()
                if self._holder_alive(holder):
                    raise RuntimeError(
                        f"Badanie Optuny '{self.study_name}' jest właśnie używane przez proces "
                        f"PID {holder.get('pid')} ({holder.get('host')}, od {holder.get('started_at')}). "
                        f"Jedno badanie = jeden proces. Jeśli ten proces już nie działa, usuń {self.path}."
                    ) from None
                log.warning(f"Przejmuję nieaktualną blokadę {self.path.name} "
                            f"(PID {None if holder is None else holder.get('pid')} nie działa)")
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            self.acquired = True
            return
        raise RuntimeError(f"Nie udało się założyć blokady {self.path}")

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            self.path.unlink()
        except OSError:
            pass
        self.acquired = False


# ─────────────────────────────────────────
# Magazyn Optuny (SQLite z pulsem)
# ─────────────────────────────────────────
def _make_storage(storage_path: Path) -> tuple[optuna.storages.BaseStorage, Callable | None]:
    """
    RDBStorage z pulsem i ponowieniem przerwanej próby (zgodne z optuna 3.5 – 5.x).

    optuna ≥ 4.9: `heartbeat_stale_trial_callback` + RetryHeartbeatStaleTrialCallback,
    starsze: `failed_trial_callback` + RetryFailedTrialCallback.
    """
    url = f"sqlite:///{Path(storage_path).resolve().as_posix()}"
    storages = optuna.storages
    callback_cls = getattr(storages, "RetryHeartbeatStaleTrialCallback", None) \
        or getattr(storages, "RetryFailedTrialCallback", None)
    accepted = inspect.signature(storages.RDBStorage.__init__).parameters
    kwargs: dict[str, Any] = {"engine_kwargs": {"connect_args": {"timeout": 60}}}
    if "heartbeat_interval" in accepted:
        kwargs["heartbeat_interval"] = OPTUNA_HEARTBEAT_INTERVAL_S
        kwargs["grace_period"] = OPTUNA_GRACE_PERIOD_S
    retry = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # ExperimentalWarning / FutureWarning (zmiany API)
        if callback_cls is not None:
            retry = callback_cls(max_retry=OPTUNA_MAX_RETRY)
            if "heartbeat_stale_trial_callback" in accepted:
                kwargs["heartbeat_stale_trial_callback"] = retry
            elif "failed_trial_callback" in accepted:
                kwargs["failed_trial_callback"] = retry
            else:
                retry = None
        storage = storages.RDBStorage(url, **kwargs)
    return storage, retry


def _fail_orphaned_trials(study: optuna.Study, retry: Callable | None, tune_logger) -> int:
    """
    Próby RUNNING po awarii procesu (blokada gwarantuje, że nikt inny ich nie liczy) → FAIL
    i jedno ponowienie z tymi samymi parametrami (bez czekania na koniec okresu pulsu).
    """
    count = 0
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        try:
            storage = study._storage            # noqa: SLF001 — publiczne API nie ma "fail"
            storage.set_trial_state_values(trial._trial_id, state=TrialState.FAIL)
            count += 1
            tune_logger.warning(f"Trial {trial.number:>4}: przerwany przez awarię poprzedniego procesu → FAIL")
            if retry is not None:
                retry(study, copy.deepcopy(storage.get_trial(trial._trial_id)))
        except Exception as exc:  # noqa: BLE001 — porządki nie mogą przerwać tuningu
            tune_logger.warning(f"Nie udało się oznaczyć próby {trial.number} jako FAIL: {exc}")
    return count


class _StopAfterConsecutiveFails:
    """Callback: `study.stop()` po N kolejnych próbach FAIL (np. stały błąd środowiska)."""

    def __init__(self, limit: int):
        self.limit = limit
        self.count = 0
        self.triggered = False

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        self.count = self.count + 1 if trial.state == TrialState.FAIL else 0
        if self.count >= self.limit:
            self.triggered = True
            study.stop()


class _ElapsedTracker:
    """Callback: łączny czas prób badania w user_attrs['elapsed_s'] (przeżywa restart)."""

    def __init__(self, base: float):
        self.total = float(base)

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.duration is not None:
            self.total += trial.duration.total_seconds()
            study.set_user_attr("elapsed_s", round(self.total, 1))


def _fold_strata(y: np.ndarray, strata: np.ndarray | None, min_count: int) -> np.ndarray:
    """
    Etykiety stratyfikacji: ESI × oddział; kombinacje z < min_count wierszami → samo ESI.
    """
    if strata is None:
        return y
    s = pd.Series(np.asarray(strata).astype(str))
    rare = s.map(s.value_counts()) < min_count
    s = s.where(~rare, "esi" + pd.Series(y).astype(str))
    if (s.map(s.value_counts()) < min_count).any():
        return y
    return s.to_numpy()


# ─────────────────────────────────────────
# Główna funkcja tuningu
# ─────────────────────────────────────────
def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    n_trials: int = 20,
    metric: str = "logloss",
    n_folds: int = TUNE_CV_FOLDS,
    max_rows: int | None = None,
    sample_weight_strategy: str = "none",
    timeout: float | None = None,
    storage_path: Path | str | None = None,
    study_name: str | None = None,
    *,
    feature_set: str = "triage",
    strata: np.ndarray | pd.Series | None = None,
    do_filter: bool = False,
    profile: str | None = None,
    n_final_fit: int | None = None,
    fixed_params: dict[str, Any] | None = None,
    data_fp: str | None = None,
) -> dict[str, Any]:
    """
    Optuna: k-fold CV na zbiorze dopasowania (część TRAIN).

    Parameters
    ----------
    n_trials : łączna liczba UKOŃCZONYCH prób (COMPLETE + PRUNED) w badaniu — wznowienie
        dokańcza brakujące.
    metric : 'logloss' (minimalizacja, domyślnie) | 'qwk' (maksymalizacja + wagi klas)
    max_rows : stratyfikowana próbka do tuningu (None = cały zbiór); finalny trening na całości.
    sample_weight_strategy : wagi przy celu 'logloss' (log loss foldu ważony tymi samymi wagami).
    timeout : łączny limit czasu badania w sekundach (sumowany między wznowieniami).
    storage_path : plik SQLite (None = pamięć, bez wznawiania, blokady i best.json).
    study_name : jawna nazwa badania (domyślnie z konfiguracji); niezgodna konfiguracja → błąd.
    feature_set, do_filter, profile : części klucza badania (nie zmieniają obliczeń poza RF max_features).
    strata : etykiety stratyfikacji foldów (np. ESI × oddział), wyrównane z X_train.
    n_final_fit : liczba wierszy finalnego treningu (strażnik rozmiaru RF; domyślnie len(X_train)).
    fixed_params : parametry stałe dla każdej próby (nadpisują TUNE_FIXED_PARAMS; część klucza).
    data_fp : odcisk danych (domyślnie sha1 manifest.json).

    Raises
    ------
    RuntimeError — brak ukończonej próby, badanie zajęte przez inny proces albo niezgodna
    konfiguracja badania o jawnej nazwie.
    """
    model_name = resolve_model_name(model_name)
    if metric not in TUNE_METRICS:
        raise ValueError(f"Nieznana metryka tuningu '{metric}'. Dostępne: {TUNE_METRICS}")
    if n_folds < 2:
        raise ValueError(f"n_folds musi być ≥ 2 (podano {n_folds})")
    if n_trials < 1:
        raise ValueError(f"n_trials musi być ≥ 1 (podano {n_trials})")

    # ── Dane: próbka do tuningu i foldy ──
    X = X_train.reset_index(drop=True)
    y = np.asarray(y_train).astype(int).ravel()
    strata_arr = None if strata is None else np.asarray(strata).ravel()
    if strata_arr is not None and len(strata_arr) != len(X):
        raise ValueError(f"len(strata)={len(strata_arr)} ≠ len(X)={len(X)}")
    if max_rows is not None and 0 < max_rows < len(X):
        idx, _ = train_test_split(np.arange(len(X)), train_size=max_rows, random_state=RANDOM_SEED,
                                  stratify=_fold_strata(y, strata_arr, 2))
        idx = np.sort(idx)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
        strata_arr = None if strata_arr is None else strata_arr[idx]
    fold_labels = _fold_strata(y, strata_arr, n_folds)
    folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED).split(X, fold_labels))

    supports_es = MODEL_REGISTRY[model_name].supports_early_stopping
    mean_fold_rows = float(np.mean([len(tr) for tr, _ in folds]))
    n_fold_train = int(round(mean_fold_rows * ((1 - EARLY_STOPPING_FRACTION) if supports_es else 1.0)))
    n_final = int(n_final_fit or len(X_train))

    fixed = {**TUNE_FIXED_PARAMS.get(model_name, {}), **(fixed_params or {})}
    direction = "minimize" if metric == "logloss" else "maximize"
    n_startup = n_startup_trials(model_name, metric)

    # ── Klucz badania ──
    config = _json_ready({
        "model": model_name,
        "feature_set": feature_set,
        "metric": metric,
        "weights": sample_weight_strategy,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_folds": int(n_folds),
        "filter": bool(do_filter),
        "search_space": int(SEARCH_SPACE_VERSION[model_name]),
        "data": data_fp or data_fingerprint(),
        "profile": profile or "custom",
        "fixed_params": fixed,
        "stratified_by": "esi_site" if strata_arr is not None else "esi",
    })
    name = study_name or build_study_name(config)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tune_logger, log_path = get_training_logger(model_name=f"{model_name}_tuning", run_id=run_id)
    tune_logger.info(f"Optuna: {model_name} | cel={metric} | {n_folds}-fold CV | n={len(X):,} "
                     f"(fold-train ≈ {n_fold_train:,}) | próby={n_trials} | limit czasu="
                     f"{'brak' if timeout is None else f'{timeout / 60:.0f} min'}")
    tune_logger.info(f"Badanie: {name}")

    lock: StudyLock | None = None
    storage: Any = None
    retry = None
    storage_dir: Path | None = None
    if storage_path is not None:
        storage_dir = Path(storage_path).parent
        storage_dir.mkdir(parents=True, exist_ok=True)
        lock = StudyLock(storage_dir / f"{_safe_file_stem(name)}.lock", name)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        if lock is not None:
            lock.acquire()
            storage, retry = _make_storage(Path(storage_path))

        pruner = optuna.pruners.MedianPruner(n_startup_trials=OPTUNA_PRUNER_STARTUP_TRIALS, n_warmup_steps=0)
        study = optuna.create_study(direction=direction, storage=storage, study_name=name,
                                    load_if_exists=True, pruner=pruner)

        # ── Zgodność konfiguracji ──
        stored = study.user_attrs.get("config")
        if stored is None and study.trials:
            raise RuntimeError(
                f"Badanie '{name}' ma {len(study.trials)} prób, ale nie ma zapisanej konfiguracji "
                f"(utworzone starszą wersją kodu). Użyj innej nazwy albo usuń badanie z {storage_path}."
            )
        if stored is not None and stored != config:
            diff = sorted(k for k in set(stored) | set(config) if stored.get(k) != config.get(k))
            raise RuntimeError(
                f"Badanie '{name}' ma inną konfigurację niż bieżący tuning (różnice: {diff}). "
                f"Próby nie zostaną wymieszane — zmień nazwę badania albo usuń je z {storage_path}."
            )
        if stored is None:
            study.set_user_attr("config", config)
        study.set_user_attr("n_trials_target", int(n_trials))
        study.set_user_attr("timeout_s", None if timeout is None else float(timeout))

        n_before = len(study.trials)          # ponowienia osieroconych prób liczą się do tej sesji
        if lock is not None:
            _fail_orphaned_trials(study, retry, tune_logger)

        # ── Budżet: brakujące próby i czas ──
        n_done = len(study.get_trials(deepcopy=False, states=FINISHED_STATES))
        remaining = max(0, int(n_trials) - n_done)
        elapsed_before = float(study.user_attrs.get("elapsed_s", 0.0) or 0.0)
        time_left = None if timeout is None else max(0.0, float(timeout) - elapsed_before)
        if n_done:
            tune_logger.info(f"Wznowienie: {n_done} ukończonych prób, {elapsed_before / 60:.1f} min — "
                             f"zostało {remaining} prób"
                             + ("" if time_left is None else f", {time_left / 60:.1f} min"))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # ExperimentalWarning (multivariate/group w starszej optunie)
            # Ziarno zależne od liczby prób w badaniu: wznowienie nie powtarza losowych prób startowych
            study.sampler = optuna.samplers.TPESampler(
                seed=RANDOM_SEED + len(study.trials), multivariate=True, group=True,
                n_startup_trials=n_startup,
            )

        objective = _make_objective(
            model_name=model_name, X=X, y=y, folds=folds, metric=metric, feature_set=feature_set,
            sample_weight_strategy=sample_weight_strategy, fixed=fixed, n_final=n_final,
            tune_logger=tune_logger,
        )
        fail_stopper = _StopAfterConsecutiveFails(OPTUNA_MAX_CONSECUTIVE_FAILS)
        elapsed = _ElapsedTracker(elapsed_before)
        session_start = time.time()
        timed_out = False
        if remaining > 0 and (time_left is None or time_left > 0):
            study.optimize(
                objective,
                # Bezpiecznik liczby podejść; właściwy limit to MaxTrialsCallback (ukończone próby)
                n_trials=3 * remaining + OPTUNA_MAX_CONSECUTIVE_FAILS,
                timeout=time_left,
                gc_after_trial=True,
                catch=(Exception,),
                callbacks=[
                    optuna.study.MaxTrialsCallback(int(n_trials), states=FINISHED_STATES),
                    elapsed, fail_stopper,
                ],
            )
            # Czas ścienny sesji (z narzutem próbkowania) — limit czasu Optuny liczy właśnie go
            session_wall = time.time() - session_start
            timed_out = time_left is not None and session_wall >= time_left
            study.set_user_attr("elapsed_s", round(max(elapsed.total, elapsed_before + session_wall), 1))
        n_session = len(study.trials) - n_before

        # ── Podsumowanie ──
        trials = study.get_trials(deepcopy=False)
        counts = {state: sum(t.state == state for t in trials)
                  for state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)}
        n_complete, n_pruned, n_fail = (counts[TrialState.COMPLETE], counts[TrialState.PRUNED],
                                        counts[TrialState.FAIL])
        n_finished = n_complete + n_pruned
        elapsed_total = float(study.user_attrs.get("elapsed_s", elapsed.total) or 0.0)
        if fail_stopper.triggered:
            stopped_reason = "consecutive_failures"
            tune_logger.error(f"{OPTUNA_MAX_CONSECUTIVE_FAILS} prób z rzędu zakończyło się błędem — "
                              f"przerywam badanie (szczegóły błędów w logu i w user_attrs prób)")
        elif n_finished >= n_trials:
            stopped_reason = "n_trials"
        elif timed_out or (timeout is not None and elapsed_total >= timeout - 1e-6):
            stopped_reason = "timeout"
        else:
            stopped_reason = "max_attempts"

        if n_complete == 0:
            errors = [t.user_attrs.get("error") for t in trials if t.state == TrialState.FAIL]
            last_error = next((e for e in reversed(errors) if e), None)
            raise RuntimeError(
                f"Optuna ({model_name}): brak ukończonej próby w badaniu '{name}' "
                f"(przerwane: {n_pruned}, błędy: {n_fail})"
                + (f"; ostatni błąd: {last_error}" if last_error else "")
            )

        best = study.best_trial
        best_iters = [b for b in (best.user_attrs.get("best_iter") or []) if b is not None]
        cap = best.user_attrs.get("iteration_cap")
        hit_cap = bool(cap and best_iters and max(best_iters) >= ITERATION_CAP_WARN_FRACTION * cap)
        if hit_cap:
            tune_logger.warning(f"Najlepsza próba: najlepsza iteracja {max(best_iters)} ≥ "
                                f"{ITERATION_CAP_WARN_FRACTION:.0%} limitu {cap} — limit iteracji zbyt niski")
        best_value_std = best.user_attrs.get("cv_std")
        tune_logger.info(f"Najlepszy {metric}: {best.value:.5f}"
                         + (f" (±{best_value_std:.5f})" if best_value_std is not None else "")
                         + f" — próba {best.number}")
        tune_logger.info(f"Najlepsze parametry: {best.params}")
        tune_logger.info(f"Próby: ukończone {n_complete}, przerwane {n_pruned}, błędy {n_fail} | "
                         f"łączny czas {elapsed_total / 60:.1f} min | koniec: {stopped_reason}")

        history_csv = TRAINING_LOGS_DIR / f"{model_name}_tuning" / f"{run_id}_trials.csv"
        try:
            history_csv.parent.mkdir(parents=True, exist_ok=True)
            study.trials_dataframe().to_csv(history_csv, index=False, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 — historia to tylko diagnostyka
            tune_logger.warning(f"Nie zapisano historii prób: {exc}")

        summary_attrs = ("qwk_argmax", "qwk_cost", "expected_cost_cost", "undertriage_cost",
                         "esi1_recall_cost", "actual_device", "device_fallback", "fold_scores")
        result: dict[str, Any] = {
            "study_name": name,
            "metric": metric,
            "direction": direction,
            "best_params": dict(best.params),
            "best_value": float(best.value),
            "best_value_std": best_value_std,
            "best_trial": int(best.number),
            "best_trial_attrs": {k: best.user_attrs.get(k) for k in summary_attrs},
            "best_iter": best_iters or None,
            "iteration_cap": cap,
            "hit_cap": hit_cap,
            "n_trials": n_finished,
            "n_trials_target": int(n_trials),
            "n_complete": n_complete,
            "n_pruned": n_pruned,
            "n_fail": n_fail,
            "n_trials_session": n_session,
            "n_startup_trials": n_startup,
            "n_rows": int(len(X)),
            "n_folds": int(n_folds),
            "n_fold_train": n_fold_train,
            "n_final_fit": n_final,
            "timeout_s": None if timeout is None else float(timeout),
            "elapsed_s": round(elapsed_total, 1),
            "stopped_reason": stopped_reason,
            "budget_exhausted": bool(n_finished >= n_trials
                                     or (timeout is not None and elapsed_total >= timeout - 1e-6)),
            "search_space_version": int(SEARCH_SPACE_VERSION[model_name]),
            "fixed_params": fixed,
            "config": config,
            "storage": None if storage_path is None else str(storage_path),
            "log_path": str(log_path),
            "history_csv": str(history_csv),
            "class_weight_keys": [f"cw_{c.lower()}" for c in CLASS_NAMES],
            "best_json": None,
        }
        if storage_dir is not None:
            best_json = storage_dir / f"{_safe_file_stem(name)}.best.json"
            payload = {**result, "best_json": str(best_json),
                       "written_at": datetime.now().isoformat(timespec="seconds")}
            write_json_atomic(best_json, payload)
            result["best_json"] = str(best_json)
            tune_logger.info(f"Wynik tuningu: {best_json}")
        return result
    finally:
        if storage is not None:
            # Zamknij połączenia SQLite (Windows blokuje otwarty plik bazy)
            try:
                storage.remove_session()
                storage.engine.dispose()
            except Exception:  # noqa: BLE001 — sprzątanie nie może przykryć właściwego błędu
                pass
        if lock is not None:
            lock.release()
        close_training_logger(tune_logger)


def _make_objective(
    *,
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    metric: str,
    feature_set: str,
    sample_weight_strategy: str,
    fixed: dict[str, Any],
    n_final: int,
    tune_logger,
) -> Callable[[optuna.Trial], float]:
    """Funkcja celu Optuny (zamknięcie na danych i foldach)."""
    suggest = SUGGEST_FUNCTIONS[model_name]
    weighted_score = metric == "logloss" and sample_weight_strategy != "none"
    state = {"parallel": model_name == "ebm"}
    n_folds = len(folds)

    def fold_inputs(tr: np.ndarray, va: np.ndarray, class_weights: dict[int, float] | None):
        y_tr = y[tr]
        w_tr = None
        if class_weights is not None or sample_weight_strategy != "none":
            w_tr = compute_sample_weights(y_tr, strategy=sample_weight_strategy, class_weights=class_weights)
        w_va = compute_sample_weights(y[va], strategy=sample_weight_strategy) if weighted_score else None
        return y_tr, w_tr, w_va

    def evaluate(trial: optuna.Trial, params: dict[str, Any], class_weights: dict[int, float] | None) -> float:
        scores: list[float] = []
        iters: list[int | None] = []
        devices: list[str] = []
        fallback: str | None = None
        oof_y: list[np.ndarray] = []
        oof_p: list[np.ndarray] = []
        cap: int | None = None

        def record(i: int, va: np.ndarray, proba: np.ndarray, info: dict, w_va) -> None:
            nonlocal fallback, cap
            scores.append(_score(metric, y[va], proba, w_va))
            iters.append(info.get("best_iteration"))
            devices.append(str(info.get("actual_device")))
            fallback = fallback or info.get("device_fallback")
            cap = info.get("iteration_cap") or cap
            oof_y.append(y[va])
            oof_p.append(proba)
            trial.set_user_attr("fold_scores", [round(s, 6) for s in scores])
            trial.set_user_attr("best_iter", iters)
            trial.set_user_attr("n_fold_train", info.get("n_train"))

        n_workers, threads = _ebm_fold_workers({**EBM_DEFAULT_PARAMS, **params}, n_folds) \
            if model_name == "ebm" else (1, 0)
        results = None
        if state["parallel"] and n_workers >= 2:
            # EBM: wszystkie foldy naraz (bez przerywania po foldzie)
            jobs, w_vas = [], []
            for tr, va in folds:
                y_tr, w_tr, w_va = fold_inputs(tr, va, class_weights)
                jobs.append((X.iloc[tr], y_tr, X.iloc[va], w_tr))
                w_vas.append(w_va)
            try:
                results = _run_folds_parallel(model_name, {**params, "n_jobs": threads}, jobs, n_workers)
            except _parallel_infra_errors() as exc:
                state["parallel"] = False
                tune_logger.warning(f"EBM: równoległe foldy nie działają ({type(exc).__name__}: {exc}) — "
                                    f"dalej foldy sekwencyjnie")
                results = None
            if results is not None:
                for i, ((_, va), (proba, info), w_va) in enumerate(zip(folds, results, w_vas, strict=True)):
                    record(i, va, proba, info, w_va)

        if results is None:
            for i, (tr, va) in enumerate(folds):
                y_tr, w_tr, w_va = fold_inputs(tr, va, class_weights)
                proba, info = fit_predict_fold(model_name, params, X.iloc[tr], y_tr, X.iloc[va], w_tr)
                record(i, va, proba, info, w_va)

                if model_name == "random_forest" and i == 0:
                    _rf_size_guard(trial, info, n_final, tune_logger)

                mean_so_far = float(np.mean(scores))
                trial.report(mean_so_far, step=i)
                # Po ostatnim foldzie nie przerywamy — w pełni oceniona próba zachowuje wynik
                if i < n_folds - 1 and trial.should_prune():
                    tune_logger.info(f"Trial {trial.number:>4}: przerwany po foldzie {i + 1} "
                                     f"({metric}={mean_so_far:.5f})")
                    raise optuna.TrialPruned()

        value = float(np.mean(scores))
        std = float(np.std(scores))
        trial.set_user_attr("cv_std", round(std, 6))
        trial.set_user_attr("actual_device", ",".join(sorted(set(devices))))
        trial.set_user_attr("iteration_cap", cap)
        if fallback:
            trial.set_user_attr("device_fallback", str(fallback)[:300])
        for key, val in _oof_summary(np.concatenate(oof_y), np.vstack(oof_p)).items():
            trial.set_user_attr(key, val)
        iters_txt = "" if not any(b is not None for b in iters) else f" | iteracje {iters}"
        tune_logger.info(f"Trial {trial.number:>4}: {metric}={value:.5f} (±{std:.5f}){iters_txt} {params}")
        return value

    def objective(trial: optuna.Trial) -> float:
        params = {**fixed, **suggest(trial, feature_set)}
        class_weights = _suggest_class_weights(trial) if metric == "qwk" else None
        try:
            return evaluate(trial, params, class_weights)
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            trial.set_user_attr("error", message[:500])
            tune_logger.warning(f"Trial {trial.number:>4}: BŁĄD — {message[:300]}")
            raise

    return objective


def _rf_size_guard(trial: optuna.Trial, info: dict, n_final: int, tune_logger) -> None:
    """Przerywa próbę RF, której finalny model (500 drzew na pełnych danych) byłby za duży."""
    nodes = float(info.get("nodes_per_tree") or 0.0)
    n_train = max(int(info.get("n_train") or 1), 1)
    n_trees = int(RF_DEFAULT_PARAMS.get("n_estimators") or 500)
    projected = nodes * (n_final / n_train) * n_trees * _RF_BYTES_PER_NODE
    trial.set_user_attr("rf_projected_bytes", float(projected))
    if projected > TUNE_RF_MAX_MODEL_BYTES:
        trial.set_user_attr("pruned_size", float(projected))
        tune_logger.info(f"Trial {trial.number:>4}: przerwany — finalny RF miałby ~{projected / 1e9:.1f} GB "
                         f"(limit {TUNE_RF_MAX_MODEL_BYTES / 1e9:.1f} GB)")
        raise optuna.TrialPruned()


__all__ = [
    "OPTUNA_STORAGE_DIR",
    "SEARCH_DIMENSIONS",
    "SUGGEST_FUNCTIONS",
    "StudyLock",
    "build_study_name",
    "data_fingerprint",
    "filter_features_by_importance_1fold",
    "fit_predict_fold",
    "n_startup_trials",
    "tune_model",
    "write_json_atomic",
]
