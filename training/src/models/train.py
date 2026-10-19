"""
Trening modeli ESI — jeden punkt wejścia (CLI: `scripts/03_train.py`).

    python scripts/03_train.py --model all                          # wszystkie modele, zestaw 'triage'
    python scripts/03_train.py --model lightgbm                     # jeden model
    python scripts/03_train.py --model all --tune                   # tuning, profil 'standard' (noc)
    python scripts/03_train.py --model all --tune --profile quick   # test całego pipeline'u (~30 min)
    python scripts/03_train.py --model all --tune --profile smoke   # test instalacji (minuty)
    python scripts/03_train.py --model xgboost --tune --n-trials 30 --tune-timeout 3600
    python scripts/03_train.py --model all --feature-set full

Przebieg dla każdego modelu:
    1. Dane: splity z `scripts/02_preprocess.py` (przy SOR_AI_NO_AUTO_PREPARE=1 brak → błąd).
    2. Budżet z profilu (config.TUNING_PROFILES; xgboost/catboost bez GPU → CPU_FALLBACK_BUDGETS).
       Jawne argumenty (n_trials, tune_rows, cv_folds, tune_timeout, train_sample, eval_sample)
       mają pierwszeństwo przed profilem.
    3. Zbiór dopasowania X_fit: modele z early stopping (XGBoost, LightGBM, CatBoost) — 90% TRAIN
       + 10% TRAIN na early stopping; RF, EBM i logreg — cały TRAIN (early stopping nie używają).
    4. (Opcjonalnie) tuning Optuną na X_fit: CV stratyfikowane po ESI × oddział, wznawialne
       (src/models/tuning.py). `val` i `test` nie biorą udziału w tuningu.
    5. Trening finalny, predykcje na `val` i `test`, metryki dla reguł argmax i koszt
       (+ podgrupa z kompletem parametrów życiowych, `vitals_complete == 1`).
    6. runs/<run>/: model, metryki, predykcje, raport, leaderboard_row.json, NA KOŃCU DONE.json;
       runs/leaderboard.csv przebudowywany ze wszystkich ukończonych runów (tylko metryki VAL).

`--filter` działa tylko z `--tune` (filtr cech jest częścią konfiguracji tuningu); bez `--tune`
jest pomijany z ostrzeżeniem.

Protokół łączenia modeli (ręcznie, zob. src/ensemble):
    wagi / progi / kalibrację dobieraj na `predictions_val.parquet`,
    `predictions_test.parquet` — wyłącznie do raportu końcowego.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.dataset import (
    early_stopping_split,
    ensure_prepared,
    get_xy,
    load_splits,
    manifest,
    resolve_feature_set,
    sample_frame,
)
from src.data.prepare import VITALS_COMPLETE_COL
from src.evaluation.metrics import decide, full_evaluation
from src.models import MODEL_REGISTRY, get_model, resolve_model_name
from src.models.base import ITERATION_CAP_WARN_FRACTION, compute_class_weights_dict
from src.models.tuning import data_fingerprint, write_json_atomic
from src.utils.config import (
    CLASS_NAMES,
    CPU_FALLBACK_BUDGETS,
    DATA_VERSION,
    DEFAULT_SAMPLE_WEIGHT_STRATEGY,
    DEFAULT_TUNING_PROFILE,
    OPTUNA_STORAGE_DIR,
    RUNS_DIR,
    TUNE_SQRT_SCALED_PARAMS,
    TUNING_PROFILES,
)
from src.utils.hardware import device_for, hardware_summary
from src.utils.logger import get_logger

log = get_logger(__name__)

DECISION_RULES = ("argmax", "cost")
PROBA_COLS = [f"p_{name.lower()}" for name in CLASS_NAMES]   # p_esi1 … p_esi5
PREDICTION_COLS = ["row_id", "site", "esi", "target", VITALS_COMPLETE_COL, *PROBA_COLS,
                   *(f"pred_{rule}" for rule in DECISION_RULES)]
LEADERBOARD_CSV = RUNS_DIR / "leaderboard.csv"
LEADERBOARD_ROW_JSON = "leaderboard_row.json"
DONE_JSON = "DONE.json"

# Metryki w leaderboardzie (VAL, obie reguły) i w raporcie
LEADERBOARD_METRICS = [
    "quadratic_weighted_kappa", "f1_macro", "auc_macro", "log_loss", "brier",
    "undertriage_rate", "critical_miss_rate", "overtriage_rate",
    "mistriage_under", "mistriage_over", "expected_cost",
]
# Podgrupa vitals_complete == 1 (komplet HR, SBP, RR przy triażu) — w leaderboardzie skrót
VC_LEADERBOARD_METRICS = [
    "quadratic_weighted_kappa", "auc_macro", "log_loss",
    "undertriage_rate", "critical_miss_rate", "expected_cost",
]
# Modele z obsługą GPU (urządzenie wybiera hardware.device_for)
GPU_MODELS = ("xgboost", "catboost")


# ─────────────────────────────────────────
# Budżet z profilu
# ─────────────────────────────────────────
def resolve_budget(
    model_name: str,
    profile: str = DEFAULT_TUNING_PROFILE,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Budżet tuningu modelu z profilu: n_trials, tune_rows (0 = cały X_fit), n_folds, timeout_s,
    train_sample, eval_sample.

    xgboost / catboost na CPU (device='cpu'; None → `hardware.device_for`, może uruchomić
    próbę GPU) → budżet z CPU_FALLBACK_BUDGETS, jeśli profil go definiuje.
    """
    model_name = resolve_model_name(model_name)
    if profile not in TUNING_PROFILES:
        raise ValueError(f"Nieznany profil '{profile}'. Dostępne: {list(TUNING_PROFILES)}")
    prof = TUNING_PROFILES[profile]
    budget: dict[str, Any] = dict(prof[model_name])
    if device is None:
        device = device_for(model_name) if model_name in GPU_MODELS else "cpu"
    fallback = CPU_FALLBACK_BUDGETS.get(profile, {}).get(model_name) if model_name in GPU_MODELS else None
    cpu_fallback = bool(device == "cpu" and fallback)
    if cpu_fallback:
        budget.update(fallback)
        log.warning(f"{model_name}: trening na CPU — budżet tuningu zastępczy "
                    f"(profil '{profile}': {fallback['n_trials']} prób, {fallback['tune_rows']:,} wierszy, "
                    f"limit {fallback['timeout_s'] / 3600:.1f} h)")
    return {
        "profile": profile,
        **budget,
        "train_sample": prof.get("train_sample"),
        "eval_sample": prof.get("eval_sample"),
        "device": device,
        "cpu_fallback_budget": cpu_fallback,
    }


def _scale_count_params(model_name: str, params: dict[str, Any], n_fit: int, n_fold_train: int | None) -> dict:
    """
    Parametry "liczności" (minimalna wielkość liścia) × sqrt(n_fit / n_fold_train) przed finałem:
    optimum rośnie ~ √n, a fold CV trenuje na mniejszej liczbie wierszy niż finalny model.
    Zmienia `params` w miejscu; zwraca opis zmian (pusty, gdy nic nie skalowano).
    """
    names = TUNE_SQRT_SCALED_PARAMS.get(model_name, [])
    if not names or not n_fold_train or n_fit <= n_fold_train:
        return {}
    factor = math.sqrt(n_fit / n_fold_train)
    changes: dict[str, dict] = {}
    for name in names:
        old = params.get(name)
        if old is None or isinstance(old, bool):
            continue
        new: int | float = old * factor
        new = max(1, int(round(new))) if isinstance(old, (int, np.integer)) else float(new)
        params[name] = new
        changes[name] = {"tuned": old, "final": new}
    if not changes:
        return {}
    return {"factor": round(factor, 4), "n_fit": int(n_fit), "n_fold_train": int(n_fold_train), "params": changes}


# ─────────────────────────────────────────
# Pomocnicze
# ─────────────────────────────────────────
def _vitals_complete_mask(df: pd.DataFrame) -> np.ndarray | None:
    """Maska wierszy z kompletem HR, SBP i RR; None, gdy splity nie mają kolumny (stare dane)."""
    if VITALS_COMPLETE_COL not in df.columns:
        return None
    return pd.to_numeric(df[VITALS_COMPLETE_COL], errors="coerce").to_numpy() == 1


def _predictions_frame(meta: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """row_id, site, esi, target, vitals_complete (NaN w starych splitach), p_esi1..5, pred_*."""
    out = meta[["row_id", "site", "esi", "target"]].reset_index(drop=True).copy()
    if VITALS_COMPLETE_COL in meta.columns:
        out[VITALS_COMPLETE_COL] = meta[VITALS_COMPLETE_COL].to_numpy()
    else:
        out[VITALS_COMPLETE_COL] = np.nan
    proba = np.asarray(proba, dtype=np.float32)
    for i, col in enumerate(PROBA_COLS):
        out[col] = proba[:, i]
    for rule in DECISION_RULES:
        out[f"pred_{rule}"] = decide(proba, rule)
    return out[PREDICTION_COLS]


def _evaluate(df: pd.DataFrame, y: pd.Series, proba: np.ndarray, split: str, print_report: bool = False) -> dict:
    """Metryki obu reguł, trafność względem przyjęcia i podgrupa `{split}_vc_*` (vitals_complete == 1)."""
    metrics: dict = {}
    for rule in DECISION_RULES:
        res = full_evaluation(y, decide(proba, rule), proba, prefix=f"{split}_{rule}",
                              print_report=print_report and rule == "argmax")
        metrics.update(res)
    if "admitted" in df.columns:
        metrics.update(_admission_validity(y, proba, df["admitted"], split))
    vc = _vitals_complete_mask(df)
    if vc is not None and vc.any():
        y_vc = np.asarray(y)[vc]
        p_vc = np.asarray(proba)[vc]
        metrics[f"{split}_vc_n"] = int(vc.sum())
        for rule in DECISION_RULES:
            metrics.update(full_evaluation(y_vc, decide(p_vc, rule), p_vc, prefix=f"{split}_vc_{rule}",
                                           print_report=False))
    return metrics


def _admission_validity(y: pd.Series, proba: np.ndarray, admitted: pd.Series, split: str) -> dict:
    """
    Trafność kliniczna niezależna od etykiety: czy przewidywana pilność przewiduje
    przyjęcie do szpitala? Porównanie z ESI nadanym przez pielęgniarkę.

    Wynik modelu = oczekiwany poziom ESI (Σ p·k); niższy = pilniej → większa szansa przyjęcia.
    """
    from sklearn.metrics import roc_auc_score

    adm = np.asarray(admitted).astype(int)
    if adm.min() == adm.max():
        return {}
    expected_level = np.asarray(proba) @ np.arange(proba.shape[1])
    return {
        f"{split}_admission_auc_model": float(roc_auc_score(adm, -expected_level)),
        f"{split}_admission_auc_nurse_esi": float(roc_auc_score(adm, -np.asarray(y))),
    }


def _site_strata(df: pd.DataFrame, index: pd.Index, y: pd.Series) -> np.ndarray | None:
    """Etykiety stratyfikacji foldów CV: ESI × oddział (jak podział danych)."""
    if "site" not in df.columns:
        return None
    site = df.loc[index, "site"].astype(str).to_numpy()
    return np.char.add(np.char.add(np.asarray(y).astype(int).astype(str), "_"), site.astype(str))


def _new_run_dir(model_name: str, feature_set: str, tag: str | None) -> tuple[str, str, Path]:
    """
    Nowy katalog runa `<YYYYmmdd_HHMMSS>_<model>_<cechy>[_<tag>]` — nigdy istniejący
    (dwa runy w tej samej sekundzie: znacznik czasu +1 s; mkdir atomowy między procesami).
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now()
    for offset in range(3600):
        run_id = (stamp + timedelta(seconds=offset)).strftime("%Y%m%d_%H%M%S")
        run_name = f"{run_id}_{model_name}_{feature_set}" + (f"_{tag}" if tag else "")
        run_dir = RUNS_DIR / run_name
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_id, run_name, run_dir
    raise RuntimeError(f"Nie udało się utworzyć katalogu runa w {RUNS_DIR}")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _write_json(path: Path, payload: Any) -> None:
    """JSON w ASCII (bezpieczny dla każdego czytelnika, także bez encoding= na Windows), UTF-8."""
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _replace_with_retry(tmp: Path, target: Path, retries: int = 5) -> None:
    """os.replace z ponowieniami (Windows: plik docelowy chwilowo otwarty, np. w Excelu)."""
    for attempt in range(retries):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if attempt == retries - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.2 * (attempt + 1))


# ─────────────────────────────────────────
# Leaderboard (przebudowa z leaderboard_row.json — bez wyścigów przy równoległych torach)
# ─────────────────────────────────────────
def _leaderboard_row(info: dict, metrics: dict) -> dict:
    tuning = info.get("tuning") or {}
    device = info.get("device") or {}
    row: dict[str, Any] = {
        "run_name": info["run_name"],
        "model": info["model"],
        "feature_set": info["feature_set"],
        "tag": info.get("tag"),
        "profile": info.get("profile"),
        "weights": info.get("weights_strategy"),
        "tuned": info.get("tuned"),
        "n_fit": info.get("n_fit"),
        "n_val": info.get("n_val"),
        "train_sample": info.get("train_sample"),
        "eval_sample": info.get("eval_sample"),
        "duration_s": round(float(info.get("training_duration_s") or 0.0), 1),
        "actual_device": device.get("actual"),
        "device_fallback": bool(device.get("fallback_reason")),
        "data_version": info.get("data_version"),
        "data_fingerprint": info.get("data_fingerprint"),
        "study_name": tuning.get("study_name"),
        "best_trial": tuning.get("best_trial"),
        "cv_metric": tuning.get("metric"),
        "cv_score": tuning.get("best_value"),
        "cv_std": tuning.get("best_value_std"),
    }
    for rule in DECISION_RULES:
        for m in LEADERBOARD_METRICS:
            key = f"val_{rule}_{m}"
            if key in metrics:
                row[key] = metrics[key]
    for key in ("val_admission_auc_model", "val_admission_auc_nurse_esi", "val_vc_n"):
        if key in metrics:
            row[key] = metrics[key]
    for rule in DECISION_RULES:
        for m in VC_LEADERBOARD_METRICS:
            key = f"val_vc_{rule}_{m}"
            if key in metrics:
                row[key] = metrics[key]
    return row


def rebuild_leaderboard(runs_dir: Path | str | None = None) -> Path:
    """
    Buduje runs/leaderboard.csv od nowa z `leaderboard_row.json` UKOŃCZONYCH runów (z DONE.json).

    Każdy run pisze tylko własny plik, więc równoległe procesy nie nadpisują sobie wierszy;
    przebudowa jest idempotentna (można ją wywołać w dowolnym momencie). Tylko metryki VAL.
    """
    if runs_dir is None:
        runs_dir, csv_path = Path(RUNS_DIR), Path(LEADERBOARD_CSV)
    else:
        runs_dir = Path(runs_dir)
        csv_path = runs_dir / "leaderboard.csv"
    runs_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row_path in sorted(runs_dir.glob(f"*/{LEADERBOARD_ROW_JSON}")):
        done = row_path.parent / DONE_JSON
        if not done.exists():
            continue
        try:
            rows.append((done.stat().st_mtime, json.loads(row_path.read_text(encoding="utf-8"))))
        except (OSError, ValueError) as exc:
            log.warning(f"Pomijam uszkodzony {row_path}: {exc}")
    # Kolejność ukończenia (czas zapisu DONE.json), potem nazwa runa
    rows.sort(key=lambda item: (item[0], str(item[1].get("run_name"))))
    board = pd.DataFrame([row for _, row in rows])
    tmp = csv_path.with_name(f"{csv_path.name}.{os.getpid()}.tmp")
    board.to_csv(tmp, index=False, encoding="utf-8")
    _replace_with_retry(tmp, csv_path)
    return csv_path


# ─────────────────────────────────────────
# Trening jednego modelu
# ─────────────────────────────────────────
def run_training(
    model_name: str,
    feature_set: str = "triage",
    tune: bool = False,
    n_trials: int | None = None,
    tune_metric: str = "logloss",
    tune_rows: int | None = None,
    cv_folds: int | None = None,
    sample_weight_strategy: str = DEFAULT_SAMPLE_WEIGHT_STRATEGY,
    use_mlflow: bool = False,
    save_model: bool = True,
    custom_params: dict | None = None,
    do_filter: bool = False,
    train_sample: int | None = None,
    eval_sample: int | None = None,
    tag: str | None = None,
    profile: str = DEFAULT_TUNING_PROFILE,
    tune_timeout: float | None = None,
) -> dict:
    """
    Pełen pipeline treningu jednego modelu.

    Parameters
    ----------
    profile : 'smoke' | 'quick' | 'standard' | 'thorough' — budżet tuningu i próbki danych.
    n_trials, tune_rows (0 = cały X_fit), cv_folds, tune_timeout (s), train_sample, eval_sample :
        None → wartość z profilu; podana wartość ma pierwszeństwo.
    tag : dopisek do nazwy runa; None → nazwa profilu, '' → bez dopisku.
    custom_params : parametry modelu (w tuningu stałe dla każdej próby; dostrojone je nadpisują).
    do_filter : filtr cech o zerowej ważności — tylko z tune=True (inaczej pomijany z ostrzeżeniem).

    Returns
    -------
    dict: run_name, run_dir, model_name, feature_set, metrics, model_path.
    """
    model_name = resolve_model_name(model_name)
    feature_set = resolve_feature_set(feature_set)
    budget = resolve_budget(model_name, profile)
    overrides = {"n_trials": n_trials, "tune_rows": tune_rows, "n_folds": cv_folds, "timeout_s": tune_timeout,
                 "train_sample": train_sample, "eval_sample": eval_sample}
    for key, value in overrides.items():
        if value is not None:
            budget[key] = value
    train_sample, eval_sample = budget["train_sample"], budget["eval_sample"]
    if tag is None:
        tag = profile

    log.info("═" * 64)
    log.info(f" {model_name} · {feature_set} · profil '{profile}'" + (" · tuning" if tune else ""))
    log.info("═" * 64)

    if do_filter and not tune:
        log.warning("--filter działa tylko razem z --tune (filtr jest częścią konfiguracji tuningu) — pomijam")
        do_filter = False

    # 1. Dane (przed utworzeniem katalogu runa — brak danych nie zostawia pustych katalogów)
    ensure_prepared()
    splits = load_splits(feature_set=feature_set)
    df_train = sample_frame(splits["train"], train_sample)
    df_val = sample_frame(splits["val"], eval_sample)
    df_test = sample_frame(splits["test"], eval_sample)

    X_train, y_train = get_xy(df_train, feature_set)
    X_val, y_val = get_xy(df_val, feature_set)
    X_test, y_test = get_xy(df_test, feature_set)

    run_id, run_name, run_dir = _new_run_dir(model_name, feature_set, tag)
    log.info(f"Run: {run_name}")

    # 2. Zbiór dopasowania: early stopping tylko dla modeli, które go używają
    uses_es = MODEL_REGISTRY[model_name].supports_early_stopping
    if uses_es:
        X_fit, y_fit, X_es, y_es = early_stopping_split(X_train, y_train)
    else:
        X_fit, y_fit, X_es, y_es = X_train, y_train, None, None
    n_es = 0 if X_es is None else len(X_es)
    log.info(f"Fit {len(X_fit):,}" + (f" (+{n_es:,} early stopping)" if uses_es else " (cały TRAIN, bez ES)")
             + f" | val {len(X_val):,} | test {len(X_test):,} | cechy: {X_train.shape[1]} ({feature_set})")

    # 3. (Opcjonalnie) filtr cech + tuning — wyłącznie na zbiorze dopasowania
    feature_names = list(X_train.columns)
    n_features_before = len(feature_names)
    tuning_info = None
    scaling: dict = {}
    best_params = dict(custom_params or {})
    optuna_cw = None
    if tune:
        from src.models.tuning import filter_features_by_importance_1fold, tune_model

        if do_filter:
            kept = filter_features_by_importance_1fold(X_fit, y_fit, feature_names, model_name=model_name)
            if len(kept) < len(feature_names):
                log.info(f"Filtr cech: {len(feature_names)} → {len(kept)}")
                feature_names = kept
        OPTUNA_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        tuning_info = tune_model(
            model_name=model_name,
            X_train=X_fit[feature_names],
            y_train=y_fit,
            n_trials=int(budget["n_trials"]),
            metric=tune_metric,
            n_folds=int(budget["n_folds"]),
            max_rows=int(budget["tune_rows"]) or None,
            sample_weight_strategy=sample_weight_strategy,
            timeout=budget["timeout_s"],
            storage_path=OPTUNA_STORAGE_DIR / f"optuna_{model_name}.db",
            feature_set=feature_set,
            strata=_site_strata(df_train, X_fit.index, y_fit),
            do_filter=do_filter,
            profile=profile,
            n_final_fit=len(X_fit),
            fixed_params=custom_params,
        )
        tuned = {k: v for k, v in tuning_info["best_params"].items() if not k.startswith("cw_")}
        scaling = _scale_count_params(model_name, tuned, len(X_fit), tuning_info.get("n_fold_train"))
        if scaling:
            log.info(f"Skalowanie parametrów liczności ×{scaling['factor']} "
                     f"(fold {scaling['n_fold_train']:,} → fit {scaling['n_fit']:,}): {scaling['params']}")
        best_params.update(tuned)
        cw = {k: v for k, v in tuning_info["best_params"].items() if k.startswith("cw_")}
        if cw:
            optuna_cw = {i: float(cw.get(f"cw_{CLASS_NAMES[i].lower()}", 1.0)) for i in range(len(CLASS_NAMES))}
            log.info(f"Wagi klas z Optuny: {optuna_cw}")

    # Wagi klas faktycznie użyte w treningu (do raportu)
    if optuna_cw is not None:
        weights_strategy, class_weights = "optuna_qwk", optuna_cw
    elif sample_weight_strategy != "none":
        weights_strategy = sample_weight_strategy
        class_weights = compute_class_weights_dict(y_fit, strategy=sample_weight_strategy)
    else:
        weights_strategy, class_weights = "none", None

    # 4. Trening finalny
    model = get_model(model_name, params=best_params or None)
    model.optuna_class_weights = optuna_cw
    model.fit(
        X_fit[feature_names], y_fit,
        X_val=X_es[feature_names] if X_es is not None else None, y_val=y_es,
        sample_weight_strategy=sample_weight_strategy,
        run_id=run_id,
        use_mlflow=use_mlflow,
        feature_set=feature_set,
    )

    # 5. Predykcje + metryki (val; test tylko do raportu końcowego)
    proba_val = model.predict_proba(X_val[feature_names])
    proba_test = model.predict_proba(X_test[feature_names])
    log.info("Ewaluacja na VAL…")
    metrics = _evaluate(df_val, y_val, proba_val, "val", print_report=True)
    metrics.update(_evaluate(df_test, y_test, proba_test, "test"))

    # 6. Zapis artefaktów
    _predictions_frame(df_val, proba_val).to_parquet(run_dir / "predictions_val.parquet", index=False)
    _predictions_frame(df_test, proba_test).to_parquet(run_dir / "predictions_test.parquet", index=False)

    model_path = None
    if save_model:
        model_path = model.save(run_dir / "model.joblib")

    importances = model.feature_importances()
    if importances is not None:
        importances.to_csv(run_dir / "feature_importance.csv", index=False, encoding="utf-8")

    try:
        hardware = hardware_summary()
    except Exception as exc:  # noqa: BLE001 — opis sprzętu nie może przerwać zapisu wyników
        hardware = {"error": f"{type(exc).__name__}: {exc}"}

    fingerprint = data_fingerprint()
    info = {
        "run_name": run_name,
        "model": model_name,
        "feature_set": feature_set,
        "profile": profile,
        "tag": tag or None,
        "n_features": len(feature_names),
        "n_features_before_filter": n_features_before,
        "feature_filter": bool(do_filter),
        "sample_weight_strategy": sample_weight_strategy,
        "weights_strategy": weights_strategy,
        "class_weights": class_weights,
        "tuned": tune,
        "tune_metric": tune_metric if tune else None,
        "budget": budget,
        "custom_params": custom_params or None,
        "params": model.params,
        "tuning_adjustments": scaling or None,
        "device": {
            "requested": model.requested_device,
            "actual": model.actual_device,
            "fallback_reason": model.device_fallback,
        },
        "best_iteration": model.best_iteration,
        "iteration_cap": model.iteration_cap,
        "data_version": DATA_VERSION,
        "data_fingerprint": fingerprint,
        "split_strategy": manifest().get("split_strategy"),
        "early_stopping": uses_es,
        "n_fit": int(len(X_fit)),
        "n_early_stopping": int(n_es),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "train_sample": train_sample,
        "eval_sample": eval_sample,
        "training_duration_s": model.training_duration_s,
        "tuning": tuning_info,
        "hardware": hardware,
        "class_names": CLASS_NAMES,
        "proba_columns": PROBA_COLS,
    }
    _write_json(run_dir / "run_info.json", info)
    _write_json(run_dir / "metrics.json", metrics)
    _write_report(run_dir / "report.md", info, metrics, importances)
    _write_json(run_dir / LEADERBOARD_ROW_JSON, _leaderboard_row(info, metrics))

    # DONE.json = znacznik ukończenia (orkiestrator) — zapisywany jako OSTATNI artefakt runa
    write_json_atomic(run_dir / DONE_JSON, json.loads(json.dumps({
        "run_name": run_name,
        "model": model_name,
        "feature_set": feature_set,
        "tag": tag or None,
        "profile": profile,
        "tuned": tune,
        "data_version": DATA_VERSION,
        "data_fingerprint": fingerprint,
        "study_name": (tuning_info or {}).get("study_name"),
        "best_trial": (tuning_info or {}).get("best_trial"),
        "actual_device": model.actual_device,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }, default=_json_default)))

    try:
        board = rebuild_leaderboard()
        log.info(f"Leaderboard (VAL): {board}")
    except Exception as exc:  # noqa: BLE001 — wspólny plik nie może unieważnić ukończonego runa
        log.warning(f"Nie udało się przebudować leaderboardu: {exc} (run jest kompletny)")

    log.info(f"QWK val (argmax/koszt): {metrics['val_argmax_quadratic_weighted_kappa']:.4f} / "
             f"{metrics['val_cost_quadratic_weighted_kappa']:.4f} | "
             f"undertriage val (argmax/koszt): {metrics['val_argmax_undertriage_rate']:.4f} / "
             f"{metrics['val_cost_undertriage_rate']:.4f}")
    log.info(f"Artefakty: {run_dir}")

    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "model_name": model_name,
        "feature_set": feature_set,
        "metrics": metrics,
        "model_path": str(model_path) if model_path else None,
    }


# ─────────────────────────────────────────
# Raport markdown
# ─────────────────────────────────────────
def _fmt(v) -> str:
    if isinstance(v, bool) or v is None:
        return "—" if v is None else str(v)
    if isinstance(v, float):
        return "—" if math.isnan(v) else f"{v:.4f}"
    return str(v)


def _metric_table(metrics: dict, columns: list[tuple[str, str]], names: list[str]) -> list[str]:
    """Tabela: wiersz = metryka, kolumna = (nagłówek, prefiks klucza)."""
    lines = ["| metryka | " + " | ".join(h for h, _ in columns) + " |",
             "|---|" + "---:|" * len(columns)]
    for m in names:
        cells = [_fmt(metrics.get(f"{prefix}_{m}")) for _, prefix in columns]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return lines


def _confusion_lines(metrics: dict, key: str) -> list[str]:
    lines = ["| | " + " | ".join(CLASS_NAMES) + " |", "|---|" + "---:|" * len(CLASS_NAMES)]
    for name, row in zip(CLASS_NAMES, metrics.get(key) or [], strict=False):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
    return lines


def _tuning_lines(info: dict, metrics: dict) -> list[str]:
    t = info.get("tuning")
    if not t:
        return ["## Tuning", "", "Bez tuningu — parametry domyślne z `config.py`"
                + (" + podane ręcznie: " + ", ".join(f"`{k}`={v}" for k, v in info["custom_params"].items())
                   if info.get("custom_params") else "") + ".", ""]
    metric = t["metric"]
    val_key = "val_argmax_log_loss" if metric == "logloss" else "val_argmax_quadratic_weighted_kappa"
    val_value = metrics.get(val_key)
    std = t.get("best_value_std")
    lines = [
        "## Tuning",
        "",
        f"- badanie Optuny: `{t['study_name']}`" + (f" (`{t['storage']}`)" if t.get("storage") else ""),
        f"- próby: ukończone {t['n_complete']}, przerwane (pruning) {t['n_pruned']}, błędy {t['n_fail']} "
        f"— cel {t['n_trials_target']} ukończonych; w tej sesji {t['n_trials_session']}",
        f"- czas tuningu łącznie: {float(t.get('elapsed_s') or 0) / 60:.1f} min"
        + (f" z limitu {t['timeout_s'] / 60:.0f} min" if t.get("timeout_s") else "")
        + f"; koniec: `{t['stopped_reason']}`",
        f"- CV: {t['n_folds']} foldy na {t['n_rows']:,} wierszach (fold-train ≈ {t['n_fold_train']:,}), "
        f"stratyfikacja `{t['config'].get('stratified_by')}`, przestrzeń v{t['search_space_version']}",
        f"- **CV {metric}: {t['best_value']:.5f}" + (f" ± {std:.5f}" if std is not None else "")
        + f"** (próba #{t['best_trial']}) vs **val {metric}: {_fmt(val_value)}**"
        + ("" if val_value is None or metric != "logloss" else
           f" (różnica {float(val_value) - t['best_value']:+.5f}; duża dodatnia = przeuczenie tuningu)"),
    ]
    if metric == "logloss" and info.get("sample_weight_strategy") not in (None, "none"):
        lines.append("- uwaga: log loss CV liczony z wagami klas (jak trening), val — bez wag")
    if metric == "qwk":
        lines.append("- uwaga: cel QWK dobiera wagi klas — prawdopodobieństwa nie są skalibrowane, "
                     "reguła kosztowa liczy asymetrię podwójnie (używaj reguły argmax)")
    attrs = t.get("best_trial_attrs") or {}
    oof = [f"{k} {attrs[k]:.4f}" for k in ("qwk_argmax", "qwk_cost", "undertriage_cost", "expected_cost_cost")
           if isinstance(attrs.get(k), (int, float))]
    if oof:
        lines.append("- najlepsza próba, predykcje out-of-fold: " + ", ".join(oof))
    if t.get("best_iter"):
        final_iter, cap = info.get("best_iteration"), info.get("iteration_cap")
        line = f"- najlepsza iteracja: CV {t['best_iter']} (limit {t.get('iteration_cap')}) → finał {final_iter}"
        if final_iter and cap:
            line += f" z limitu {cap}"
            if final_iter >= ITERATION_CAP_WARN_FRACTION * cap:
                line += " — **UWAGA: limit iteracji osiągnięty, model może być niedotrenowany**"
        if t.get("hit_cap"):
            line += " — **UWAGA: w CV najlepsza iteracja przy limicie**"
        lines.append(line)
    scaling = info.get("tuning_adjustments")
    if scaling:
        changes = ", ".join(f"`{k}` {v['tuned']:.4g} → {v['final']:.4g}" for k, v in scaling["params"].items())
        lines.append(f"- skalowanie parametrów liczności ×{scaling['factor']} (√(fit/fold)): {changes}")
    if t.get("best_json"):
        lines.append(f"- wynik: `{t['best_json']}`")
    lines += ["", "| parametr | wartość |", "|---|---:|"]
    for key, value in sorted(t["best_params"].items()):
        lines.append(f"| `{key}` | {value:.6g} |" if isinstance(value, float) else f"| `{key}` | {value} |")
    if t.get("fixed_params"):
        fixed = ", ".join(f"`{k}`={v}" for k, v in sorted(t["fixed_params"].items()))
        lines.append(f"\nStałe w tuningu: {fixed}. Finalny model: parametry podane ręcznie + dostrojone, "
                     f"reszta z `config.py` (np. RF 500 drzew, EBM 14 worków).")
    lines.append("")
    return lines


def _write_report(path: Path, info: dict, metrics: dict, importances: pd.DataFrame | None) -> None:
    device = info.get("device") or {}
    hw = info.get("hardware") or {}
    smi = hw.get("nvidia_smi") or {}
    es_txt = (f"fit {info['n_fit']:,} + early stopping {info['n_early_stopping']:,}" if info.get("early_stopping")
              else f"fit {info['n_fit']:,} (cały TRAIN, model bez early stopping)")
    cw = info.get("class_weights")
    lines = [
        f"# {info['run_name']}",
        "",
        f"- model: `{info['model']}`, zestaw cech: `{info['feature_set']}` ({info['n_features']} cech"
        + (f", filtr z {info['n_features_before_filter']}" if info.get("feature_filter") else "") + ")",
        f"- profil: `{info.get('profile')}`, tag: `{info.get('tag')}`",
        f"- wagi klas: `{info['weights_strategy']}`"
        + (" (" + ", ".join(f"ESI {int(k) + 1}: {float(v):.3g}" for k, v in cw.items()) + ")" if cw else ""),
        f"- dane: `{info['data_version']}` (odcisk `{info.get('data_fingerprint')}`), podział "
        f"`{info['split_strategy']}`; {es_txt}, val {info['n_val']:,}, test {info['n_test']:,}"
        + (f"; próbki: train {info['train_sample']}, val/test {info['eval_sample']}"
           if info.get("train_sample") or info.get("eval_sample") else ""),
        f"- urządzenie: żądane `{device.get('requested')}`, faktyczne `{device.get('actual')}`"
        + (f" — **GPU → CPU: {device['fallback_reason']}**" if device.get("fallback_reason") else ""),
        f"- sprzęt: {hw.get('cpu_name', '?')}, {hw.get('physical_cores', '?')} rdzeni fizycznych, "
        f"n_jobs {hw.get('n_jobs', '?')}, RAM {hw.get('ram_gb', '?')} GB"
        + (f", GPU {smi.get('name')} (sterownik {smi.get('driver_version')})" if smi else ", brak GPU NVIDIA"),
        f"- czas treningu finalnego: {info['training_duration_s'] or 0:.1f} s"
        + (f"; najlepsza iteracja {info['best_iteration']} z limitu {info['iteration_cap']}"
           if info.get("best_iteration") is not None else ""),
        "",
    ]
    lines += _tuning_lines(info, metrics)

    lines += ["## Metryki — VAL (wybór modeli, wag i progów)", ""]
    columns = [("argmax", "val_argmax"), ("koszt", "val_cost")]
    if "val_vc_n" in metrics:
        columns += [("argmax · vitals complete", "val_vc_argmax"), ("koszt · vitals complete", "val_vc_cost")]
    lines += _metric_table(metrics, columns, LEADERBOARD_METRICS + ["accuracy", "within_one_accuracy"])
    if "val_vc_n" in metrics:
        lines += ["", f"Podgrupa *vitals complete* (HR, SBP i RR zmierzone przy triażu): "
                      f"{metrics['val_vc_n']:,} z {info['n_val']:,} wizyt val."]
    if "val_admission_auc_model" in metrics:
        lines += [
            "",
            "Trafność kliniczna (AUC przewidywania przyjęcia do szpitala z pilności, val): "
            f"model (oczekiwany poziom ESI) {_fmt(metrics.get('val_admission_auc_model'))}, "
            f"ESI pielęgniarki {_fmt(metrics.get('val_admission_auc_nurse_esi'))}.",
        ]
    lines += [
        "",
        "Reguła **koszt**: klasa minimalizująca oczekiwany koszt (niedotriażowanie 3× droższe).",
        "",
        "### Macierz pomyłek (val, argmax; wiersze = prawda ESI 1–5)",
        "",
    ]
    lines += _confusion_lines(metrics, "val_argmax_confusion_matrix")
    if importances is not None:
        lines += ["", "## Top 25 cech", "", "| cecha | ważność |", "|---|---:|"]
        for _, r in importances.head(25).iterrows():
            lines.append(f"| `{r['feature']}` | {r['importance']:.4f} |")

    lines += [
        "",
        "## Test — tylko do raportu końcowego",
        "",
        "Nie używaj tych liczb do wyboru modelu, wag łączenia ani progów — do tego służy val.",
        "",
    ]
    test_columns = [("argmax", "test_argmax"), ("koszt", "test_cost")]
    if "test_vc_n" in metrics:
        test_columns += [("argmax · vitals complete", "test_vc_argmax"), ("koszt · vitals complete", "test_vc_cost")]
    lines += _metric_table(metrics, test_columns, LEADERBOARD_METRICS + ["accuracy", "within_one_accuracy"])
    if "test_admission_auc_model" in metrics:
        lines += ["", f"AUC przyjęcia (test): model {_fmt(metrics.get('test_admission_auc_model'))}, "
                      f"ESI pielęgniarki {_fmt(metrics.get('test_admission_auc_nurse_esi'))}."]
    lines += ["", "### Macierz pomyłek (test, argmax)", ""]
    lines += _confusion_lines(metrics, "test_argmax_confusion_matrix")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"oczekiwano liczby > 0, podano {value}")
    return number


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trening modeli klasyfikacji ESI (SOR-AI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m", default="all",
        help=f"'all', jeden model albo lista po przecinku. Dostępne: {', '.join(MODEL_REGISTRY)}",
    )
    parser.add_argument("--feature-set", "-f", default="triage", help="triage | full")
    parser.add_argument(
        "--profile", choices=list(TUNING_PROFILES), default=DEFAULT_TUNING_PROFILE,
        help="Budżet tuningu i próbki danych (config.TUNING_PROFILES); jawne flagi poniżej mają pierwszeństwo",
    )
    parser.add_argument("--tune", "-t", action="store_true", help="Tuning Optuną przed treningiem (wznawialny)")
    parser.add_argument("--n-trials", "-n", type=_positive_int, default=None,
                        help="Łączna liczba ukończonych prób badania (domyślnie z profilu)")
    parser.add_argument("--tune-metric", choices=["logloss", "qwk"], default="logloss",
                        help="Cel tuningu: logloss (jakość prawdopodobieństw) albo qwk (+ wagi klas)")
    parser.add_argument("--tune-rows", type=int, default=None,
                        help="Próbka do tuningu (0 = cały zbiór dopasowania; domyślnie z profilu)")
    parser.add_argument("--cv-folds", type=int, default=None, help="Liczba foldów CV (domyślnie z profilu)")
    parser.add_argument("--tune-timeout", type=float, default=None,
                        help="Łączny limit czasu tuningu modelu w sekundach (domyślnie z profilu)")
    parser.add_argument("--n-jobs", type=_positive_int, default=None,
                        help="Wątki CPU (SOR_AI_N_JOBS; skuteczne przez scripts/03_train.py, przed importem)")
    parser.add_argument(
        "--filter", action="store_true",
        help="Przed tuningiem odrzuć cechy z zerową ważnością (tylko z --tune; bez niego pomijane)",
    )
    parser.add_argument(
        "--weights", "-w", choices=["none", "balanced", "sqrt_balanced", "custom"],
        default=DEFAULT_SAMPLE_WEIGHT_STRATEGY, help="Wagi klas w treningu",
    )
    parser.add_argument("--sample", type=_positive_int, default=None,
                        help="Stratyfikowana próbka TRAIN (domyślnie z profilu; standard = całość)")
    parser.add_argument("--eval-sample", type=_positive_int, default=None,
                        help="Próbka VAL i TEST (domyślnie z profilu; nie do raportowania!)")
    parser.add_argument("--tag", default=None, help="Dopisek do nazwy runa (domyślnie nazwa profilu)")
    parser.add_argument("--mlflow", action="store_true", help="Loguj równolegle do MLflow")
    parser.add_argument("--no-save", action="store_true", help="Nie zapisuj pliku modelu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    CLI. Każdy model trenowany osobno — błąd jednego nie przerywa pozostałych.

    Returns
    -------
    0 = wszystko OK, 1 = co najmniej jeden model się nie udał, 130 = przerwano (Ctrl+C).
    """
    args = _parse_args(argv)
    if args.n_jobs is not None:
        from src.utils import config as _config

        if int(_config.N_JOBS) != args.n_jobs:
            log.warning(f"--n-jobs {args.n_jobs} podane za późno (config.N_JOBS={_config.N_JOBS} ustalone przy "
                        f"imporcie) — uruchamiaj przez scripts/03_train.py albo ustaw SOR_AI_N_JOBS")
    if args.model.lower() == "all":
        models = list(MODEL_REGISTRY)
    else:
        models = [resolve_model_name(m.strip()) for m in args.model.split(",") if m.strip()]
    if args.filter and not args.tune:
        log.warning("--filter bez --tune nic nie robi (filtr cech działa tylko przed tuningiem)")

    results: list[dict] = []
    failures: list[tuple[str, str]] = []
    try:
        for name in models:
            try:
                results.append(run_training(
                    model_name=name,
                    feature_set=args.feature_set,
                    tune=args.tune,
                    n_trials=args.n_trials,
                    tune_metric=args.tune_metric,
                    tune_rows=args.tune_rows,
                    cv_folds=args.cv_folds,
                    sample_weight_strategy=args.weights,
                    use_mlflow=args.mlflow,
                    save_model=not args.no_save,
                    do_filter=args.filter,
                    train_sample=args.sample,
                    eval_sample=args.eval_sample,
                    tag=args.tag,
                    profile=args.profile,
                    tune_timeout=args.tune_timeout,
                ))
            except Exception as exc:  # noqa: BLE001 — jeden model nie może zatrzymać nocnego przebiegu
                log.exception(f"{name}: TRENING NIEUDANY — {type(exc).__name__}: {exc}")
                failures.append((name, f"{type(exc).__name__}: {exc}"))
    except KeyboardInterrupt:
        log.warning("Przerwano (Ctrl+C). Tuning wznowi się od miejsca przerwania — uruchom to samo polecenie.")
        return 130

    log.info("╔════════════════════════════════════════════════════════════╗")
    log.info("║                     TRENING ZAKOŃCZONY                     ║")
    log.info("╚════════════════════════════════════════════════════════════╝")
    for r in results:
        m = r["metrics"]
        log.info(
            f"{r['model_name']:14s} QWK val {m['val_argmax_quadratic_weighted_kappa']:.4f} "
            f"| log loss {m.get('val_argmax_log_loss', float('nan')):.4f} "
            f"| undertriage (koszt) {m['val_cost_undertriage_rate']:.4f} → {r['run_dir']}"
        )
    for name, error in failures:
        log.error(f"{name:14s} NIEUDANY: {error[:300]}")
    log.info(f"Zestawienie runów (VAL): {LEADERBOARD_CSV}")
    log.info("Porównanie i łączenie: python scripts/07_compare_runs.py")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
