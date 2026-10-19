"""
Cross-validation dla modeli triażowych (stratified k-fold na zbiorze TRAIN).

Protokół (bez przecieku danych):
    - CV działa WYŁĄCZNIE na splicie `train` — `val` i `test` zostają nietknięte.
    - StratifiedKFold po ESI × oddział (`make_strata`; gdy oddział nieznany — po ESI),
      tak jak podział train/val/test — każdy fold ma rozkład ESI 1–5 i oddziałów jak całość.
    - W każdym foldzie z części treningowej wydzielane jest 10% na early stopping
      (`src.data.dataset.early_stopping_split`); fold walidacyjny służy tylko do oceny.
    - Trening foldu: `model.fit_estimator` (bez logu treningu i JSON-a eksperymentu
      na każdy fold; błąd GPU → ponowienie na CPU), wagi jak w `fit`.
    - Metryki dla obu reguł decyzyjnych: argmax i minimalny koszt oczekiwany
      (klucze `argmax_<metryka>` i `cost_<metryka>`).
    - Raport: średnia ± odchylenie standardowe po foldach.

Brak wariantu czasowego (TimeSeriesSplit): Yale EMMLC nie ma roku ani znacznika
czasu wizyty — kolejność wierszy nie jest chronologiczna, więc podział "czasowy"
byłby w rzeczywistości arbitralny i mylący.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.dataset import early_stopping_split
from src.evaluation.metrics import decide, full_evaluation
from src.models.base import compute_sample_weights
from src.utils.config import (
    CV_N_SPLITS,
    DEFAULT_SAMPLE_WEIGHT_STRATEGY,
    EVAL_LOGS_DIR,
    RANDOM_SEED,
)
from src.utils.logger import close_training_logger, get_logger, get_training_logger

log = get_logger(__name__)

DECISION_RULES = ("argmax", "cost")

# Metryki wypisywane w podsumowaniu (CSV zawiera wszystkie)
KEY_METRICS = [
    "quadratic_weighted_kappa", "f1_macro", "auc_macro", "log_loss", "brier",
    "undertriage_rate", "critical_miss_rate", "overtriage_rate",
    "mistriage_under", "mistriage_over", "expected_cost",
]

# Kolumny techniczne — pomijane w podsumowaniu mean ± std
_META_COLUMNS = ("fold", "n_fit", "n_early_stopping", "n_val", "best_iteration")


# ─────────────────────────────────────────
# Stratyfikacja
# ─────────────────────────────────────────
def make_strata(
    y: np.ndarray | pd.Series,
    site: np.ndarray | pd.Series | None = None,
    n_splits: int = CV_N_SPLITS,
) -> np.ndarray:
    """
    Etykiety warstw do StratifiedKFold: ESI × oddział.

    Warstwy mające co najmniej `n_splits` wierszy zostają; rzadsze (np. ESI 1 na małym oddziale)
    wracają do warstwy samego ESI, żeby każdy fold dostał swoją część. Bez `site` — samo ESI.
    """
    y_str = pd.Series(np.asarray(y)).astype(int).astype(str).to_numpy()
    if site is None:
        return y_str
    site_str = pd.Series(np.asarray(site), dtype="object").fillna("?").astype(str).to_numpy()
    if len(site_str) != len(y_str):
        raise ValueError(f"make_strata: len(site)={len(site_str)} ≠ len(y)={len(y_str)}")
    combined = pd.Series(y_str) + "_" + pd.Series(site_str)
    counts = combined.map(combined.value_counts())
    return np.where(counts.to_numpy() >= n_splits, combined.to_numpy(), y_str)


# ─────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────
def cross_validate_model(
    model_factory: Callable,
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    n_splits: int = CV_N_SPLITS,
    sample_weight_strategy: str = DEFAULT_SAMPLE_WEIGHT_STRATEGY,
    feature_set: str = "triage",
    save_path: Path | str | None = None,
    model_name: str | None = None,
    strata: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """
    Stratified k-fold CV dla modelu triażowego.

    Parameters
    ----------
    model_factory : callable
        Funkcja zwracająca świeżą instancję BaseTriageModel
        (np. lambda: get_model("lightgbm", params=best_params)).
    X : pd.DataFrame
        Cechy z TRAIN (np. `get_xy(splits["train"], feature_set)[0]`).
    y : np.ndarray | pd.Series
        Klasy 0–4 (= ESI 1–5).
    n_splits : int
        Liczba foldów (domyślnie CV_N_SPLITS).
    sample_weight_strategy : str
        none | balanced | sqrt_balanced | custom.
    feature_set : str
        Nazwa zestawu cech (do logów treningu).
    save_path : Path
        Bazowa ścieżka CSV z wynikami (opcjonalne) — zob. `save_cv_results`.
    model_name : str, optional
        Nazwa do logów; domyślnie `model_factory().name`.
    strata : array, optional
        Etykiety warstw (np. `make_strata(y, site)`); domyślnie klasa ESI.

    Returns
    -------
    pd.DataFrame z metrykami per fold (każdy wiersz = jeden fold).
    """
    X = X.reset_index(drop=True)
    y_series = pd.Series(np.asarray(y), name="target")
    model_name = model_name or getattr(model_factory(), "name", "model")
    strata_arr = y_series.to_numpy() if strata is None else np.asarray(strata)
    if len(strata_arr) != len(X):
        raise ValueError(f"len(strata)={len(strata_arr)} ≠ len(X)={len(X)}")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    # Logger CV (logs/evaluation/cv_<model>/<run_id>.log)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_logger, cv_log_path = get_training_logger(
        model_name=f"cv_{model_name}",
        run_id=run_id,
        log_dir=EVAL_LOGS_DIR,
    )

    try:
        cv_logger.info(f"=== Cross-validation: {model_name} (stratified, k={n_splits}) ===")
        strata_desc = "ESI" if strata is None else f"{len(np.unique(strata_arr))} (ESI × oddział)"
        cv_logger.info(f"Próbki TRAIN: {len(X):,}, cechy: {X.shape[1]} ({feature_set}), "
                       f"wagi: {sample_weight_strategy}, warstwy: {strata_desc}")

        fold_results: list[dict] = []
        failed: list[int] = []
        t0 = time.time()

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, strata_arr)):
            cv_logger.info(f"──── Fold {fold_idx + 1} / {n_splits} ────")

            X_fit, y_fit, X_es, y_es = early_stopping_split(
                X.iloc[train_idx], y_series.iloc[train_idx]
            )
            X_val = X.iloc[val_idx]
            y_val = y_series.iloc[val_idx].to_numpy()
            cv_logger.info(
                f"  fit: {len(X_fit):,} (+{len(X_es):,} early stopping), ocena: {len(X_val):,}"
            )

            # Świeża instancja modelu; wagi jak w `fit` (wagi z Optuny > strategia)
            model = model_factory()
            try:
                weights = compute_sample_weights(
                    y_fit, strategy=sample_weight_strategy,
                    class_weights=getattr(model, "optuna_class_weights", None),
                )
                model.fit_estimator(X_fit, y_fit, X_es, y_es, sample_weight=weights)
                proba = model.predict_proba(X_val)
            except Exception as e:
                cv_logger.exception(f"Fold {fold_idx} — błąd treningu/predykcji: {e}")
                failed.append(fold_idx)
                continue

            metrics: dict = {
                "fold": fold_idx,
                "n_fit": len(X_fit),
                "n_early_stopping": len(X_es),
                "n_val": len(X_val),
                "training_duration_s": model.training_duration_s,
                "best_iteration": getattr(model, "best_iteration", None),
                "device": getattr(model, "actual_device", None),
            }
            for rule in DECISION_RULES:
                metrics.update(full_evaluation(
                    y_val, decide(proba, rule), proba, print_report=False, prefix=rule,
                ))
            fold_results.append(metrics)

            cv_logger.info(
                f"  Fold {fold_idx} → QWK={metrics['argmax_quadratic_weighted_kappa']:.4f} "
                f"(koszt: {metrics['cost_quadratic_weighted_kappa']:.4f}), "
                f"undertriage={metrics['argmax_undertriage_rate']:.4f} "
                f"(koszt: {metrics['cost_undertriage_rate']:.4f}), "
                f"AUC_macro={metrics.get('argmax_auc_macro', float('nan')):.4f}"
            )

        if not fold_results:
            raise RuntimeError(f"Wszystkie foldy zakończyły się błędem — sprawdź log: {cv_log_path}")
        if failed:
            cv_logger.warning(f"Foldy z błędem (pominięte w średniej): {failed}")

        cv_df = pd.DataFrame(fold_results)

        # Macierze pomyłek → JSON (zapisywalne do CSV)
        for col in [c for c in cv_df.columns if c.endswith("confusion_matrix")]:
            cv_df[col] = cv_df[col].apply(lambda cm: json.dumps(cm) if isinstance(cm, list) else str(cm))

        summary = summarize_cv(cv_df)

        cv_logger.info("══════════════════════════════════════════════")
        cv_logger.info(f"  WYNIKI CV (mean ± std, {len(cv_df)}/{n_splits} foldów)")
        cv_logger.info("══════════════════════════════════════════════")
        for rule in DECISION_RULES:
            cv_logger.info(f"  ─── reguła: {rule} ───")
            for metric in KEY_METRICS:
                key = f"{rule}_{metric}"
                if key not in summary.index:
                    continue
                row = summary.loc[key]
                cv_logger.info(
                    f"  {metric:28s} {row['mean']:.4f} ± {row['std']:.4f}  "
                    f"[{row['min']:.4f}, {row['max']:.4f}]"
                )
        if "training_duration_s" in summary.index:
            mean_s = summary.loc["training_duration_s", "mean"]
            cv_logger.info(f"  Czas treningu foldu: {mean_s:.1f} s (średnio), "
                           f"całe CV: {time.time() - t0:.1f} s")
        cv_logger.info(f"Log CV: {cv_log_path}")

        if save_path is not None:
            save_cv_results(cv_df, summary, save_path, run_id=run_id)

        return cv_df
    finally:
        close_training_logger(cv_logger)


def summarize_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    """mean / std / min / max każdej metryki numerycznej po foldach (wiersz = metryka)."""
    numeric_cols = [
        c for c in cv_df.select_dtypes(include=[np.number]).columns
        if c not in _META_COLUMNS and "_support_" not in c
    ]
    return cv_df[numeric_cols].agg(["mean", "std", "min", "max"]).T


def save_cv_results(
    cv_df: pd.DataFrame,
    summary: pd.DataFrame,
    save_path: Path | str,
    run_id: str | None = None,
) -> dict[str, Path]:
    """Zapisuje wyniki CV: `<save_path>_<run_id>_per_fold.csv` i `..._summary.csv`."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = save_path.parent / save_path.stem

    paths = {
        "per_fold": Path(f"{base}_{run_id}_per_fold.csv"),
        "summary": Path(f"{base}_{run_id}_summary.csv"),
    }
    cv_df.to_csv(paths["per_fold"], index=False, encoding="utf-8")
    summary.to_csv(paths["summary"], index=True, index_label="metric", encoding="utf-8")

    log.info("Zapisano wyniki CV:")
    log.info(f"  Per fold: {paths['per_fold']}")
    log.info(f"  Summary:  {paths['summary']}")

    return paths
