"""
Cross-validation dla modeli triażowych.

Stratified k-fold (k=10) — zachowuje rozkład klas w każdym foldzie.
Raportuje średnią ± odchylenie standardowe wszystkich metryk.

UWAGA: Dla preferowanego workflow chronologicznego — stosuj TimeSeriesSplit
zamiast StratifiedKFold. K-fold jest przydatny gdy chcemy oszacować
"theoretical upper bound" performance bez efektu temporalnego.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from src.evaluation.metrics import full_evaluation
from src.utils.config import (
    CV_N_SPLITS,
    EVAL_LOGS_DIR,
    RANDOM_SEED,
    TABLES_DIR,
)
from src.utils.logger import get_logger, get_training_logger

log = get_logger(__name__)


def cross_validate_model(
    model_factory,
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    n_splits: int = CV_N_SPLITS,
    cv_strategy: str = "stratified",
    sample_weight_strategy: str = "custom",
    save_path: Path | str | None = None,
    fit_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Cross-validation dla modelu triażowego.

    Parameters
    ----------
    model_factory : callable
        Funkcja zwracająca świeżą instancję BaseTriageModel
        (np. lambda: XGBoostTriageModel(params=best_params)).
    X : pd.DataFrame
    y : np.ndarray | pd.Series
    n_splits : int
        Liczba foldów (domyślnie 10).
    cv_strategy : str
        'stratified' | 'temporal' (TimeSeriesSplit).
    sample_weight_strategy : str
    save_path : Path
        CSV z wynikami per fold (opcjonalne).
    fit_kwargs : dict
        Dodatkowe argumenty do model.fit().

    Returns
    -------
    pd.DataFrame z metrykami per fold (każdy wiersz = jeden fold)
    """
    fit_kwargs = fit_kwargs or {}

    # CV splitter
    if cv_strategy == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        split_iter = splitter.split(X, y)
    elif cv_strategy == "temporal":
        splitter = TimeSeriesSplit(n_splits=n_splits)
        split_iter = splitter.split(X)
    else:
        raise ValueError(f"Nieznana strategia CV: {cv_strategy}")

    # Logger CV
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_logger, cv_log_path = get_training_logger(
        model_name=f"cv_{cv_strategy}",
        run_id=run_id,
        log_dir=EVAL_LOGS_DIR,
    )

    cv_logger.info(f"=== Cross-validation ({cv_strategy}, k={n_splits}) ===")
    cv_logger.info(f"Total samples: {len(X):,}, features: {X.shape[1]}")

    fold_results: list[dict] = []
    y_array = np.asarray(y)

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        cv_logger.info(f"\n──── Fold {fold_idx + 1} / {n_splits} ────")
        cv_logger.info(f"  train: {len(train_idx):,}, val: {len(val_idx):,}")

        X_train = X.iloc[train_idx]
        y_train = y_array[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y_array[val_idx]

        # Świeża instancja modelu
        model = model_factory()
        # Wyłączamy zapisywanie do JSON eksperymentu per fold (zapisujemy zbiorczo)
        try:
            model.fit(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                sample_weight_strategy=sample_weight_strategy,
                run_id=f"{run_id}_fold{fold_idx}",
                **fit_kwargs,
            )
        except Exception as e:
            cv_logger.error(f"Fold {fold_idx} błąd treningu: {e}")
            continue

        # Ewaluacja
        try:
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
            metrics = full_evaluation(y_val, y_pred, y_proba, print_report=False)
            metrics["fold"] = fold_idx
            metrics["n_train"] = len(train_idx)
            metrics["n_val"] = len(val_idx)
            metrics["training_duration_s"] = model.training_duration_s
            fold_results.append(metrics)

            cv_logger.info(
                f"  Fold {fold_idx} → QWK={metrics['quadratic_weighted_kappa']:.4f}, "
                f"undertriage={metrics['undertriage_rate']:.4f}, "
                f"AUC_macro={metrics.get('auc_macro', float('nan')):.4f}"
            )
        except Exception as e:
            cv_logger.error(f"Fold {fold_idx} błąd ewaluacji: {e}")

    if not fold_results:
        raise RuntimeError("Wszystkie foldy się wywaliły — sprawdź log.")

    cv_df = pd.DataFrame(fold_results)

    # Spłaszcz confusion_matrix do stringa, by można było zapisać do CSV
    if "confusion_matrix" in cv_df.columns:
        cv_df["confusion_matrix"] = cv_df["confusion_matrix"].apply(
            lambda cm: json.dumps(cm) if isinstance(cm, list) else str(cm)
        )

    # Statystyki
    numeric_cols = cv_df.select_dtypes(include=[np.number]).columns.tolist()
    summary = cv_df[numeric_cols].agg(["mean", "std", "min", "max"]).T

    cv_logger.info("\n══════════════════════════════════════════════")
    cv_logger.info("  WYNIKI CV (mean ± std)")
    cv_logger.info("══════════════════════════════════════════════")
    for metric, row in summary.iterrows():
        if any(metric.startswith(p) for p in ["fold", "n_train", "n_val", "support_"]):
            continue
        cv_logger.info(
            f"  {metric:30s} {row['mean']:.4f} ± {row['std']:.4f}  [{row['min']:.4f}, {row['max']:.4f}]"
        )

    if save_path is not None:
        save_cv_results(cv_df, summary, save_path, run_id=run_id)

    return cv_df


def save_cv_results(
    cv_df: pd.DataFrame,
    summary: pd.DataFrame,
    save_path: Path | str,
    run_id: str | None = None,
) -> dict[str, Path]:
    """Zapisuje wyniki CV (per fold + summary)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = save_path.parent / save_path.stem

    paths = {}
    paths["per_fold"] = Path(f"{base}_{run_id}_per_fold.csv")
    paths["summary"] = Path(f"{base}_{run_id}_summary.csv")

    cv_df.to_csv(paths["per_fold"], index=False)
    summary.to_csv(paths["summary"], index=True)

    log.info(f"Zapisano CV results:")
    log.info(f"  Per fold: {paths['per_fold']}")
    log.info(f"  Summary:  {paths['summary']}")

    return paths
