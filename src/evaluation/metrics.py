"""
Metryki ewaluacji dla 5-klasowej klasyfikacji triażu MTS.

Kluczowe metryki:
    1. QUADRATIC WEIGHTED KAPPA — główna metryka (ordinalna).
    2. UNDERTRIAGE RATE — KRYTYCZNA metryka bezpieczeństwa.
       (Red/Orange błędnie zaklasyfikowany jako Yellow/Green/Blue)
    3. OVERTRIAGE RATE — efektywność oddziału.
    4. Per-class AUC-ROC — szczególnie Red/Orange.
    5. 5×5 confusion matrix — wizualizacja błędów ordinalnych.

UWAGA: Accuracy jest BEZUŻYTECZNE — 98% accuracy może oznaczać 100% pominięć Red.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

from src.utils.config import CLASS_NAMES
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────
# Metryki bezpieczeństwa (kluczowe dla SOR-AI)
# ─────────────────────────────────────────
def undertriage_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    high_acuity_classes: tuple[int, ...] = (0, 1),
) -> float:
    """
    Procent pacjentów wymagających pilnej pomocy (Red/Orange) błędnie zaklasyfikowanych
    jako mniej pilni (Yellow/Green/Blue).

    To NAJWAŻNIEJSZA metryka bezpieczeństwa w medycynie ratunkowej:
        Lepiej overtriage zdrowego niż przeoczyć zawał.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    high_acuity_classes : tuple[int]
        Indeksy klas o wysokiej pilności (domyślnie Red=0, Orange=1).

    Returns
    -------
    float w [0, 1] (lub NaN jeśli brak high-acuity przypadków).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isin(y_true, high_acuity_classes)
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] > max(high_acuity_classes)).mean())


def overtriage_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    low_acuity_classes: tuple[int, ...] = (3, 4),
    threshold_class: int = 1,
) -> float:
    """
    Procent pacjentów Green/Blue błędnie zaklasyfikowanych jako Red/Orange.

    Parameters
    ----------
    low_acuity_classes : tuple[int]
        Klasy nieurgentne (Green=3, Blue=4).
    threshold_class : int
        Klasy ≤ threshold_class to "high acuity overtriage" (domyślnie ≤ Orange).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isin(y_true, low_acuity_classes)
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] <= threshold_class).mean())


def critical_miss_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Specjalny przypadek undertriage: Red (0) błędnie zaklasyfikowany jako Green/Blue (3, 4).
    To "missed critical" — najgorszy możliwy błąd.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = (y_true == 0)
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] >= 3).mean())


# ─────────────────────────────────────────
# Quadratic Weighted Kappa (główna metryka)
# ─────────────────────────────────────────
def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Cohen's Kappa z kwadratowym ważeniem — penalizuje większe błędy ordinalne mocniej.

    Wartości:
        QWK = 1.0  → idealna zgodność
        QWK = 0.0  → losowa zgodność
        QWK < 0    → gorzej niż losowo

    Interpretacja (Landis & Koch):
        0.81–1.00  prawie idealna
        0.61–0.80  silna
        0.41–0.60  umiarkowana
        0.21–0.40  słaba
    """
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def linear_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Liniowo ważony Kappa — alternatywa dla QWK."""
    return float(cohen_kappa_score(y_true, y_pred, weights="linear"))


# ─────────────────────────────────────────
# Pełna ewaluacja
# ─────────────────────────────────────────
def full_evaluation(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
    print_report: bool = True,
    prefix: str = "",
) -> dict[str, float | dict | list]:
    """
    Pełna ewaluacja modelu triażowego.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    y_proba : np.ndarray, optional
        Prawdopodobieństwa per klasa, shape (n, 5).
    class_names : list[str], optional
    print_report : bool
        Czy wyświetlić classification report.
    prefix : str
        Prefiks dla kluczy w wyniku (np. 'val' lub 'test').

    Returns
    -------
    dict z metrykami.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    results: dict[str, float | dict | list] = {}

    def _key(k: str) -> str:
        return f"{prefix}_{k}" if prefix else k

    # 1. Kappa
    results[_key("quadratic_weighted_kappa")] = quadratic_weighted_kappa(y_true, y_pred)
    results[_key("linear_weighted_kappa")] = linear_weighted_kappa(y_true, y_pred)
    results[_key("cohen_kappa")] = float(cohen_kappa_score(y_true, y_pred))

    # 2. Accuracy (uzupełniająco — UWAGA: nie używać jako głównej metryki)
    results[_key("accuracy")] = float((y_true == y_pred).mean())

    # 3. F1 macro / weighted
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results[_key("f1_macro")] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        results[_key("f1_weighted")] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # 4. AUC-ROC (One-vs-Rest)
    if y_proba is not None:
        try:
            # Macro AUC
            results[_key("auc_macro")] = float(roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            ))
            # Per-class AUC
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auc_per_class = roc_auc_score(y_true, y_proba, multi_class="ovr", average=None)
            for i, name in enumerate(class_names):
                if i < len(auc_per_class):
                    results[_key(f"auc_{name.lower()}")] = float(auc_per_class[i])

            # Log loss (multi-class)
            try:
                results[_key("log_loss")] = float(log_loss(
                    y_true, y_proba,
                    labels=list(range(len(class_names))),
                ))
            except Exception:
                pass
        except (ValueError, IndexError) as e:
            log.warning(f"Nie udało się policzyć AUC-ROC: {e}")

    # 5. METRYKI BEZPIECZEŃSTWA (KRYTYCZNE)
    results[_key("undertriage_rate")] = undertriage_rate(y_true, y_pred)
    results[_key("overtriage_rate")] = overtriage_rate(y_true, y_pred)
    results[_key("critical_miss_rate")] = critical_miss_rate(y_true, y_pred)

    # 6. Per-class metrics z classification_report
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report_dict = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
    for class_name in class_names:
        if class_name in report_dict:
            class_metrics = report_dict[class_name]
            results[_key(f"precision_{class_name.lower()}")] = float(class_metrics["precision"])
            results[_key(f"recall_{class_name.lower()}")] = float(class_metrics["recall"])
            results[_key(f"f1_{class_name.lower()}")] = float(class_metrics["f1-score"])
            results[_key(f"support_{class_name.lower()}")] = int(class_metrics["support"])

    # 7. Confusion matrix (zapisz jako lista list)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    results[_key("confusion_matrix")] = cm.tolist()

    if print_report:
        _print_evaluation(y_true, y_pred, results, class_names, cm, prefix)

    return results


def _print_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    results: dict,
    class_names: list[str],
    cm: np.ndarray,
    prefix: str,
) -> None:
    """Wyświetla czytelny raport."""
    p = f"{prefix}_" if prefix else ""

    log.info(f"{'═' * 60}")
    log.info(f"  EWALUACJA{' (' + prefix.upper() + ')' if prefix else ''}")
    log.info(f"{'═' * 60}")

    log.info(f"  Quadratic Weighted Kappa: {results[f'{p}quadratic_weighted_kappa']:.4f}")
    log.info(f"  Macro F1:                 {results[f'{p}f1_macro']:.4f}")
    if f"{p}auc_macro" in results:
        log.info(f"  Macro AUC-ROC:            {results[f'{p}auc_macro']:.4f}")

    log.info(f"\n  ─── METRYKI BEZPIECZEŃSTWA ───")
    log.info(f"  Undertriage rate:         {results[f'{p}undertriage_rate']:.4f}  (Red/Orange → Yellow/Green/Blue)")
    log.info(f"  Critical miss rate:       {results[f'{p}critical_miss_rate']:.4f}  (Red → Green/Blue)")
    log.info(f"  Overtriage rate:          {results[f'{p}overtriage_rate']:.4f}  (Green/Blue → Red/Orange)")

    log.info(f"\n  ─── PER-CLASS ───")
    for name in class_names:
        n = name.lower()
        if f"{p}precision_{n}" in results:
            log.info(
                f"  {name:8s} P={results[f'{p}precision_{n}']:.3f} "
                f"R={results[f'{p}recall_{n}']:.3f} "
                f"F1={results[f'{p}f1_{n}']:.3f} "
                f"AUC={results.get(f'{p}auc_{n}', float('nan')):.3f} "
                f"(n={results[f'{p}support_{n}']})"
            )

    # Confusion matrix
    log.info(f"\n  ─── CONFUSION MATRIX (wiersze=true, kolumny=pred) ───")
    header = "          " + " ".join(f"{n[:6]:>7}" for n in class_names)
    log.info(f"  {header}")
    for i, name in enumerate(class_names):
        row = "  ".join(f"{cm[i, j]:>7d}" for j in range(len(class_names)))
        log.info(f"  {name:8s}  {row}")
