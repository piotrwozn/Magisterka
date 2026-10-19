"""
Metryki ewaluacji dla 5-klasowej klasyfikacji ESI (indeks 0–4 = ESI 1–5).

Kluczowe metryki:
    1. QUADRATIC WEIGHTED KAPPA — główna metryka (ordinalna).
    2. UNDERTRIAGE RATE — KRYTYCZNA metryka bezpieczeństwa
       (ESI 1–2 przewidziane jako ESI 3–5).
    3. MISTRIAGE — niedo-/przetriażowanie o dowolny poziom (standard w literaturze).
    4. Per-class AUC-ROC, log loss, Brier — jakość prawdopodobieństw (ważne przy łączeniu modeli).
    5. 5×5 confusion matrix — wizualizacja błędów ordinalnych.

Decyzja: `decide(proba, "argmax")` albo `decide(proba, "cost")` — minimalizacja
oczekiwanego kosztu z macierzy, w której niedotriażowanie kosztuje 3× więcej.

UWAGA: Accuracy jest mylące — 98% accuracy może oznaczać 100% pominięć ESI 1.
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

from src.utils.config import (
    CLASS_NAMES,
    HIGH_ACUITY_CLASSES,
    LOW_ACUITY_CLASSES,
    N_CLASSES,
    OVERTRIAGE_COST_MULTIPLIER,
    UNDERTRIAGE_COST_MULTIPLIER,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────
# Metryki bezpieczeństwa (kluczowe dla SOR-AI)
# ─────────────────────────────────────────
def undertriage_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    high_acuity_classes: tuple[int, ...] = HIGH_ACUITY_CLASSES,
) -> float:
    """
    Procent pacjentów wymagających pilnej pomocy (ESI 1–2) błędnie zaklasyfikowanych
    jako mniej pilni (ESI 3–5).

    To NAJWAŻNIEJSZA metryka bezpieczeństwa w medycynie ratunkowej:
        Lepiej overtriage zdrowego niż przeoczyć zawał.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    high_acuity_classes : tuple[int]
        Indeksy klas o wysokiej pilności (domyślnie ESI 1 = 0, ESI 2 = 1).

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
    low_acuity_classes: tuple[int, ...] = LOW_ACUITY_CLASSES,
    threshold_class: int = 1,
) -> float:
    """
    Procent pacjentów ESI 4–5 błędnie zaklasyfikowanych jako ESI 1–2.

    Parameters
    ----------
    low_acuity_classes : tuple[int]
        Klasy niskiej pilności (ESI 4 = 3, ESI 5 = 4).
    threshold_class : int
        Predykcje ≤ threshold_class to przetriażowanie do wysokiej pilności (≤ ESI 2).
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
    Specjalny przypadek undertriage: ESI 1 (0) zaklasyfikowany jako ESI 4–5 (3, 4).
    To "missed critical" — najgorszy możliwy błąd.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = (y_true == 0)
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] >= 3).mean())


def mistriage_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Błędny triaż o dowolny poziom (definicja z literatury ESI):
        under = przewidziano NIŻSZĄ pilność (wyższy numer ESI) niż prawdziwa,
        over  = przewidziano WYŻSZĄ pilność niż prawdziwa.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "mistriage_under": float((y_pred > y_true).mean()),
        "mistriage_over": float((y_pred < y_true).mean()),
        "within_one_accuracy": float((np.abs(y_pred - y_true) <= 1).mean()),
    }


# ─────────────────────────────────────────
# Macierz kosztów i reguła decyzyjna
# ─────────────────────────────────────────
def cost_matrix(
    n_classes: int = N_CLASSES,
    under: float = UNDERTRIAGE_COST_MULTIPLIER,
    over: float = OVERTRIAGE_COST_MULTIPLIER,
) -> np.ndarray:
    """
    koszt[t, p] = (p - t)^2 × (under jeśli p > t, over jeśli p < t).

    p > t to niedotriażowanie (niższa pilność niż prawdziwa) — droższe.
    """
    t = np.arange(n_classes)[:, None]
    p = np.arange(n_classes)[None, :]
    dist2 = (p - t).astype(float) ** 2
    return np.where(p > t, under * dist2, over * dist2)


def decide(proba: np.ndarray, rule: str = "argmax", costs: np.ndarray | None = None) -> np.ndarray:
    """
    Prawdopodobieństwa → klasy.

    'argmax' — najbardziej prawdopodobna klasa.
    'cost'   — klasa minimalizująca oczekiwany koszt (Bayes decision):
               argmin_p Σ_t P(t | x) · koszt[t, p].
    """
    proba = np.asarray(proba, dtype=float)
    if rule == "argmax":
        return proba.argmax(axis=1)
    if rule == "cost":
        costs = cost_matrix(proba.shape[1]) if costs is None else costs
        return (proba @ costs).argmin(axis=1)
    raise ValueError(f"Nieznana reguła decyzyjna: '{rule}' (argmax | cost)")


def expected_cost(y_true: np.ndarray, y_pred: np.ndarray, costs: np.ndarray | None = None) -> float:
    """Średni koszt decyzji wg macierzy kosztów (im mniej, tym bezpieczniej)."""
    costs = cost_matrix() if costs is None else costs
    return float(costs[np.asarray(y_true), np.asarray(y_pred)].mean())


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Wieloklasowy Brier score (średnia po wierszach; im mniej, tym lepiej)."""
    proba = np.asarray(proba, dtype=float)
    onehot = np.eye(proba.shape[1])[np.asarray(y_true)]
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


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
        Prawdopodobieństwa per klasa ESI 1–5, shape (n, 5).
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

    # 4. AUC-ROC (One-vs-Rest) — liczone dla klas obecnych w y_true, tak żeby mała
    #    próbka bez np. ESI 1 nie kasowała wszystkich metryk probabilistycznych
    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=np.float64)
        labels = list(range(len(class_names)))
        aucs = []
        for i, name in enumerate(class_names):
            positives = y_true == i
            if positives.any() and not positives.all():
                auc_i = float(roc_auc_score(positives.astype(int), y_proba[:, i]))
                results[_key(f"auc_{name.lower()}")] = auc_i
                aucs.append(auc_i)
        if aucs:
            results[_key("auc_macro")] = float(np.mean(aucs))
        else:
            log.warning("Nie udało się policzyć AUC-ROC: w y_true jest tylko jedna klasa")

        # Log loss + Brier (jakość prawdopodobieństw)
        y_proba_norm = y_proba / y_proba.sum(axis=1, keepdims=True)
        results[_key("log_loss")] = float(log_loss(y_true, y_proba_norm, labels=labels))
        results[_key("brier")] = brier_score(y_true, y_proba_norm)

    # 5. METRYKI BEZPIECZEŃSTWA (KRYTYCZNE)
    results[_key("undertriage_rate")] = undertriage_rate(y_true, y_pred)
    results[_key("overtriage_rate")] = overtriage_rate(y_true, y_pred)
    results[_key("critical_miss_rate")] = critical_miss_rate(y_true, y_pred)
    for name, value in mistriage_rates(y_true, y_pred).items():
        results[_key(name)] = value
    results[_key("expected_cost")] = expected_cost(y_true, y_pred)

    # 6. Per-class metrics z classification_report
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report_dict = classification_report(
            y_true, y_pred,
            labels=list(range(len(class_names))),
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

    log.info("\n  ─── METRYKI BEZPIECZEŃSTWA ───")
    log.info(f"  Undertriage rate:         {results[f'{p}undertriage_rate']:.4f}  (ESI 1–2 → ESI 3–5)")
    log.info(f"  Critical miss rate:       {results[f'{p}critical_miss_rate']:.4f}  (ESI 1 → ESI 4–5)")
    log.info(f"  Overtriage rate:          {results[f'{p}overtriage_rate']:.4f}  (ESI 4–5 → ESI 1–2)")
    log.info(f"  Mistriage under / over:   {results[f'{p}mistriage_under']:.4f} / {results[f'{p}mistriage_over']:.4f}")
    log.info(f"  Expected cost:            {results[f'{p}expected_cost']:.4f}")

    log.info("\n  ─── PER-CLASS ───")
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
    log.info("\n  ─── CONFUSION MATRIX (wiersze=true, kolumny=pred) ───")
    header = "          " + " ".join(f"{n[:6]:>7}" for n in class_names)
    log.info(f"  {header}")
    for i, name in enumerate(class_names):
        row = "  ".join(f"{cm[i, j]:>7d}" for j in range(len(class_names)))
        log.info(f"  {name:8s}  {row}")
