"""
Korekta prawdopodobieństw na inny rozkład klas (prior shift) — przeniesienie modelu do innego SOR.

Przy założeniu, że zmienia się tylko rozkład ESI, a P(x | ESI) zostaje ten sam (label shift):

    P_nowe(k | x) ∝ P(k | x) · π_nowe(k) / π_trening(k)

π_nowe — znany lokalny rozkład ESI (np. z rejestru triażu szpitala) albo estymacja bez etykiet
algorytmem EM (Saerens, Latinne, Decaestecker 2002, Neural Computation 14(1)): na zmianę
E: skorygowane prawdopodobieństwa przy bieżącym π, M: π = średnia skorygowanych prawdopodobieństw.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.config import N_CLASSES

_EPS = 1e-12
# EM bywa niestabilne przy źle skalibrowanym modelu: klasa wyzerowana (π_EM / π_trening < MIN) albo
# zawyżona wielokrotnie (> MAX; rzadkie ESI 1 legalnie rośnie kilka–kilkanaście razy między oddziałami)
# → `suspect` w opisie wyniku (heurystyka do raportu, estymator się nie zmienia)
SUSPECT_MIN_RATIO: float = 0.1
SUSPECT_MAX_RATIO: float = 20.0


def class_prior(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Rozkład klas 0..n_classes−1 z etykiet (udziały, suma 1)."""
    y = np.asarray(y).astype(int)
    if len(y) == 0:
        raise ValueError("class_prior: puste etykiety")
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return counts / counts.sum()


def normalize_prior(prior: np.ndarray | list[float]) -> np.ndarray:
    """Nieujemny wektor → udziały (suma 1); ValueError dla zer / wartości ujemnych / NaN."""
    p = np.asarray(prior, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError(f"Nieprawidłowy rozkład klas: {prior}")
    return p / p.sum()


def apply_prior_shift(proba: np.ndarray, new_prior: np.ndarray, train_prior: np.ndarray) -> np.ndarray:
    """
    P(k | x) · π_nowe(k) / π_trening(k), znormalizowane w wierszu. Klasa nieobecna w treningu
    (π_trening = 0) zostaje z prawdopodobieństwem 0 (model jej nie przewiduje).
    """
    proba = np.asarray(proba, dtype=np.float64)
    new = normalize_prior(new_prior)
    old = normalize_prior(train_prior)
    if proba.ndim != 2 or proba.shape[1] != len(new) or len(new) != len(old):
        raise ValueError(f"apply_prior_shift: proba {proba.shape}, priory {len(new)} / {len(old)}")
    weights = np.where(old > 0, new / np.maximum(old, _EPS), 0.0)
    shifted = proba * weights
    total = shifted.sum(axis=1, keepdims=True)
    # Wiersz z samymi zerami (np. cała masa na klasie z wagą 0) — zostaw oryginał
    return np.where(total > 0, shifted / np.maximum(total, _EPS), proba)


def em_prior(
    proba: np.ndarray,
    train_prior: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Rozkład klas w NOWEJ populacji bez etykiet (EM, Saerens i in. 2002).

    Start: π = π_trening. Iteracja: P_π = apply_prior_shift(proba, π, π_trening), π ← średnia P_π.
    Koniec, gdy max |Δπ| < tol albo po max_iter iteracjach.

    Returns
    -------
    (π (n_classes,), {'iterations', 'converged', 'max_delta', 'min_ratio', 'max_ratio', 'suspect'})
    — ratio = π_EM / π_trening (klasy obecne w treningu); suspect = iloraz poza
    [SUSPECT_MIN_RATIO, SUSPECT_MAX_RATIO] (np. model nieskalibrowany — skalibruj przed EM; Alexandari
    i in. 2020) albo brak zbieżności.
    """
    proba = np.asarray(proba, dtype=np.float64)
    if len(proba) == 0:
        raise ValueError("em_prior: brak wierszy")
    old = normalize_prior(train_prior)
    pi = old.copy()
    delta = float("inf")
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        new = apply_prior_shift(proba, pi, old).mean(axis=0)
        delta = float(np.abs(new - pi).max())
        pi = new
        if delta < tol:
            break
    pi = pi / pi.sum()
    present = old > 0
    ratio = pi[present] / old[present]
    converged = bool(delta < tol)
    min_ratio, max_ratio = float(ratio.min()), float(ratio.max())
    suspect = (not converged) or min_ratio < SUSPECT_MIN_RATIO or max_ratio > SUSPECT_MAX_RATIO
    return pi, {"iterations": iterations, "converged": converged, "max_delta": delta,
                "min_ratio": min_ratio, "max_ratio": max_ratio, "suspect": bool(suspect)}


__all__ = ["apply_prior_shift", "class_prior", "em_prior", "normalize_prior"]
