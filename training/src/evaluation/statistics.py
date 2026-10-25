"""
Statystyka wyników na ZAPISANYCH predykcjach (bez ponownego treningu): przedziały ufności,
porównania parami i kalibracja prawdopodobieństw.

    from src.evaluation.statistics import bootstrap_ci, paired_bootstrap_diff, calibration_summary

    ci = bootstrap_ci(y, proba, rule="cost", n_boot=1000)      # DataFrame: metric, value, ci_low, ci_high, …
    d = paired_bootstrap_diff(y, p_blend, p_xgb, "quadratic_weighted_kappa")   # Δ = A − B
    calibration_summary(y, proba)                               # ECE (top-label, per klasa), MCE, …

Bootstrap percentylowy
    Losujemy n wierszy ze zwracaniem (domyślnie osobno w każdej prawdziwej klasie ESI —
    `stratify=True` — więc rzadkie ESI 1 jest w każdym losowaniu tyle samo razy co w danych),
    liczymy metrykę, powtarzamy n_boot razy. 95% przedział ufności = 2,5. i 97,5. percentyl
    tych wartości. Przedział opisuje niepewność wynikającą ze SKOŃCZONEGO zbioru testowego
    (inna próbka pacjentów z tej samej populacji dałaby nieco inny wynik); nie obejmuje
    zmienności treningu (inne ziarno, inne hiperparametry).

Odsetki pacjentów (czułość ESI 1, krytyczne pominięcia, niedotriażowanie, trafność…)
    Domyślnie (`proportion_ci="exact"`) dokładny przedział Clopper–Pearsona z liczebności k/n,
    a nie percentyle bootstrapu. Przy losowaniu w warstwach mianownik (np. liczba ESI 1) jest
    w każdym losowaniu taki sam, więc bootstrap odsetka to tylko ponowne losowanie obserwowanej
    proporcji: przy 0/38 zdarzeń daje „przedział” [0; 0], a przy kilku zdarzeniach wyraźnie za
    wąski. Clopper–Pearson przy 0/38 daje [0; 9,3%] — tyle niepewności rzeczywiście jest.
    Bootstrap zostaje dla QWK, AUC, log loss, Brier, ECE, kosztu i dla różnic sparowanych.

ECE (expected calibration error) jest obciążone w górę na skończonej próbie — nawet idealnie
skalibrowany model ma ECE > 0. Przedział bootstrap ECE nie mówi więc, czy model jest „skalibrowany”;
do tego służy `ece_consistency` (ECE idealnego modelu przy tej samej liczności + p-wartość),
a do porównań — sparowane Δ ECE na tych samych pacjentach.

Implementacja
    Losowanie ze zwracaniem = wagi całkowite w_i (ile razy wiersz i trafił do próby). Każda
    metryka jest liczona jako ważona suma, więc tysiąc losowań to mnożenia macierzy:
    macierz pomyłek (B × 25), log loss, Brier i ECE (B × biny), a AUC — szybki, rangowy
    estymator Manna–Whitneya na raz posortowanych wynikach (z obsługą remisów). Wiersze
    losowań generowane są po kolei z jednego generatora, więc wynik nie zależy od podziału
    na porcje (te same `seed` i `y` → te same losowania, także dla dwóch modeli naraz).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import cost_matrix, decide
from src.utils.config import CLASS_NAMES, HIGH_ACUITY_CLASSES, LOW_ACUITY_CLASSES

# ─────────────────────────────────────────
# Stałe
# ─────────────────────────────────────────
DEFAULT_CI_METRICS: tuple[str, ...] = (
    "quadratic_weighted_kappa",
    "accuracy",
    "within_one_accuracy",
    "undertriage_rate",
    "critical_miss_rate",
    "overtriage_rate",
    "mistriage_under",
    "mistriage_over",
    "recall_esi1",
    "recall_esi12",
    "auc_macro",
    "log_loss",
    "brier",
    "expected_cost",
    "ece",
)
# Metryki z samych prawdopodobieństw — nie zależą od reguły decyzyjnej
PROBA_METRICS: frozenset[str] = frozenset(
    {"auc_macro", "log_loss", "brier", "ece"} | {f"auc_{c.lower()}" for c in CLASS_NAMES}
)
# Metryki „im mniej, tym lepiej” (kierunek interpretacji Δ w porównaniach parami)
LOWER_IS_BETTER: frozenset[str] = frozenset({
    "undertriage_rate", "critical_miss_rate", "overtriage_rate", "mistriage_under", "mistriage_over",
    "log_loss", "brier", "expected_cost", "ece",
})
# Odsetki z macierzy pomyłek (licznik / mianownik w pacjentach) — przedział Clopper–Pearsona z k/n
PROPORTION_METRICS: frozenset[str] = frozenset(
    {"accuracy", "within_one_accuracy", "undertriage_rate", "critical_miss_rate", "overtriage_rate",
     "mistriage_under", "mistriage_over", "recall_esi12"} | {f"recall_{c.lower()}" for c in CLASS_NAMES}
)
PROPORTION_CI_MODES: tuple[str, ...] = ("exact", "bootstrap")
CI_METHOD_EXACT = "clopper-pearson"
CI_METHOD_BOOTSTRAP = "bootstrap-percentile"
DEFAULT_N_BINS: int = 15
CI_COLUMNS: list[str] = ["metric", "value", "ci_low", "ci_high", "se", "n_valid", "k", "n", "ci_method"]
PAIRED_COLUMNS: list[str] = ["rule", "metric", "value_a", "value_b", "delta", "ci_low", "ci_high",
                             "p_two_sided", "n_valid", "better"]

_LOG_EPS = float(np.finfo(np.float64).eps)      # jak sklearn.metrics.log_loss (float64)
_CHUNK_ELEMENTS = 3_000_000                      # elementów macierzy wag na porcję (≈ 24 MB float64)
_CLASS_METRIC = re.compile(r"^(recall|auc)_esi([1-9])$")


# ─────────────────────────────────────────
# Kontrola wejścia
# ─────────────────────────────────────────
def check_proba(proba: np.ndarray, name: str = "prawdopodobieństwa") -> np.ndarray:
    """
    Macierz (n, k) skończonych, nieujemnych prawdopodobieństw o dodatniej sumie wiersza,
    znormalizowana do sumy 1. Inaczej ValueError z liczbą złych wierszy — NaN nie może po cichu
    wyłączyć log loss / ECE ani trafić przez argmax do ESI 1.
    """
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError(f"{name}: oczekiwano macierzy (n, k), jest kształt {proba.shape}")
    bad = ~np.isfinite(proba).all(axis=1) | (proba < 0).any(axis=1)
    sums = np.where(bad, 1.0, proba.sum(axis=1))
    bad |= ~(sums > 0)
    if bad.any():
        raise ValueError(f"{name}: {int(bad.sum()):,} z {len(proba):,} wierszy ma wartości nieskończone / NaN, "
                         f"ujemne albo zerową sumę (np. wiersz {int(np.flatnonzero(bad)[0])})")
    return proba / sums[:, None]


# ─────────────────────────────────────────
# Losowania bootstrap (wagi = krotności wierszy)
# ─────────────────────────────────────────
def _strata(y: np.ndarray, stratify: bool) -> list[np.ndarray]:
    """Indeksy wierszy każdej warstwy (prawdziwa klasa) albo jedna warstwa = wszystkie wiersze."""
    if not stratify:
        return [np.arange(len(y))]
    return [np.flatnonzero(y == c) for c in np.unique(y)]


def _chunk_size(n: int, n_boot: int) -> int:
    return int(max(1, min(n_boot, _CHUNK_ELEMENTS // max(n, 1))))


def resample_weights(
    y: np.ndarray,
    n_boot: int,
    seed: int = 42,
    stratify: bool = True,
    chunk: int | None = None,
) -> Iterator[np.ndarray]:
    """
    Porcje macierzy krotności W (b × n): W[r, i] = ile razy wiersz i trafił do losowania r.

    stratify=True — losowanie ze zwracaniem osobno w każdej prawdziwej klasie (liczności klas
    w każdym losowaniu = liczności w danych). Losowania powstają po kolei z jednego generatora,
    więc ten sam `seed` daje te same losowania niezależnie od rozmiaru porcji.
    """
    y = np.asarray(y)
    n = len(y)
    if n == 0:
        raise ValueError("Pusty zbiór — brak wierszy do losowania")
    strata = _strata(y, stratify)
    rng = np.random.default_rng(seed)
    chunk = chunk or _chunk_size(n, n_boot)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        W = np.zeros((b, n), dtype=np.float64)
        for r in range(b):
            for idx in strata:
                draws = rng.integers(0, len(idx), size=len(idx))
                W[r, idx] = np.bincount(draws, minlength=len(idx))
        done += b
        yield W


# ─────────────────────────────────────────
# Ważone metryki (wagi = krotności z bootstrapu; wagi 1 = zwykła metryka)
# ─────────────────────────────────────────
def _validate_metrics(metrics: Sequence[str] | None) -> list[str]:
    names = list(DEFAULT_CI_METRICS if metrics is None else metrics)
    known = set(DEFAULT_CI_METRICS) | {f"{k}_{c.lower()}" for k in ("recall", "auc") for c in CLASS_NAMES}
    unknown = [m for m in names if m not in known]
    if unknown:
        raise ValueError(f"Nieznane metryki {unknown}; dostępne: {sorted(known)}")
    return names


def _cm_masks(n_classes: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Metryki z macierzy pomyłek jako (maska licznika, maska mianownika) 5 × 5 (wiersz = prawda)."""
    t = np.arange(n_classes)[:, None] * np.ones((1, n_classes), dtype=int)
    p = np.ones((n_classes, 1), dtype=int) * np.arange(n_classes)[None, :]
    every = np.ones((n_classes, n_classes))
    high = np.isin(t, HIGH_ACUITY_CLASSES)
    low = np.isin(t, LOW_ACUITY_CLASSES)
    masks = {
        "accuracy": (t == p, every),
        "within_one_accuracy": (np.abs(t - p) <= 1, every),
        "undertriage_rate": (high & (p > max(HIGH_ACUITY_CLASSES)), high),
        "critical_miss_rate": ((t == 0) & (p >= 3), t == 0),
        "overtriage_rate": (low & (p <= 1), low),
        "mistriage_under": (p > t, every),
        "mistriage_over": (p < t, every),
        "recall_esi12": (high & np.isin(p, HIGH_ACUITY_CLASSES), high),
    }
    for k, name in enumerate(CLASS_NAMES[:n_classes]):
        masks[f"recall_{name.lower()}"] = ((t == k) & (p == k), t == k)
    return {k: (num.astype(float).ravel(), den.astype(float).ravel()) for k, (num, den) in masks.items()}


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(np.shape(num), np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _qwk_from_cm(C: np.ndarray, n_classes: int) -> np.ndarray:
    """QWK (jak sklearn cohen_kappa_score(weights='quadratic')) z macierzy pomyłek (b × k × k)."""
    i = np.arange(n_classes)
    wq = ((i[:, None] - i[None, :]) ** 2).astype(float)
    total = C.sum(axis=(1, 2))
    rows = C.sum(axis=2)                     # prawdziwe klasy
    cols = C.sum(axis=1)                     # predykcje
    expected = rows[:, :, None] * cols[:, None, :] / np.where(total > 0, total, 1.0)[:, None, None]
    num = (C * wq).sum(axis=(1, 2))
    den = (expected * wq).sum(axis=(1, 2))
    return 1.0 - _ratio(num, den)


def _bin_index(scores: np.ndarray, n_bins: int, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    """Numer binu każdego wyniku i krawędzie binów ('uniform' — równe przedziały [0, 1]; 'quantile')."""
    scores = np.asarray(scores, dtype=float)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.floor(scores * n_bins).astype(int), 0, n_bins - 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([scores.min(), scores.max()])
        idx = np.clip(np.searchsorted(edges[1:-1], scores, side="right"), 0, len(edges) - 2)
    else:
        raise ValueError(f"Nieznana strategia binów '{strategy}' (uniform | quantile)")
    return idx, edges


class _AucPrep:
    """
    Rangowe (Mann–Whitney) AUC jednego wyniku przy dowolnych wagach wierszy; remisy = ½.

    U = Σ_{i ∈ pos} w_i · (Σ_{j ∈ neg, s_j < s_i} w_j + ½ Σ_{j ∈ neg, s_j = s_i} w_j).
    Raz sortujemy wyniki „drugiej” grupy i zapamiętujemy dla każdego wiersza „pierwszej” grupy
    pozycje (searchsorted lewa / prawa) — w każdym losowaniu wystarczy suma skumulowana wag
    drugiej grupy i dwa odczyty. Pierwsza = mniejsza z grup (mniej odczytów); gdy to negatywy,
    liczymy U' dla zamienionych ról i AUC = 1 − U'/(P·N).
    """

    def __init__(self, score: np.ndarray, positive: np.ndarray) -> None:
        score = np.asarray(score, dtype=float)
        positive = np.asarray(positive, dtype=bool)
        pos_idx, neg_idx = np.flatnonzero(positive), np.flatnonzero(~positive)
        self.swapped = len(pos_idx) > len(neg_idx)
        first, second = (neg_idx, pos_idx) if self.swapped else (pos_idx, neg_idx)
        order = np.argsort(score[second], kind="mergesort")
        self.second = second[order]
        second_sorted = score[self.second]
        self.first = first
        self.lo = np.searchsorted(second_sorted, score[first], side="left")
        self.hi = np.searchsorted(second_sorted, score[first], side="right")

    def __call__(self, W: np.ndarray) -> np.ndarray:
        w_first = W[:, self.first]
        w_second = W[:, self.second]
        cum = np.zeros((len(W), w_second.shape[1] + 1))
        np.cumsum(w_second, axis=1, out=cum[:, 1:])
        u = np.einsum("ij,ij->i", w_first, cum[:, self.lo] + cum[:, self.hi]) * 0.5
        p_n = w_first.sum(axis=1) * cum[:, -1]
        auc = _ratio(u, p_n)
        return 1.0 - auc if self.swapped else auc


class _Prepared:
    """
    Wstępnie policzone składniki metryk jednego zestawu prawdopodobieństw — potem każda
    porcja losowań W (b × n) to kilka mnożeń macierzy.
    """

    def __init__(
        self,
        y: np.ndarray,
        proba: np.ndarray,
        rules: Sequence[str],
        metrics: Sequence[str],
        n_bins: int = DEFAULT_N_BINS,
    ) -> None:
        y = np.asarray(y).astype(int)
        proba = np.asarray(proba, dtype=float)
        if proba.ndim != 2 or len(proba) != len(y):
            raise ValueError(f"Kształt prawdopodobieństw {proba.shape} nie pasuje do {len(y)} etykiet")
        proba = check_proba(proba)
        self.k = proba.shape[1]
        self.metrics = list(metrics)
        self.rules = list(rules)
        self.present = np.unique(y)
        decision = [m for m in self.metrics if m not in PROBA_METRICS]

        # Macierz pomyłek: one-hot komórki (prawda, predykcja) dla każdej reguły (n × k²)
        self.cell_onehot: dict[str, np.ndarray] = {}
        if decision:
            for rule in self.rules:
                cell = y * self.k + decide(proba, rule)
                onehot = np.zeros((len(y), self.k * self.k))
                onehot[np.arange(len(y)), cell] = 1.0
                self.cell_onehot[rule] = onehot
        self.masks = _cm_masks(self.k)
        self.costs = cost_matrix(self.k).ravel()

        # Prawdopodobieństwa: log loss, Brier, ECE (top-label, biny równe), AUC
        self.nll = self.brier = self.ece_terms = None
        if "log_loss" in self.metrics:
            self.nll = -np.log(np.clip(proba[np.arange(len(y)), y], _LOG_EPS, 1.0))
        if "brier" in self.metrics:
            self.brier = ((proba - np.eye(self.k)[y]) ** 2).sum(axis=1)
        if "ece" in self.metrics:
            conf = proba.max(axis=1)
            correct = (proba.argmax(axis=1) == y).astype(float)
            idx, _ = _bin_index(conf, n_bins, "uniform")
            self.ece_terms = np.zeros((len(y), n_bins))
            self.ece_terms[np.arange(len(y)), idx] = conf - correct
        self.auc: dict[int, _AucPrep] = {}
        wanted = set()
        if "auc_macro" in self.metrics:
            wanted |= set(int(c) for c in self.present)
        for m in self.metrics:
            match = _CLASS_METRIC.match(m)
            if match and match.group(1) == "auc":
                wanted.add(int(match.group(2)) - 1)
        for c in sorted(wanted):
            if c < self.k:
                self.auc[c] = _AucPrep(proba[:, c], y == c)

    def counts(self) -> dict[tuple[str, str], tuple[int, int]]:
        """(reguła, odsetek) → (k, n): licznik i mianownik odsetka na całym zbiorze (w pacjentach)."""
        out: dict[tuple[str, str], tuple[int, int]] = {}
        for rule, onehot in self.cell_onehot.items():
            cm = onehot.sum(axis=0)
            for m in self.metrics:
                if m in PROPORTION_METRICS:
                    num, den = self.masks[m]
                    out[(rule, m)] = (int(round(cm @ num)), int(round(cm @ den)))
        return out

    def compute(self, W: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
        """(reguła, metryka) → wartości dla każdego wiersza W; metryki prawdopodobieństw pod regułą '*'."""
        out: dict[tuple[str, str], np.ndarray] = {}
        total = W.sum(axis=1)
        for rule, onehot in self.cell_onehot.items():
            C = W @ onehot                                   # (b × k²)
            for m in self.metrics:
                if m in PROBA_METRICS:
                    continue
                if m == "quadratic_weighted_kappa":
                    out[(rule, m)] = _qwk_from_cm(C.reshape(-1, self.k, self.k), self.k)
                elif m == "expected_cost":
                    out[(rule, m)] = _ratio(C @ self.costs, total)
                else:
                    num, den = self.masks[m]
                    out[(rule, m)] = _ratio(C @ num, C @ den)
        if self.nll is not None:
            out[("*", "log_loss")] = _ratio(W @ self.nll, total)
        if self.brier is not None:
            out[("*", "brier")] = _ratio(W @ self.brier, total)
        if self.ece_terms is not None:
            out[("*", "ece")] = _ratio(np.abs(W @ self.ece_terms).sum(axis=1), total)
        aucs = {c: prep(W) for c, prep in self.auc.items()}
        if "auc_macro" in self.metrics:
            # Makro po klasach obecnych w danych; losowanie bez którejś z nich → metryka nieokreślona
            stacked = np.stack([aucs[int(c)] for c in self.present])
            out[("*", "auc_macro")] = stacked.mean(axis=0) if len(self.present) > 1 else np.full(len(W), np.nan)
        for m in self.metrics:
            match = _CLASS_METRIC.match(m)
            if match and match.group(1) == "auc":
                c = int(match.group(2)) - 1
                out[("*", m)] = aucs.get(c, np.full(len(W), np.nan))
        return out


def _key(rule: str, metric: str) -> tuple[str, str]:
    return ("*", metric) if metric in PROBA_METRICS else (rule, metric)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Dokładny (Clopper–Pearson) przedział ufności 1 − alpha dla odsetka k/n.

    Konserwatywny (pokrycie ≥ 1 − alpha dla każdego prawdziwego odsetka) i poprawny także przy
    0 albo n zdarzeniach: 0/38 → [0; 0,093]. n = 0 → (NaN, NaN).
    """
    from scipy.stats import beta

    k, n = int(k), int(n)
    if n <= 0:
        return float("nan"), float("nan")
    if not 0 <= k <= n:
        raise ValueError(f"Liczba zdarzeń {k} poza zakresem 0…{n}")
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return low, high


def _summarize(
    point: float,
    samples: np.ndarray,
    alpha: float,
    counts: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """
    Wartość, przedział, bootstrapowy błąd standardowy i liczba użytecznych losowań.

    counts=(k, n) — odsetek: przedział Clopper–Pearsona z liczebności zamiast percentyli
    bootstrapu (kolumny k, n i ci_method to pokazują).
    """
    valid = samples[np.isfinite(samples)]
    k, n = counts if counts is not None else (np.nan, np.nan)
    method = CI_METHOD_EXACT if counts is not None else CI_METHOD_BOOTSTRAP
    se = float(valid.std(ddof=1)) if len(valid) > 1 else np.nan
    if counts is not None:
        low, high = clopper_pearson(k, n, alpha)
        return {"value": point, "ci_low": low, "ci_high": high, "se": se, "n_valid": int(len(valid)),
                "k": k, "n": n, "ci_method": method}
    if len(valid) == 0:
        return {"value": point, "ci_low": np.nan, "ci_high": np.nan, "se": np.nan, "n_valid": 0,
                "k": k, "n": n, "ci_method": method}
    low, high = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"value": point, "ci_low": float(low), "ci_high": float(high), "se": se, "n_valid": int(len(valid)),
            "k": k, "n": n, "ci_method": method}


def _check_inputs(y: np.ndarray, n_boot: int, alpha: float) -> np.ndarray:
    y = np.asarray(y).astype(int)
    if len(y) == 0:
        raise ValueError("Pusty zbiór — brak wierszy do oceny")
    if n_boot < 1:
        raise ValueError("n_boot musi być ≥ 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha musi być w (0, 1)")
    return y


# ─────────────────────────────────────────
# Przedziały ufności
# ─────────────────────────────────────────
def point_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    metrics: Sequence[str] | None = None,
    rule: str = "argmax",
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, float]:
    """Wartości metryk na całym zbiorze (te same definicje co w bootstrapie)."""
    y = np.asarray(y_true).astype(int)
    names = _validate_metrics(metrics)
    prep = _Prepared(y, proba, [rule], names, n_bins)
    values = prep.compute(np.ones((1, len(y))))
    return {m: float(values[_key(rule, m)][0]) for m in names}


@dataclass
class BootstrapSamples:
    """
    Wyniki jednego bootstrapu dla kilku modeli na TYCH SAMYCH losowaniach.

    point[model][(reguła, metryka)]   — wartość na całym zbiorze,
    samples[model][(reguła, metryka)] — n_boot wartości (NaN = metryka nieokreślona w losowaniu),
    counts[model][(reguła, odsetek)]  — (k, n) odsetka na całym zbiorze;
    reguła '*' = metryka z samych prawdopodobieństw. Ponieważ losowania są wspólne, różnica
    próbek dwóch modeli to sparowany bootstrap (`paired`). proportion_ci='exact' — przedziały
    odsetków z Clopper–Pearsona (k/n), 'bootstrap' — z percentyli losowań.
    """

    names: list[str]
    rules: list[str]
    metrics: list[str]
    n: int
    n_boot: int
    alpha: float
    seed: int
    stratify: bool
    point: dict[str, dict[tuple[str, str], float]] = field(default_factory=dict)
    samples: dict[str, dict[tuple[str, str], np.ndarray]] = field(default_factory=dict)
    counts: dict[str, dict[tuple[str, str], tuple[int, int]]] = field(default_factory=dict)
    proportion_ci: str = "exact"

    def ci(self, name: str, rule: str | None = None) -> pd.DataFrame:
        """DataFrame[metric, value, ci_low, ci_high, se, n_valid, k, n, ci_method] modelu dla reguły."""
        rule = rule or self.rules[0]
        rows = []
        for m in self.metrics:
            k = _key(rule, m)
            counts = self.counts.get(name, {}).get(k) if self.proportion_ci == "exact" else None
            rows.append({"metric": m, **_summarize(self.point[name][k], self.samples[name][k], self.alpha, counts)})
        return pd.DataFrame(rows, columns=CI_COLUMNS)

    def ci_rules(self, name: str) -> dict[str, pd.DataFrame]:
        return {rule: self.ci(name, rule) for rule in self.rules}

    def paired(
        self,
        a: str,
        b: str,
        metrics: Sequence[str] | None = None,
        rules: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Sparowane Δ = metryka(a) − metryka(b) (te same losowania).

        Returns
        -------
        DataFrame[rule, metric, value_a, value_b, delta, ci_low, ci_high, p_two_sided, n_valid,
                  better] — `rule` = '*' dla metryk niezależnych od reguły; `better` = 'A' / 'B' /
        'brak istotnej różnicy' (przedział zawiera 0), z uwzględnieniem kierunku metryki.
        """
        names = list(self.metrics if metrics is None else metrics)
        rows = []
        seen: set[tuple[str, str]] = set()
        for rule in (self.rules if rules is None else rules):
            for m in names:
                k = _key(rule, m)
                if k in seen:
                    continue
                seen.add(k)
                pa, pb = self.point[a][k], self.point[b][k]
                d = self.samples[a][k] - self.samples[b][k]
                rows.append({"rule": k[0], "metric": m, "value_a": pa, "value_b": pb,
                             **_paired_summary(pa - pb, d, self.alpha, m)})
        return pd.DataFrame(rows, columns=PAIRED_COLUMNS)


def bootstrap_samples(
    y_true: np.ndarray,
    probas: dict[str, np.ndarray],
    rules: Sequence[str] = ("argmax", "cost"),
    metrics: Sequence[str] | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    stratify: bool = True,
    n_bins: int = DEFAULT_N_BINS,
    proportion_ci: str = "exact",
) -> BootstrapSamples:
    """
    Bootstrap wielu modeli naraz: jedna macierz losowań na porcję, wszystkie modele, reguły
    i metryki. Podstawa `bootstrap_ci`, `paired_bootstrap_diff` i raportu końcowego.
    proportion_ci: 'exact' (Clopper–Pearson z k/n dla odsetków) | 'bootstrap'.
    """
    y = _check_inputs(y_true, n_boot, alpha)
    names = _validate_metrics(metrics)
    if not probas:
        raise ValueError("Brak prawdopodobieństw do oceny")
    if proportion_ci not in PROPORTION_CI_MODES:
        raise ValueError(f"proportion_ci musi być jednym z {PROPORTION_CI_MODES}")
    rules = list(rules)
    preps = {name: _Prepared(y, p, rules, names, n_bins) for name, p in probas.items()}
    ones = np.ones((1, len(y)))
    result = BootstrapSamples(names=list(probas), rules=rules, metrics=names, n=len(y), n_boot=int(n_boot),
                              alpha=float(alpha), seed=int(seed), stratify=bool(stratify),
                              proportion_ci=proportion_ci)
    parts: dict[str, dict[tuple[str, str], list[np.ndarray]]] = {}
    for name, prep in preps.items():
        result.point[name] = {k: float(v[0]) for k, v in prep.compute(ones).items()}
        result.counts[name] = prep.counts()
        parts[name] = {k: [] for k in result.point[name]}
    for W in resample_weights(y, n_boot, seed, stratify):
        for name, prep in preps.items():
            for k, v in prep.compute(W).items():
                parts[name][k].append(v)
    result.samples = {name: {k: np.concatenate(v) for k, v in d.items()} for name, d in parts.items()}
    return result


def bootstrap_ci_rules(
    y_true: np.ndarray,
    proba: np.ndarray,
    rules: Sequence[str] = ("argmax", "cost"),
    metrics: Sequence[str] | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    stratify: bool = True,
    n_bins: int = DEFAULT_N_BINS,
    proportion_ci: str = "exact",
) -> dict[str, pd.DataFrame]:
    """
    Jak `bootstrap_ci`, ale dla kilku reguł decyzyjnych naraz (te same losowania; metryki
    prawdopodobieństw liczone raz). Zwraca {reguła: DataFrame (kolumny CI_COLUMNS)}.
    """
    res = bootstrap_samples(y_true, {"model": proba}, rules, metrics, n_boot, alpha, seed, stratify, n_bins,
                            proportion_ci)
    return res.ci_rules("model")


def bootstrap_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    metrics: Sequence[str] | None = None,
    rule: str = "argmax",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    stratify: bool = True,
    n_bins: int = DEFAULT_N_BINS,
    proportion_ci: str = "exact",
) -> pd.DataFrame:
    """
    Przedziały ufności dla metryk ESI: bootstrap percentylowy, a dla odsetków pacjentów
    (domyślnie) dokładny przedział Clopper–Pearsona z liczebności k/n.

    Parameters
    ----------
    y_true : (n,) klasy 0–4; proba : (n, 5) prawdopodobieństwa
    metrics : nazwy metryk (domyślnie DEFAULT_CI_METRICS; także recall_esiK / auc_esiK)
    rule : reguła decyzyjna dla metryk z decyzji ('argmax' | 'cost')
    stratify : losowanie osobno w każdej prawdziwej klasie (rzadkie ESI 1 zawsze obecne)
    proportion_ci : 'exact' — odsetki (PROPORTION_METRICS) z przedziałem Clopper–Pearsona z k/n
        (poprawny także przy 0 zdarzeń: 0/38 → [0; 9,3%]); 'bootstrap' — percentyle losowań
        (przy warstwach i rzadkich zdarzeniach za wąskie, 0 zdarzeń → [0; 0])

    Returns
    -------
    DataFrame[metric, value, ci_low, ci_high, se, n_valid, k, n, ci_method] — `value` na całym
    zbiorze, `se` = odchylenie standardowe wartości z losowań (bootstrapowy błąd standardowy),
    `n_valid` = liczba losowań, w których metryka była określona (np. bez ESI 1 w losowaniu
    recall_esi1 jest pomijany), `k`/`n` = licznik i mianownik odsetka (NaN dla innych metryk),
    `ci_method` = 'clopper-pearson' | 'bootstrap-percentile'.
    """
    return bootstrap_ci_rules(y_true, proba, [rule], metrics, n_boot, alpha, seed, stratify, n_bins,
                              proportion_ci)[rule]


def _paired_summary(delta: float, d: np.ndarray, alpha: float, metric: str) -> dict[str, Any]:
    """
    Przedział, p i wniosek z JEDNYCH I TYCH SAMYCH ogonów rozkładu Δ (spójne z definicji).

    Przedział = statystyki pozycyjne [Δ_(c), Δ_(B−c+1)], c = ⌈B·alpha/2⌉ (c-ta najmniejsza i c-ta
    największa wartość — percentyle bez interpolacji). p = min(1, 2·min(P(Δ ≤ 0), P(Δ ≥ 0)))
    (konserwatywne: losowania z Δ dokładnie 0, częste przy odsetkach, liczą się do OBU ogonów;
    identyczne modele → p = 1). Wtedy: przedział bez zera ⇔ mniej niż c losowań po „drugiej”
    stronie zera ⇔ p < alpha ⇔ wniosek „A”/„B”. Rozdzielczość bootstrapu: p ≥ 1/B (przy
    B ≥ 2/alpha nie psuje zgodności).
    """
    valid = d[np.isfinite(d)]
    if len(valid) == 0:
        return {"delta": delta, "ci_low": np.nan, "ci_high": np.nan, "p_two_sided": np.nan,
                "n_valid": 0, "better": "nieokreślone"}
    b = len(valid)
    ordered = np.sort(valid)
    c = max(1, int(np.ceil(round(b * alpha / 2, 9))))
    low, high = float(ordered[c - 1]), float(ordered[b - c])
    n_le, n_ge = int(np.sum(valid <= 0)), int(np.sum(valid >= 0))
    p = float(min(1.0, max(2.0 * min(n_le, n_ge) / b, 1.0 / b)))
    if min(n_le, n_ge) >= c:                       # ⇔ low ≤ 0 ≤ high
        better = "brak istotnej różnicy"
    else:
        a_higher = n_le < c                        # ⇔ low > 0
        better = "B" if a_higher == (metric in LOWER_IS_BETTER) else "A"
    return {"delta": delta, "ci_low": low, "ci_high": high, "p_two_sided": p, "n_valid": int(b), "better": better}


def paired_bootstrap_diffs(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    metrics: Sequence[str],
    rules: Sequence[str] = ("argmax",),
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    stratify: bool = True,
    n_bins: int = DEFAULT_N_BINS,
) -> pd.DataFrame:
    """Sparowane Δ = metryka(A) − metryka(B) dla kilku metryk i reguł (zob. `paired_bootstrap_diff`)."""
    res = bootstrap_samples(y_true, {"A": proba_a, "B": proba_b}, rules, metrics, n_boot, alpha, seed,
                            stratify, n_bins)
    return res.paired("A", "B")


def paired_bootstrap_diff(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    metric: str = "quadratic_weighted_kappa",
    rule: str = "argmax",
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    stratify: bool = True,
) -> dict[str, Any]:
    """
    Sparowany bootstrap różnicy Δ = metryka(A) − metryka(B) na TYCH SAMYCH pacjentach.

    W każdym losowaniu oba modele oceniane są na identycznym zestawie wierszy, więc zmienność
    „trudności pacjentów” znosi się w różnicy. Taki przedział jest zwykle dużo węższy niż
    porównywanie dwóch osobnych przedziałów — nakładające się przedziały pojedynczych modeli
    NIE oznaczają braku różnicy.

    Jak czytać:
        delta        — różnica na całym zbiorze (A − B); dla QWK/AUC dodatnia = A lepszy,
                       dla undertriage / log loss / kosztu dodatnia = A GORSZY;
        ci_low/high  — 95% przedział (percentylowy, c-ta najmniejsza i c-ta największa Δ,
                       c = ⌈B·alpha/2⌉) różnicy; nie zawiera 0 → różnica istotna na poziomie alpha;
        p_two_sided  — dwustronna p-wartość z tych samych ogonów: 2 × odsetek losowań z Δ po
                       „drugiej” stronie zera, Δ = 0 liczone do obu stron (konserwatywnie;
                       najmniej 1/B); p < alpha ⇔ przedział nie zawiera 0;
        n_valid      — liczba losowań, w których metryka była określona dla obu modeli;
        better       — 'A' / 'B' / 'brak istotnej różnicy' z uwzględnieniem kierunku metryki.

    Returns
    -------
    dict(value_a, value_b, delta, ci_low, ci_high, p_two_sided, n_valid, better)
    """
    table = paired_bootstrap_diffs(y_true, proba_a, proba_b, [metric], [rule], n_boot, alpha, seed, stratify)
    row = table.iloc[0].to_dict()
    row.pop("rule", None)
    row.pop("metric", None)
    return row


# ─────────────────────────────────────────
# AUC dowolnych wyników (np. trafność względem przyjęcia do szpitala)
# ─────────────────────────────────────────
def bootstrap_auc(
    y_binary: np.ndarray,
    scores: dict[str, np.ndarray],
    reference: str | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    stratify: bool = True,
) -> pd.DataFrame:
    """
    AUC kilku wyników względem binarnego kryterium (te same losowania dla wszystkich).

    Wyższy wynik = większe prawdopodobieństwo zdarzenia (np. przyjęcia). Z `reference`
    dochodzą wiersze sparowanych różnic `Δ <wynik> − <reference>`.

    Returns
    -------
    DataFrame[score, value, ci_low, ci_high, se, n_valid] (+ wiersze 'Δ …' z kolumną p_two_sided)
    """
    y = _check_inputs(np.asarray(y_binary).astype(int), n_boot, alpha)
    if len(np.unique(y)) < 2:
        raise ValueError("Kryterium binarne ma jedną klasę — AUC nieokreślone")
    preps = {name: _AucPrep(s, y == 1) for name, s in scores.items()}
    ones = np.ones((1, len(y)))
    point = {name: float(p(ones)[0]) for name, p in preps.items()}
    samples: dict[str, list[np.ndarray]] = {name: [] for name in preps}
    for W in resample_weights(y, n_boot, seed, stratify):
        for name, prep in preps.items():
            samples[name].append(prep(W))
    rows = []
    for name in preps:
        summary = _summarize(point[name], np.concatenate(samples[name]), alpha)
        rows.append({"score": name, **{k: v for k, v in summary.items() if k not in ("k", "n")}})
    if reference is not None:
        if reference not in preps:
            raise ValueError(f"Brak wyniku odniesienia '{reference}'")
        ref = np.concatenate(samples[reference])
        for name in preps:
            if name == reference:
                continue
            d = np.concatenate(samples[name]) - ref
            summary = _paired_summary(point[name] - point[reference], d, alpha, "auc")
            rows.append({"score": f"Δ {name} − {reference}", "value": summary["delta"],
                         "ci_low": summary["ci_low"], "ci_high": summary["ci_high"], "se": np.nan,
                         "n_valid": summary["n_valid"], "p_two_sided": summary["p_two_sided"]})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# Kalibracja
# ─────────────────────────────────────────
def calibration_error(
    scores: np.ndarray, hits: np.ndarray, n_bins: int = DEFAULT_N_BINS, strategy: str = "uniform"
) -> tuple[float, float]:
    """(ECE, MCE): ważona liczebnością / największa różnica |trafność − średnie p| w binach."""
    idx, edges = _bin_index(scores, n_bins, strategy)
    m = len(edges) - 1
    count = np.bincount(idx, minlength=m).astype(float)
    gap = np.bincount(idx, weights=scores - hits, minlength=m)
    ece = float(np.abs(gap).sum() / len(scores))
    nonempty = count > 0
    mce = float(np.max(np.abs(gap[nonempty]) / count[nonempty])) if nonempty.any() else float("nan")
    return ece, mce


def ece_top_label(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    strategy: str = "uniform",
) -> float:
    """
    Expected Calibration Error dla NAJBARDZIEJ prawdopodobnej klasy (top-label, „confidence”):
    ECE = Σ_b (n_b / n) · |trafność_b − średnia pewność_b|. 0 = idealna kalibracja.
    """
    proba = check_proba(proba)
    hits = (proba.argmax(axis=1) == np.asarray(y_true)).astype(float)
    return calibration_error(proba.max(axis=1), hits, n_bins, strategy)[0]


def ece_classwise(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    strategy: str = "uniform",
) -> dict[str, float]:
    """ECE jeden-przeciw-reszcie dla każdego poziomu ESI (p_k vs y == k) + średnia ('mean')."""
    y = np.asarray(y_true)
    proba = check_proba(proba)
    out = {}
    for k in range(proba.shape[1]):
        name = CLASS_NAMES[k] if k < len(CLASS_NAMES) else f"class{k}"
        out[name] = calibration_error(proba[:, k], (y == k).astype(float), n_bins, strategy)[0]
    out["mean"] = float(np.mean(list(out.values())))
    return out


def reliability_table(
    y_true: np.ndarray,
    proba: np.ndarray,
    class_idx: int | None = None,
    n_bins: int = DEFAULT_N_BINS,
    strategy: str = "uniform",
) -> pd.DataFrame:
    """
    Tabela do wykresu niezawodności (puste biny pominięte).

    class_idx=None — top-label (pewność = max p, trafienie = argmax == y);
    class_idx=k    — poziom ESI k+1 jeden-przeciw-reszcie (p_k, y == k).

    Returns
    -------
    DataFrame[bin, bin_low, bin_high, mean_pred, observed_freq, count]
    """
    y = np.asarray(y_true)
    proba = check_proba(proba)
    if class_idx is None:
        scores = proba.max(axis=1)
        hits = (proba.argmax(axis=1) == y).astype(float)
    else:
        scores = proba[:, class_idx]
        hits = (y == class_idx).astype(float)
    idx, edges = _bin_index(scores, n_bins, strategy)
    m = len(edges) - 1
    count = np.bincount(idx, minlength=m)
    sum_pred = np.bincount(idx, weights=scores, minlength=m)
    sum_hit = np.bincount(idx, weights=hits, minlength=m)
    nonempty = count > 0
    bins = np.arange(m)[nonempty]
    return pd.DataFrame({
        "bin": bins,
        "bin_low": edges[:-1][nonempty],
        "bin_high": edges[1:][nonempty],
        "mean_pred": sum_pred[nonempty] / count[nonempty],
        "observed_freq": sum_hit[nonempty] / count[nonempty],
        "count": count[nonempty].astype(int),
    })


def ece_consistency(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_sim: int = 200,
    n_bins: int = DEFAULT_N_BINS,
    strategy: str = "uniform",
    seed: int = 42,
) -> dict[str, float]:
    """
    Wartość odniesienia dla ECE i test kalibracji („consistency resampling”, Bröcker i Smith 2007).

    ECE z binów jest obciążone w górę: nawet IDEALNIE skalibrowany model ma na skończonej próbie
    ECE > 0 (szum w binach), a przedział bootstrap ECE opisuje zmienność tej obciążonej wielkości —
    nie odległość od zera. Dlatego losujemy n_sim razy „trafienia” z pewności samego modelu
    (trafienie ~ Bernoulli(max p) — świat, w którym model jest idealnie skalibrowany) i liczymy ECE
    przy tej samej liczności i tych samych binach.

    Returns
    -------
    {"ece": obserwowane, "null_mean": średnie ECE idealnego modelu, "null_q95": jego 95. percentyl,
     "p_value": odsetek losowań z ECE ≥ obserwowanego (z poprawką +1) — małe p = kalibracja
     gorsza niż wynikałoby z samego szumu}
    """
    y = np.asarray(y_true)
    proba = check_proba(proba)
    conf = proba.max(axis=1)
    observed = calibration_error(conf, (proba.argmax(axis=1) == y).astype(float), n_bins, strategy)[0]
    idx, edges = _bin_index(conf, n_bins, strategy)
    m = len(edges) - 1
    rng = np.random.default_rng(seed)
    n = len(conf)
    null = []
    chunk = int(max(1, min(n_sim, _CHUNK_ELEMENTS // max(n, 1))))
    done = 0
    while done < n_sim:
        b = min(chunk, n_sim - done)
        hits = rng.random((b, n)) < conf                    # trafienie z prawdopodobieństwem = pewność
        flat = (np.arange(b)[:, None] * m + idx[None, :]).ravel()
        gaps = np.bincount(flat, weights=(conf[None, :] - hits).ravel(), minlength=b * m).reshape(b, m)
        null.append(np.abs(gaps).sum(axis=1) / n)
        done += b
    null_arr = np.concatenate(null)
    return {
        "ece": float(observed),
        "null_mean": float(null_arr.mean()),
        "null_q95": float(np.quantile(null_arr, 0.95)),
        "p_value": float((np.sum(null_arr >= observed) + 1) / (len(null_arr) + 1)),
        "n_sim": int(n_sim),
    }


def calibration_summary(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    n_null: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Komplet miar kalibracji:
        ece_top_label (biny równe) i ece_top_label_quantile (biny o równej liczności),
        mce_top_label (najgorszy bin), ece_classwise (per poziom ESI + mean),
        mean_confidence − accuracy = overconfidence (> 0: model zbyt pewny siebie),
        log_loss, brier (właściwe reguły oceny — łączą kalibrację i rozdzielczość).
    n_null > 0 dodaje `ece_consistency` (ECE idealnie skalibrowanego modelu przy tej liczności
    i p-wartość testu kalibracji) pod kluczem "ece_reference".
    """
    y = np.asarray(y_true).astype(int)
    proba = check_proba(proba)
    hits = (proba.argmax(axis=1) == y).astype(float)
    conf = proba.max(axis=1)
    ece_u, mce_u = calibration_error(conf, hits, n_bins, "uniform")
    ece_q, _ = calibration_error(conf, hits, n_bins, "quantile")
    point = point_metrics(y, proba, ["log_loss", "brier"])
    return {
        "n": int(len(y)),
        "n_bins": int(n_bins),
        "ece_top_label": ece_u,
        "ece_top_label_quantile": ece_q,
        "mce_top_label": mce_u,
        "ece_classwise": ece_classwise(y, proba, n_bins),
        "mean_confidence": float(conf.mean()),
        "accuracy": float(hits.mean()),
        "overconfidence": float(conf.mean() - hits.mean()),
        "log_loss": point["log_loss"],
        "brier": point["brier"],
        "ece_reference": ece_consistency(y, proba, n_null, n_bins, seed=seed) if n_null > 0 else None,
    }


# ─────────────────────────────────────────
# Temperatura (dopasowanie WYŁĄCZNIE na val)
# ─────────────────────────────────────────
def calibrate_on_val(
    p_val: np.ndarray,
    y_val: np.ndarray,
    p_test: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray | None]:
    """
    Temperatura dopasowana na VAL i zastosowana do val oraz (opcjonalnie) test.

    Uwaga: metryki `p_val_cal` na val są optymistyczne (T dopasowano na tych samych wierszach) —
    uczciwą ocenę val daje `crossfit_temperature`.

    Returns
    -------
    (T, p_val_cal, p_test_cal | None)
    """
    from src.ensemble.runs import apply_temperature, fit_temperature

    t = fit_temperature(np.asarray(p_val, dtype=float), np.asarray(y_val))
    p_test_cal = None if p_test is None else apply_temperature(np.asarray(p_test, dtype=float), t)
    return t, apply_temperature(np.asarray(p_val, dtype=float), t), p_test_cal


def stratified_folds(y: np.ndarray, n_folds: int = 2, seed: int = 42) -> np.ndarray:
    """Numer foldu każdego wiersza; w każdej klasie wiersze rozdzielone po równo (losowo)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        folds[idx] = np.arange(len(idx)) % n_folds
    return folds


def crossfit_temperature(
    p_val: np.ndarray,
    y_val: np.ndarray,
    n_folds: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, list[float]]:
    """
    Uczciwa („cross-fitted”) kalibracja val: T dopasowana na pozostałych foldach val
    i zastosowana do bieżącego foldu. Metryki na wyniku nie widzą własnych etykiet przy
    doborze T — przybliżają to, co temperatura da na nowych danych.

    Returns
    -------
    (p_val_crossfit, [T foldu 0, T foldu 1, …])
    """
    from src.ensemble.runs import apply_temperature, fit_temperature

    p_val = np.asarray(p_val, dtype=float)
    y_val = np.asarray(y_val)
    folds = stratified_folds(y_val, n_folds, seed)
    out = np.empty_like(p_val)
    temps = []
    for f in range(n_folds):
        held = folds == f
        t = fit_temperature(p_val[~held], y_val[~held])
        out[held] = apply_temperature(p_val[held], t)
        temps.append(float(t))
    return out, temps


# ─────────────────────────────────────────
# Formatowanie / JSON
# ─────────────────────────────────────────
def ci_to_dict(df: pd.DataFrame, key: str = "metric") -> dict[str, dict[str, Any]]:
    """
    DataFrame z przedziałami → {metryka: {value, ci_low, ci_high, se, n_valid, k, n, ci_method}}
    (NaN → None; k, n, n_valid jako liczby całkowite).
    """
    out = {}
    for row in df.to_dict(orient="records"):
        name = row.pop(key)
        clean = {k: _clean(v) for k, v in row.items()}
        for k in ("k", "n", "n_valid"):
            if isinstance(clean.get(k), float):
                clean[k] = int(round(clean[k]))
        out[str(name)] = clean
    return out


def _clean(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    return value


def format_ci(value: float | None, low: float | None, high: float | None, digits: int = 3,
              percent: bool = False, decimal_comma: bool = False) -> str:
    """'0.712 [0.701; 0.723]' (albo w procentach: '12.3% [11.0; 13.6]'); brak wartości → '—'."""
    def one(v: float | None) -> str:
        if v is None or not np.isfinite(v):
            return "—"
        text = f"{100 * v:.{max(digits - 2, 1)}f}" if percent else f"{v:.{digits}f}"
        return text.replace(".", ",") if decimal_comma else text

    main = one(value) + ("%" if percent and value is not None and np.isfinite(value) else "")
    if low is None or high is None or not (np.isfinite(low) and np.isfinite(high)):
        return main
    return f"{main} [{one(low)}; {one(high)}]"


__all__ = [
    "BootstrapSamples",
    "CI_COLUMNS",
    "DEFAULT_CI_METRICS",
    "LOWER_IS_BETTER",
    "PAIRED_COLUMNS",
    "PROBA_METRICS",
    "PROPORTION_METRICS",
    "bootstrap_auc",
    "bootstrap_ci",
    "bootstrap_ci_rules",
    "bootstrap_samples",
    "calibrate_on_val",
    "calibration_summary",
    "calibration_error",
    "check_proba",
    "ci_to_dict",
    "clopper_pearson",
    "crossfit_temperature",
    "ece_classwise",
    "ece_consistency",
    "ece_top_label",
    "format_ci",
    "paired_bootstrap_diff",
    "paired_bootstrap_diffs",
    "point_metrics",
    "reliability_table",
    "resample_weights",
    "stratified_folds",
]
