"""
Reguły decyzyjne na prawdopodobieństwach ESI — warstwa po blendzie i temperaturze.

Spec reguły (słownik; DOKŁADNIE ten format trafia do backendu, `ensemble_weights.json`):

    {"rule": "argmax"}                       najbardziej prawdopodobna klasa
    {"rule": "cost"}                         decyzja Bayesa z macierzą kosztów repo (k = 3)
    {"rule": "target", "cutpoints": [c1, c2, c3, c4], "safety_q": q | None}
        B1: oczekiwany poziom EL = Σ k·p_k (k = 0…4) dzielony 4 progami — klasa = liczba progów
        ostro przekroczonych. Próg ESI2/ESI3 (c2) wynika z celu undertriage T, pozostałe trzy
        maksymalizują QWK. B2 (opcjonalnie): P(ESI1) + P(ESI2) ≥ q → najwyżej ESI3.
    {"rule": "bayes_target", "lam": λ}
        B4: decyzja Bayesa ze stratą (p − t)² + λ·1[t ∈ ESI1–2, p ∈ ESI3–5]; λ dobrane do celu T.

Dopasowanie (`fit_target_rule`, `fit_safety_net`, `fit_bayes_rule`) WYŁĄCZNIE na val —
domyślnie w podgrupie vitals_complete == 1 (`mask`), populacji najbliższej TOPSOR — a test
oceniany raz, tymi samymi parametrami. Uczciwe liczby na samym val: `crossfit_rule`
(reguła dopasowana na pozostałych foldach val, stosowana do bieżącego).

Obliczenia EL i siatki bezpieczeństwa są jawnie elementowe (bez BLAS), więc wynik jest bit
w bit taki sam dla macierzy predykcji i dla pojedynczego wektora w backendzie.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    cost_matrix,
    decide,
    expected_cost,
    overtriage_rate,
    quadratic_weighted_kappa,
)
from src.evaluation.statistics import stratified_folds
from src.utils.config import (
    DECISION_CUTPOINT_GRID,
    DECISION_FIT_POPULATION,
    DECISION_GUARANTEE_ALPHA,
    DECISION_TARGET_UNDERTRIAGE,
    HIGH_ACUITY_CLASSES,
    N_CLASSES,
    RANDOM_SEED,
    SAFETY_NET_FIT_POPULATION,
    SAFETY_NET_MAX_CRITICAL_MISS,
    SAFETY_NET_MIN_Q,
    TRADEOFF_COST_MULTIPLIERS,
    TRADEOFF_TARGETS,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────
# Stałe
# ─────────────────────────────────────────
RULES: tuple[str, ...] = ("argmax", "cost", "target", "bayes_target")
SPEC_KEYS: frozenset[str] = frozenset({"rule", "cutpoints", "safety_q", "lam"})
# Parametry dozwolone dla danej reguły (inne klucze = błąd, jak w backendzie)
RULE_PARAMS: dict[str, frozenset[str]] = {
    "argmax": frozenset(),
    "cost": frozenset(),
    "target": frozenset({"cutpoints", "safety_q"}),
    "bayes_target": frozenset({"lam"}),
}
N_CUTPOINTS: int = N_CLASSES - 1
URGENT_MAX: int = max(HIGH_ACUITY_CLASSES)    # 1 = ESI2; predykcja > URGENT_MAX = ESI3–5
SAFETY_NET_CAP: int = URGENT_MAX + 1          # siatka bezpieczeństwa: najwyżej ESI3 (indeks 2)
CRITICAL_FROM: int = 3                        # critical miss = ESI1 przewidziane jako ESI4–5
FIT_POPULATIONS: tuple[str, ...] = ("vc", "all")
_LAM_MIN: float = 1e-6                        # λ > 0 (wymóg formatu backendu)
_LAM_MAX: float = 1e4


# ─────────────────────────────────────────
# Stosowanie reguł
# ─────────────────────────────────────────
def expected_level(proba: np.ndarray) -> np.ndarray:
    """
    Oczekiwany poziom EL = proba @ [0, 1, 2, 3, 4] (0 = ESI 1, …, 4 = ESI 5).

    Liczone jawnie, kolumna po kolumnie — ten sam wynik (bit w bit) dla macierzy (n, 5)
    i dla pojedynczego wektora (backend); mnożenie macierzy przez BLAS może sumować inaczej.
    """
    proba = np.asarray(proba, dtype=np.float64)
    el = np.zeros(proba.shape[:-1])
    for k in range(1, proba.shape[-1]):
        el = el + k * proba[..., k]
    return el


def _as_cutpoints(cutpoints: Any) -> np.ndarray:
    """4 skończone, niemalejące progi → tablica float64 (inaczej ValueError)."""
    if isinstance(cutpoints, (str, bytes, dict)) or not isinstance(cutpoints, (Sequence, np.ndarray)):
        raise ValueError(f"Progi (cutpoints) muszą być listą {N_CUTPOINTS} liczb, podano: {cutpoints!r}")
    values = list(cutpoints)
    if len(values) != N_CUTPOINTS or any(isinstance(v, (bool, np.bool_)) for v in values):
        raise ValueError(f"Progi (cutpoints) muszą być listą {N_CUTPOINTS} liczb, podano: {cutpoints!r}")
    try:
        cuts = np.array([float(v) for v in values], dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"Progi (cutpoints) muszą być liczbami, podano: {cutpoints!r}") from None
    if not np.isfinite(cuts).all():
        raise ValueError(f"Progi (cutpoints) muszą być skończone, podano: {cutpoints!r}")
    if (np.diff(cuts) < 0).any():
        raise ValueError(f"Progi (cutpoints) muszą być posortowane rosnąco, podano: {cutpoints!r}")
    return cuts


def apply_cutpoints(el: np.ndarray, cutpoints: Sequence[float]) -> np.ndarray:
    """Klasa = liczba progów OSTRO przekroczonych przez EL (EL równe progowi → klasa niższa)."""
    cuts = _as_cutpoints(cutpoints)
    el = np.asarray(el, dtype=np.float64)
    return (el[..., None] > cuts).sum(axis=-1).astype(int)


def apply_safety_net(pred: np.ndarray, proba: np.ndarray, q: float | None) -> np.ndarray:
    """B2: P(ESI1) + P(ESI2) ≥ q → najwyżej ESI3 (indeks ≤ 2). q = None → bez zmian."""
    pred = np.asarray(pred).astype(int)
    if q is None:
        return pred
    proba = np.asarray(proba, dtype=np.float64)
    p_high = proba[..., 0] + proba[..., 1]
    return np.where(p_high >= q, np.minimum(pred, SAFETY_NET_CAP), pred)


def bayes_cost_matrix(lam: float, n_classes: int = N_CLASSES) -> np.ndarray:
    """B4: koszt[t, p] = (p − t)² + λ·1[t ∈ ESI1–2 i p ∈ ESI3–5]."""
    t = np.arange(n_classes)[:, None]
    p = np.arange(n_classes)[None, :]
    return (p - t).astype(float) ** 2 + float(lam) * ((t <= URGENT_MAX) & (p > URGENT_MAX))


def _number(value: Any, name: str) -> float:
    """Skończona liczba (bool odrzucany) albo ValueError."""
    if isinstance(value, (bool, np.bool_)) or value is None:
        raise ValueError(f"'{name}' musi być liczbą, podano: {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' musi być liczbą, podano: {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"'{name}' musi być skończone, podano: {value!r}")
    return out


def normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """
    Sprawdza spec reguły (te same zasady co backend) i zwraca jego czystą kopię JSON.

    Błędy (ValueError): nieznany klucz albo reguła, brak wymaganego parametru, parametr innej
    reguły (np. 'lam' przy 'target'), progi nie 4 / nieposortowane / nieskończone,
    safety_q poza (0, 1], λ ≤ 0.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"Spec reguły musi być słownikiem, podano: {spec!r}")
    unknown = sorted(set(spec) - SPEC_KEYS)
    if unknown:
        raise ValueError(f"Nieznane klucze spec reguły: {unknown} (dozwolone: {sorted(SPEC_KEYS)})")
    rule = spec.get("rule", "argmax")
    if rule not in RULES:
        raise ValueError(f"Nieznana reguła decyzyjna: {rule!r} (dozwolone: {RULES})")
    foreign = sorted(set(spec) - {"rule"} - RULE_PARAMS[rule])
    if foreign:
        raise ValueError(f"Reguła {rule!r} nie przyjmuje parametrów {foreign}")
    out: dict[str, Any] = {"rule": rule}
    if rule == "target":
        if "cutpoints" not in spec:
            raise ValueError("Reguła 'target' wymaga klucza 'cutpoints' (4 progi EL)")
        out["cutpoints"] = [float(c) for c in _as_cutpoints(spec["cutpoints"])]
        q = spec.get("safety_q")
        if q is not None:
            q = _number(q, "safety_q")
            if not 0.0 < q <= 1.0:
                raise ValueError(f"'safety_q' musi należeć do (0, 1], podano: {q!r}")
        out["safety_q"] = q
    elif rule == "bayes_target":
        if "lam" not in spec:
            raise ValueError("Reguła 'bayes_target' wymaga klucza 'lam' (λ > 0)")
        lam = _number(spec["lam"], "lam")
        if lam <= 0:
            raise ValueError(f"'lam' musi być > 0, podano: {lam!r}")
        out["lam"] = lam
    return out


def decide_rule(proba: np.ndarray, spec: dict[str, Any] | TargetRule | BayesRule) -> np.ndarray:
    """
    Jedno wejście dla wszystkich reguł: prawdopodobieństwa (n, 5) + spec → klasy 0–4.

    `spec` = słownik w formacie backendu (patrz docstring modułu) albo dopasowana reguła
    (TargetRule / BayesRule — używany jest jej `spec()`).
    """
    if isinstance(spec, (TargetRule, BayesRule)):
        spec = spec.spec()
    s = normalize_spec(spec)
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != N_CLASSES:
        raise ValueError(f"Oczekiwano prawdopodobieństw o kształcie (n, {N_CLASSES}), są {proba.shape}")
    rule = s["rule"]
    if rule in ("argmax", "cost"):
        return decide(proba, rule)
    if rule == "bayes_target":
        return decide(proba, "cost", costs=bayes_cost_matrix(s["lam"]))
    pred = apply_cutpoints(expected_level(proba), s["cutpoints"])
    return apply_safety_net(pred, proba, s["safety_q"])


# ─────────────────────────────────────────
# Dopasowane reguły (JSON-owe)
# ─────────────────────────────────────────
def _json_safe(value: Any) -> Any:
    """numpy → typy Pythona, NaN/inf → None (JSON bez NaN)."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


@dataclass(frozen=True)
class TargetRule:
    """
    Reguła B1 (+ B2): progi EL dobrane do celu undertriage (+ opcjonalna siatka bezpieczeństwa).

    achieved : metryki na danych dopasowania (undertriage z górną granicą Clopper–Pearsona,
               critical miss, QWK, …) — opis, nie gwarancja dla nowych danych.
    """

    cutpoints: tuple[float, ...]
    target: float
    guarantee: bool = False
    safety_q: float | None = None
    fit_n: int = 0
    achieved: dict[str, Any] = field(default_factory=dict)
    alpha: float = DECISION_GUARANTEE_ALPHA

    def spec(self) -> dict[str, Any]:
        """Spec w formacie backendu: {"rule": "target", "cutpoints": [...], "safety_q": q|None}."""
        return {"rule": "target", "cutpoints": [float(c) for c in self.cutpoints],
                "safety_q": None if self.safety_q is None else float(self.safety_q)}

    def predict(self, proba: np.ndarray) -> np.ndarray:
        return decide_rule(proba, self.spec())

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({**self.spec(), "target": self.target, "guarantee": self.guarantee,
                           "alpha": self.alpha, "fit_n": self.fit_n, "achieved": self.achieved})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetRule:
        if data.get("rule", "target") != "target":
            raise ValueError(f"To nie jest reguła 'target': {data.get('rule')!r}")
        spec = normalize_spec({"rule": "target", "cutpoints": data["cutpoints"],
                               "safety_q": data.get("safety_q")})
        return cls(cutpoints=tuple(spec["cutpoints"]), target=float(data["target"]),
                   guarantee=bool(data.get("guarantee", False)), safety_q=spec["safety_q"],
                   fit_n=int(data.get("fit_n", 0)), achieved=dict(data.get("achieved") or {}),
                   alpha=float(data.get("alpha", DECISION_GUARANTEE_ALPHA)))


@dataclass(frozen=True)
class BayesRule:
    """Reguła B4: decyzja Bayesa z macierzą `bayes_cost_matrix(lam)`, λ dobrane do celu T."""

    lam: float
    costs: tuple[tuple[float, ...], ...]
    target: float
    achieved: dict[str, Any] = field(default_factory=dict)
    fit_n: int = 0

    def spec(self) -> dict[str, Any]:
        """Spec w formacie backendu: {"rule": "bayes_target", "lam": λ}."""
        return {"rule": "bayes_target", "lam": float(self.lam)}

    def predict(self, proba: np.ndarray) -> np.ndarray:
        return decide(np.asarray(proba, dtype=np.float64), "cost", costs=np.asarray(self.costs, dtype=float))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({**self.spec(), "costs": [list(r) for r in self.costs], "target": self.target,
                           "fit_n": self.fit_n, "achieved": self.achieved})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BayesRule:
        if data.get("rule", "bayes_target") != "bayes_target":
            raise ValueError(f"To nie jest reguła 'bayes_target': {data.get('rule')!r}")
        lam = normalize_spec({"rule": "bayes_target", "lam": data["lam"]})["lam"]
        return cls(lam=lam, costs=_costs_tuple(bayes_cost_matrix(lam)), target=float(data["target"]),
                   achieved=dict(data.get("achieved") or {}), fit_n=int(data.get("fit_n", 0)))


def _costs_tuple(costs: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(v) for v in row) for row in costs)


def rule_from_dict(data: dict[str, Any]) -> TargetRule | BayesRule:
    """`to_dict()` dowolnej dopasowanej reguły → obiekt (po kluczu 'rule')."""
    rule = data.get("rule")
    if rule == "target":
        return TargetRule.from_dict(data)
    if rule == "bayes_target":
        return BayesRule.from_dict(data)
    raise ValueError(f"Nieznany typ dopasowanej reguły: {rule!r} (target | bayes_target)")


# ─────────────────────────────────────────
# Wspólne narzędzia dopasowania
# ─────────────────────────────────────────
def population_mask(vitals_complete: np.ndarray, population: str = DECISION_FIT_POPULATION) -> np.ndarray | None:
    """Maska wierszy dopasowania: 'vc' → vitals_complete == 1, 'all' → None (wszystkie)."""
    if population == "all":
        return None
    if population == "vc":
        return np.asarray(vitals_complete).astype(bool)
    raise ValueError(f"Nieznana populacja dopasowania: {population!r} (dozwolone: {FIT_POPULATIONS})")


def _check_inputs(y: np.ndarray, proba: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y, proba, maska bool) całych danych po sprawdzeniu kształtów."""
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != N_CLASSES or len(proba) != len(y):
        raise ValueError(f"Oczekiwano y (n,) i prawdopodobieństw (n, {N_CLASSES}); są {y.shape} i {proba.shape}")
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    else:
        mask = np.asarray(mask).astype(bool)
        if mask.shape != y.shape:
            raise ValueError(f"Maska ma kształt {mask.shape}, a y {y.shape}")
    if not mask.any():
        raise ValueError("Maska nie wybiera żadnego wiersza do dopasowania reguły")
    return y, proba, mask


def _check_rate(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"'{name}' musi należeć do (0, 1), podano: {value!r}")
    return value


def _cp_upper(k: int, n: int, alpha: float) -> float:
    """Jednostronna górna granica Clopper–Pearsona (poziom 1 − alpha) dla odsetka k/n."""
    from scipy.stats import beta

    if n <= 0:
        return float("nan")
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def _achieved(y: np.ndarray, pred: np.ndarray, alpha: float) -> dict[str, Any]:
    """Metryki reguły na danych dopasowania (JSON-owe)."""
    high = np.isin(y, HIGH_ACUITY_CLASSES)
    n_high = int(high.sum())
    n_under = int((high & (pred > URGENT_MAX)).sum())
    esi1 = y == 0
    n_esi1 = int(esi1.sum())
    n_crit = int((esi1 & (pred >= CRITICAL_FROM)).sum())
    return _json_safe({
        "n": len(y),
        "qwk": quadratic_weighted_kappa(y, pred) if len(np.unique(y)) > 1 else float("nan"),
        "undertriage": n_under / n_high if n_high else float("nan"),
        "undertriage_upper": _cp_upper(n_under, n_high, alpha),
        "n_high": n_high,
        "n_undertriaged": n_under,
        "critical_miss": n_crit / n_esi1 if n_esi1 else float("nan"),
        "n_esi1": n_esi1,
        "n_critical_miss": n_crit,
        "overtriage": overtriage_rate(y, pred),
        "accuracy": float((y == pred).mean()),
    })


# ─────────────────────────────────────────
# B1 / B6 — progi EL z celem undertriage
# ─────────────────────────────────────────
def _allowed_undertriaged(n_high: int, target: float, guarantee: bool, alpha: float) -> int:
    """
    Największa liczba niedotriażowanych k spośród n_high pacjentów ESI1–2, przy której
    k / n_high ≤ T (średnio) albo — guarantee — górna granica CP(k, n_high) ≤ T.
    """
    k = min(int(math.floor(target * n_high + 1e-9)), n_high - 1)
    if not guarantee:
        return max(k, 0)
    if _cp_upper(0, n_high, alpha) > target:
        log.warning(f"Gwarancja niemożliwa: przy {n_high} pacjentach ESI1–2 nawet 0 niedotriażowań "
                    f"daje górną granicę CP {_cp_upper(0, n_high, alpha):.3f} > T = {target:.3f}")
        return 0
    lo, hi = 0, max(k, 0)                     # upper(lo) ≤ T; szukamy największego takiego k
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _cp_upper(mid, n_high, alpha) <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _target_cutpoint(el_high: np.ndarray, k: int) -> float:
    """Próg c: co najwyżej k wartości EL pacjentów ESI1–2 jest > c (środek przerwy, bez remisów)."""
    s = np.sort(el_high)
    n = len(s)
    while k > 0 and s[n - k - 1] >= s[n - k]:   # remis na granicy — próg nie rozdzieli równych EL
        k -= 1
    if k == 0:
        return float(s[-1])
    a, b = s[n - k - 1], s[n - k]
    return float(a + (b - a) / 2)


def _qwk_batch(confusions: np.ndarray) -> np.ndarray:
    """QWK dla stosu macierzy pomyłek (m, K, K) [prawda, predykcja] — jak cohen_kappa_score."""
    k = confusions.shape[-1]
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2.0
    n = confusions.sum(axis=(1, 2))
    rows = confusions.sum(axis=2)
    cols = confusions.sum(axis=1)
    expected = rows[:, :, None] * cols[:, None, :] / n[:, None, None]
    num = (w * confusions).sum(axis=(1, 2))
    den = (w * expected).sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 - num / den
    return np.where(den > 0, out, -np.inf)


class _CutpointSearch:
    """Przeszukiwanie po współrzędnych progów EL maksymalizujące QWK (siatka kwantyli EL)."""

    def __init__(self, el: np.ndarray, y: np.ndarray, n_grid: int) -> None:
        order = np.argsort(el, kind="stable")
        self.el = el[order]
        n = len(self.el)
        onehot = np.zeros((n, N_CLASSES))
        onehot[np.arange(n), y[order]] = 1.0
        # cum[i, t] = liczba wierszy klasy t wśród i najmniejszych EL
        self.cum = np.vstack([np.zeros((1, N_CLASSES)), np.cumsum(onehot, axis=0)])
        self.n = n
        if n > 1:
            pos = np.unique(np.linspace(1, n - 1, max(int(n_grid), 2)).round().astype(int))
            pos = pos[self.el[pos - 1] < self.el[pos]]
            self.cand = self.el[pos - 1] + (self.el[pos] - self.el[pos - 1]) / 2
        else:
            self.cand = np.empty(0)
        self.cand_idx = self._index(self.cand)

    def _index(self, values: np.ndarray) -> np.ndarray:
        """Liczba wierszy z EL ≤ wartość (pozycja progu w posortowanym EL)."""
        return np.searchsorted(self.el, values, side="right")

    def qwk(self, idx: np.ndarray) -> np.ndarray:
        """QWK dla wierszy pozycji progów (m, 4)."""
        m = idx.shape[0]
        bounds = np.hstack([np.zeros((m, 1), dtype=int), idx, np.full((m, 1), self.n)])
        cols = self.cum[bounds[:, 1:]] - self.cum[bounds[:, :-1]]       # (m, predykcja, prawda)
        return _qwk_batch(cols.transpose(0, 2, 1))

    def optimize(self, cuts: np.ndarray, fixed: int, max_passes: int = 10) -> tuple[np.ndarray, float]:
        cuts = np.array(cuts, dtype=float)
        idx = self._index(cuts)
        best = float(self.qwk(idx[None])[0])
        for _ in range(max_passes):
            improved = False
            for j in range(N_CUTPOINTS):
                if j == fixed:
                    continue
                lo = cuts[j - 1] if j > 0 else -np.inf
                hi = cuts[j + 1] if j < N_CUTPOINTS - 1 else np.inf
                ok = (self.cand > lo) & (self.cand < hi)
                if not ok.any():
                    continue
                trial = np.repeat(idx[None], int(ok.sum()), axis=0)
                trial[:, j] = self.cand_idx[ok]
                values = self.qwk(trial)
                b = int(np.argmax(values))
                if values[b] > best + 1e-12:
                    best = float(values[b])
                    cuts[j] = self.cand[ok][b]
                    idx[j] = self.cand_idx[ok][b]
                    improved = True
            if not improved:
                break
        return cuts, best


def _initial_cutpoints(el: np.ndarray, y: np.ndarray, c_fixed: float) -> list[np.ndarray]:
    """Dwa starty: progi „połówkowe” (0,5 / 2,5 / 3,5) i progi odtwarzające rozkład klas."""
    def ordered(c0: float, c2: float, c3: float) -> np.ndarray:
        c0 = min(c0, c_fixed - 1e-6)
        c2 = max(c2, c_fixed + 1e-6)
        return np.array([c0, c_fixed, c2, max(c3, c2 + 1e-6)])

    share = np.cumsum(np.bincount(y, minlength=N_CLASSES)) / len(y)
    q = np.quantile(el, np.clip(share[:N_CUTPOINTS], 0.0, 1.0))
    return [ordered(0.5, 2.5, 3.5), ordered(q[0], q[2], q[3])]


def fit_target_rule(
    y: np.ndarray,
    proba: np.ndarray,
    target_undertriage: float = DECISION_TARGET_UNDERTRIAGE,
    guarantee: bool = False,
    alpha: float = DECISION_GUARANTEE_ALPHA,
    mask: np.ndarray | None = None,
    *,
    safety_net: bool = False,
    max_critical_miss: float = SAFETY_NET_MAX_CRITICAL_MISS,
    safety_population: str = SAFETY_NET_FIT_POPULATION,
    n_grid: int = DECISION_CUTPOINT_GRID,
) -> TargetRule:
    """
    B1: progi EL; próg ESI2/ESI3 ustawiony tak, żeby undertriage na wierszach dopasowania
    był ≤ T (guarantee=False: odsetek ≤ T; guarantee=True — B6: jednostronna górna granica
    Clopper–Pearsona na poziomie 1 − alpha ≤ T). Pozostałe 3 progi maksymalizują QWK
    (przeszukiwanie po współrzędnych na siatce kwantyli EL, dwa punkty startowe).

    mask : wiersze dopasowania (np. vitals_complete == 1); None = wszystkie.
    safety_net : dodatkowo dopasuj siatkę bezpieczeństwa B2 (`fit_safety_net`) na wierszach
        `safety_population`: 'all' (domyślnie: wszystkie podane wiersze — w vc jest za mało
        pacjentów ESI1) albo 'mask' (te same wiersze co progi).
    """
    target = _check_rate(target_undertriage, "target_undertriage")
    if safety_population not in ("mask", "all"):
        raise ValueError(f"safety_population musi być 'mask' albo 'all', podano: {safety_population!r}")
    y_all, p_all, fit = _check_inputs(y, proba, mask)
    yf, pf = y_all[fit], p_all[fit]
    el = expected_level(pf)
    high = np.isin(yf, HIGH_ACUITY_CLASSES)
    n_high = int(high.sum())
    if n_high == 0:
        raise ValueError("Brak pacjentów ESI1–2 w danych dopasowania — nie da się dobrać progu ESI2/ESI3")

    k = _allowed_undertriaged(n_high, target, guarantee, alpha)
    c_fixed = _target_cutpoint(el[high], k)
    search = _CutpointSearch(el, yf, n_grid)
    best_cuts, best_qwk = None, -np.inf
    for init in _initial_cutpoints(el, yf, c_fixed):
        cuts, value = search.optimize(init, fixed=1)
        if value > best_qwk + 1e-12:
            best_cuts, best_qwk = cuts, value

    rule = TargetRule(cutpoints=tuple(float(c) for c in best_cuts), target=target, guarantee=bool(guarantee),
                      fit_n=int(fit.sum()), alpha=float(alpha))
    extra: dict[str, Any] = {}
    if safety_net:
        net_rows = fit if safety_population == "mask" else np.ones_like(fit)
        base_all = rule.predict(p_all)
        q = fit_safety_net(y_all, p_all, base_all, max_critical_miss, mask=net_rows)
        rule = replace(rule, safety_q=q)
        net_all = rule.predict(p_all)
        base_fit = base_all[fit]
        on_net_rows = _achieved(y_all[net_rows], net_all[net_rows], alpha)
        extra = {
            "critical_miss_base": _achieved(yf, base_fit, alpha)["critical_miss"],
            "safety_changed": float((net_all[fit] != base_fit).mean()),
            "safety_population": safety_population,
            "max_critical_miss": float(max_critical_miss),
            # wartości na wierszach, na których dobrano q (przy 'all' — więcej pacjentów ESI1)
            "safety_fit_critical_miss": on_net_rows["critical_miss"],
            "safety_fit_n_esi1": on_net_rows["n_esi1"],
        }
    achieved = {**_achieved(yf, rule.predict(pf), alpha), **_json_safe(extra), "allowed_undertriaged": int(k)}
    if achieved["undertriage"] is not None and achieved["undertriage"] > target + 1e-12:
        log.warning(f"Reguła 'target' nie osiągnęła celu na danych dopasowania: undertriage "
                    f"{achieved['undertriage']:.4f} > T = {target:.4f} (remisy EL)")
    return replace(rule, achieved=achieved)


# ─────────────────────────────────────────
# B2 — siatka bezpieczeństwa
# ─────────────────────────────────────────
def fit_safety_net(
    y: np.ndarray,
    proba: np.ndarray,
    base_pred: np.ndarray,
    max_critical_miss: float = SAFETY_NET_MAX_CRITICAL_MISS,
    mask: np.ndarray | None = None,
    *,
    min_q: float = SAFETY_NET_MIN_Q,
) -> float | None:
    """
    B2: próg q reguły „P(ESI1) + P(ESI2) ≥ q → najwyżej ESI3” nałożonej na `base_pred`.

    Zwraca NAJWIĘKSZE q ≥ min_q, przy którym critical miss (ESI1 → ESI4–5) na wierszach
    dopasowania ≤ max_critical_miss. Gdy cel jest nieosiągalny dla q ≥ min_q — największe q
    z najmniejszym osiągalnym critical miss. None, gdy siatka nic nie zmienia: reguła bazowa
    już spełnia cel, brak pacjentów ESI1 albo żaden pominięty pacjent ESI1 nie ma
    P(ESI1–2) ≥ min_q.
    """
    if not 0.0 <= float(max_critical_miss) < 1.0:
        raise ValueError(f"'max_critical_miss' musi należeć do [0, 1), podano: {max_critical_miss!r}")
    y, proba, fit = _check_inputs(y, proba, mask)
    base_pred = np.asarray(base_pred).astype(int)
    if base_pred.shape != y.shape:
        raise ValueError(f"base_pred ma kształt {base_pred.shape}, a y {y.shape}")
    y, proba, base_pred = y[fit], proba[fit], base_pred[fit]
    n_esi1 = int((y == 0).sum())
    if n_esi1 == 0:
        log.warning("Siatka bezpieczeństwa: brak pacjentów ESI1 w danych dopasowania — pomijam")
        return None
    allowed = int(math.floor(float(max_critical_miss) * n_esi1 + 1e-9))
    missed = (y == 0) & (base_pred >= CRITICAL_FROM)
    p_missed = np.sort(proba[missed, 0] + proba[missed, 1])
    if len(p_missed) <= allowed:
        log.info(f"Siatka bezpieczeństwa niepotrzebna na danych dopasowania: critical miss "
                 f"{len(p_missed)}/{n_esi1} ≤ cel {max_critical_miss:.3%}")
        return None
    # Pominięci pacjenci z P(ESI1–2) < q zostają pominięci → q = (allowed+1)-sza najmniejsza wartość
    q = float(p_missed[allowed])
    if q >= min_q:
        return q
    reachable = p_missed[p_missed >= min_q]
    if len(reachable) == 0:
        log.warning(f"Siatka bezpieczeństwa bezużyteczna: każdy pominięty pacjent ESI1 ma P(ESI1–2) "
                    f"< {min_q} — pomijam")
        return None
    q = float(reachable[0])
    left = int((p_missed < q).sum())
    log.warning(f"Siatka bezpieczeństwa: cel critical miss {max_critical_miss:.3%} nieosiągalny dla "
                f"q ≥ {min_q}; q = {q:.4f} daje {left}/{n_esi1} (najmniej możliwe)")
    return q


# ─────────────────────────────────────────
# B4 — decyzja Bayesa z λ dobranym do celu
# ─────────────────────────────────────────
def fit_bayes_rule(
    y: np.ndarray,
    proba: np.ndarray,
    target_undertriage: float = DECISION_TARGET_UNDERTRIAGE,
    mask: np.ndarray | None = None,
) -> BayesRule:
    """
    B4: najmniejsze λ (bisekcja), przy którym decyzja Bayesa z `bayes_cost_matrix(λ)` daje
    undertriage ≤ T na wierszach dopasowania. Undertriage nie rośnie z λ (λ podnosi koszt
    wszystkich decyzji ESI3–5 o λ·P(ESI1–2), wybór wewnątrz grup się nie zmienia).
    """
    target = _check_rate(target_undertriage, "target_undertriage")
    y_all, p_all, fit = _check_inputs(y, proba, mask)
    yf, pf = y_all[fit], p_all[fit]
    high = np.isin(yf, HIGH_ACUITY_CLASSES)
    if not high.any():
        raise ValueError("Brak pacjentów ESI1–2 w danych dopasowania — nie da się dobrać λ")
    p_high = pf[high]

    def under(lam: float) -> float:
        return float((decide(p_high, "cost", costs=bayes_cost_matrix(lam)) > URGENT_MAX).mean())

    if under(_LAM_MIN) <= target:
        lam = _LAM_MIN
    else:
        lo, hi = _LAM_MIN, 1.0
        while under(hi) > target and hi < _LAM_MAX:
            lo, hi = hi, hi * 2.0
        if under(hi) > target:
            log.warning(f"Reguła 'bayes_target': cel T = {target:.3f} nieosiągalny dla λ ≤ {hi:g}")
        else:
            for _ in range(100):
                if hi - lo <= 1e-7 * hi:
                    break
                mid = (lo + hi) / 2
                if under(mid) <= target:
                    hi = mid
                else:
                    lo = mid
        lam = hi
    costs = bayes_cost_matrix(lam)
    rule = BayesRule(lam=float(lam), costs=_costs_tuple(costs), target=target, fit_n=int(fit.sum()))
    return replace(rule, achieved=_achieved(yf, rule.predict(pf), DECISION_GUARANTEE_ALPHA))


# ─────────────────────────────────────────
# Uczciwe liczby na val i krzywa kompromisu
# ─────────────────────────────────────────
def _predict_with(rule: Any, proba: np.ndarray) -> np.ndarray:
    if isinstance(rule, dict):
        return decide_rule(proba, rule)
    return np.asarray(rule.predict(proba)).astype(int)


def crossfit_rule(
    y: np.ndarray,
    proba: np.ndarray,
    fitter: Callable[..., Any],
    n_folds: int = 2,
    mask: np.ndarray | None = None,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Predykcje „cross-fitted”: reguła dopasowana na pozostałych foldach (wywołanie
    `fitter(y, proba, mask=...)` → obiekt z `.predict` albo spec), zastosowana do bieżącego.
    Foldy stratyfikowane po klasie i masce. Zwraca predykcje dla WSZYSTKICH wierszy.

    Przykład: crossfit_rule(y, p, functools.partial(fit_target_rule, target_undertriage=0.15), mask=vc)
    """
    y, proba, fit = _check_inputs(y, proba, mask)
    if n_folds < 2:
        raise ValueError(f"n_folds musi być ≥ 2, podano: {n_folds}")
    folds = stratified_folds(y * 2 + fit.astype(int), n_folds, seed)
    pred = np.empty(len(y), dtype=int)
    for f in range(n_folds):
        held = folds == f
        rule = fitter(y[~held], proba[~held], mask=None if mask is None else fit[~held])
        pred[held] = _predict_with(rule, proba[held])
    return pred


def _eval_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    out = _achieved(y, pred, DECISION_GUARANTEE_ALPHA)
    out["expected_cost"] = expected_cost(y, pred)
    return out


def tradeoff_curve(
    y: np.ndarray,
    proba: np.ndarray,
    targets: Sequence[float] = TRADEOFF_TARGETS,
    mask: np.ndarray | None = None,
    crossfit: bool = True,
    *,
    cost_multipliers: Sequence[float] = TRADEOFF_COST_MULTIPLIERS,
    guarantee: bool = False,
    max_critical_miss: float = SAFETY_NET_MAX_CRITICAL_MISS,
    safety_population: str = SAFETY_NET_FIT_POPULATION,
    n_folds: int = 2,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    C6: undertriage vs QWK na wierszach `mask` (populacja dopasowania) dla:
    argmax, reguły kosztowej z mnożnikiem k ∈ cost_multipliers (bez dopasowania),
    reguły 'target', 'target' + siatka bezpieczeństwa i 'bayes_target' dla każdego celu T.

    crossfit=True → reguły dopasowane na pozostałych foldach (uczciwe liczby na val);
    False → dopasowanie i ocena na tych samych wierszach (optymistycznie).
    Kolumny: method, target, param, crossfit, n, qwk, undertriage, undertriage_upper,
    critical_miss, overtriage, accuracy, expected_cost, liczności.
    """
    y, proba, rows = _check_inputs(y, proba, mask)
    y_eval = y[rows]
    records: list[dict[str, Any]] = []

    def add(method: str, pred: np.ndarray, target: float = float("nan"), param: float = float("nan")) -> None:
        records.append({"method": method, "target": target, "param": param, "crossfit": bool(crossfit),
                        **_eval_metrics(y_eval, np.asarray(pred)[rows])})

    add("argmax", decide(proba, "argmax"))
    for k in cost_multipliers:
        add("cost", decide(proba, "cost", costs=cost_matrix(under=float(k))), param=float(k))

    for t in targets:
        t = float(t)
        fitters: dict[str, Callable[..., Any]] = {
            "target": lambda yy, pp, mask=None, t=t: fit_target_rule(yy, pp, t, guarantee, mask=mask),
            "target_safety": lambda yy, pp, mask=None, t=t: fit_target_rule(
                yy, pp, t, guarantee, mask=mask, safety_net=True, max_critical_miss=max_critical_miss,
                safety_population=safety_population),
            "bayes_target": lambda yy, pp, mask=None, t=t: fit_bayes_rule(yy, pp, t, mask=mask),
        }
        for method, fitter in fitters.items():
            if crossfit:
                pred = crossfit_rule(y, proba, fitter, n_folds=n_folds, mask=mask, seed=seed)
                add(method, pred, target=t)
            else:
                rule = fitter(y, proba, mask=mask)
                add(method, rule.predict(proba), target=t,
                    param=float(rule.lam) if isinstance(rule, BayesRule) else float(rule.cutpoints[1]))
    return pd.DataFrame.from_records(records)


__all__ = [
    "BayesRule",
    "RULES",
    "SPEC_KEYS",
    "TargetRule",
    "apply_cutpoints",
    "apply_safety_net",
    "bayes_cost_matrix",
    "crossfit_rule",
    "decide_rule",
    "expected_level",
    "fit_bayes_rule",
    "fit_safety_net",
    "fit_target_rule",
    "normalize_spec",
    "population_mask",
    "rule_from_dict",
    "tradeoff_curve",
]
