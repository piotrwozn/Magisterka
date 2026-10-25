"""
Sekcje raportu końcowego do pracy (`scripts/08_final_report.py`): obliczenia na ZAPISANYCH
predykcjach (bez treningu) i ich tabele Markdown po polsku. Skrypt 08 tylko składa całość.

    Reguły decyzyjne     fit_rules / predict_rules — argmax, koszt, progi oczekiwanego poziomu
                         z celem undertriage T (B1, + siatka B2, opcjonalnie gwarancja B6) i decyzja
                         Bayesa z λ (B4). Dopasowanie WYŁĄCZNIE na VAL (domyślnie podgrupa vc),
                         TEST oceniany raz tymi samymi parametrami.
                         bootstrap_decisions — przedziały ufności dla DOWOLNYCH reguł (predykcje
                         podane wprost; `statistics.bootstrap_samples` zna tylko argmax i koszt).
    C6   tradeoff_section       kompromis undertriage–QWK na VAL (cross-fitting) + punkt pracy
    C2   subgroup_section       płeć, wiek, przybycie, oddział, vc: przedziały, Δ względem grupy
                                odniesienia, undertriage „zakotwiczone w przyjęciu” (model vs pielęgniarka)
    C4   admission_section, admission_subgroups, discordance_table
    C10  error_analysis         pary ESI, grupy dolegliwości, przyjęcia niedotriażowanych ESI 2
    C7   dca_label_section, dca_admission_section — net benefit (Vickers i Elkin 2006)
    C8   calibration_groups     O/E per poziom ESI; wyraz wolny i nachylenie kalibracji P(ESI 1–2)
    C13  deferral_section, conformal_section — odsyłanie niepewnych, zbiory conformal warunkowe po ESI
    C1/C3/C9  load_analyses + render_* — podsumowanie wyników runs/analyses/ (gdy są)

Każda sekcja przy braku danych (brak kolumny `admitted`, brak atrybutów pacjentów, za mała
podgrupa, brak wyników analiz) zwraca {"skipped": powód} — raport opisuje pominięcie, zamiast
przerywać.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.decision import (
    BayesRule,
    TargetRule,
    bayes_cost_matrix,
    crossfit_rule,
    decide_rule,
    expected_level,
    fit_bayes_rule,
    fit_target_rule,
    population_mask,
    tradeoff_curve,
)
from src.evaluation.metrics import cost_matrix
from src.evaluation.statistics import (
    DEFAULT_N_BINS,
    LOWER_IS_BETTER,
    PROBA_METRICS,
    BootstrapSamples,
    _Prepared,
    _validate_metrics,
    bootstrap_auc,
    clopper_pearson,
    format_ci,
    resample_weights,
)
from src.utils.config import (
    CLASS_NAMES,
    DECISION_GUARANTEE_ALPHA,
    HIGH_ACUITY_CLASSES,
    N_CLASSES,
    TRADEOFF_TARGETS,
    UNDERTRIAGE_COST_MULTIPLIER,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────
# Stałe
# ─────────────────────────────────────────
RULE_ORDER: tuple[str, ...] = ("argmax", "cost", "target", "bayes_target")
RULE_LABELS: dict[str, str] = {"argmax": "argmax", "cost": "koszt", "target": "progi T", "bayes_target": "Bayes T"}
FITTED_RULES: frozenset[str] = frozenset({"target", "bayes_target"})
URGENT_MAX: int = max(HIGH_ACUITY_CLASSES)        # 1 = ESI 2; predykcja > URGENT_MAX = ESI 3–5
NURSE = "ESI z triażu (pielęgniarka)"
MIN_SUBGROUP_N: int = 30                           # mniejsza podgrupa → wiersz pominięty z notą
MIN_FIT_HIGH: int = 5                              # min. pacjentów ESI 1–2 w populacji dopasowania
DCA_THRESHOLDS: np.ndarray = np.round(np.arange(0.02, 0.605, 0.01), 2)
DCA_TABLE_THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
DCA_DELTA_THRESHOLDS: tuple[float, ...] = (0.10, 0.20, 0.25, 0.30, 0.40)
DEFERRAL_FRACTIONS: tuple[float, ...] = (0.0, 0.10, 0.20, 0.30)
CONFORMAL_ALPHA: float = 0.10
TRADEOFF_TABLE_TARGETS: tuple[float, ...] = (0.10, 0.15, 0.20)
# Minimalna istotna różnica (MID) — ustalona z góry (rigor lens, ULEPSZENIA C1). Wyższość ogłaszamy
# tylko, gdy przedział Δ wyklucza 0 I |Δ| > MID I |Δ| > 2 × SD seedów. Zmień PRZED oceną testu.
MINIMAL_IMPORTANT_DIFFERENCE: dict[str, float] = {
    "quadratic_weighted_kappa": 0.01,
    "undertriage_rate": 0.01,
    "auc_macro": 0.005,
    "log_loss": 0.005,
}
SEED_SD_FACTOR: float = 2.0
# metryka 08 → klucz metryki w runs/analyses/seeds__*/results.json (reguła dopisywana jako _<reguła>)
SEED_METRIC_KEYS: dict[str, str] = {"quadratic_weighted_kappa": "qwk", "undertriage_rate": "undertriage",
                                    "critical_miss_rate": "critical_miss", "recall_esi1": "recall_esi1"}
SEED_PROBA_KEYS: dict[str, str] = {"log_loss": "log_loss", "auc_macro": "auc_macro"}
SEED_RULES: frozenset[str] = frozenset({"argmax", "cost", "target"})
AGE_BANDS: tuple[tuple[float, float, str], ...] = (
    (-math.inf, 40.0, "18–39"), (40.0, 65.0, "40–64"), (65.0, 80.0, "65–79"), (80.0, math.inf, "80+"))
ARRIVAL_GROUPS: tuple[tuple[str, str], ...] = (
    ("arrival_ambulance", "karetka"), ("arrival_self", "samodzielnie"), ("arrival_other", "inny"))
CC_GROUP_LABELS: dict[str, str] = {
    "cc_group_cardiac": "kardiologiczne",
    "cc_group_respiratory": "oddechowe",
    "cc_group_neuro": "neurologiczne",
    "cc_group_abdominal": "brzuszne",
    "cc_group_trauma": "urazy",
    "cc_group_psych_tox": "psychiatryczne / zatrucia",
    "cc_group_low_acuity": "drobne (niska pilność)",
}
ATTRIBUTE_COLUMNS: tuple[str, ...] = ("age", "is_male", *(c for c, _ in ARRIVAL_GROUPS), *CC_GROUP_LABELS)

# Metryki raportowane procentowo (odsetki pacjentów)
PERCENT_METRICS = frozenset({
    "accuracy", "within_one_accuracy", "undertriage_rate", "critical_miss_rate", "overtriage_rate",
    "mistriage_under", "mistriage_over", "recall_esi1", "recall_esi12",
})
# Odsetki z małym mianownikiem (klasy ESI) — w tabelach także liczebność k/n
COUNT_METRICS = frozenset({"undertriage_rate", "critical_miss_rate", "overtriage_rate", "recall_esi1",
                           "recall_esi12"})
SUBGROUP_METRICS: list[str] = ["quadratic_weighted_kappa", "undertriage_rate", "overtriage_rate",
                               "critical_miss_rate"]


# ─────────────────────────────────────────
# Formatowanie (report.md)
# ─────────────────────────────────────────
def fnum(value: Any) -> float | None:
    """Liczba skończona albo None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def int_text(value: float) -> str:
    """Liczba całkowita z odstępem tysięcy: 12 345."""
    return f"{int(round(value)):,}".replace(",", " ")


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def jsonable(obj: Any) -> Any:
    """Rekurencyjnie: numpy → Python, NaN/inf → None, DataFrame → lista rekordów."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


@dataclass
class Fmt:
    """Formatowanie liczb w report.md (opcjonalnie przecinek dziesiętny)."""

    decimal_comma: bool = False
    alpha: float = 0.05

    def _dc(self, text: str) -> str:
        return text.replace(".", ",") if self.decimal_comma else text

    def ci(self, metric: str, row: dict[str, Any] | pd.Series | None) -> str:
        """'12.3% [11.0; 13.6]'; odsetki z małym mianownikiem dodatkowo z liczebnością: '0.0% [0.0; 9.3] (0/38)'."""
        if row is None:
            return "—"
        text = format_ci(fnum(row["value"]), fnum(row["ci_low"]), fnum(row["ci_high"]), digits=3,
                         percent=metric in PERCENT_METRICS, decimal_comma=self.decimal_comma)
        k, n = fnum(row.get("k")), fnum(row.get("n"))
        if metric in COUNT_METRICS and k is not None and n is not None:
            text += f" ({int_text(k)}/{int_text(n)})"
        return text

    def num(self, value: Any, digits: int = 3) -> str:
        v = fnum(value)
        return "—" if v is None else self._dc(f"{v:.{digits}f}")

    def signed(self, value: Any, digits: int = 3) -> str:
        v = fnum(value)
        return "—" if v is None else self._dc(f"{v:+.{digits}f}")

    def pct(self, value: Any, digits: int = 1) -> str:
        v = fnum(value)
        return "—" if v is None else self._dc(f"{100 * v:.{digits}f}%")

    def pct_g(self, value: Any) -> str:
        """Odsetek bez zbędnych zer: 0.15 → '15%', 0.075 → '7.5%' (cele T)."""
        v = fnum(value)
        return "—" if v is None else self._dc(f"{round(100 * v, 4):g}%")

    def num_ci(self, value: Any, low: Any, high: Any, digits: int = 3) -> str:
        return format_ci(fnum(value), fnum(low), fnum(high), digits=digits, decimal_comma=self.decimal_comma)

    def pct_ci(self, value: Any, low: Any, high: Any) -> str:
        return format_ci(fnum(value), fnum(low), fnum(high), digits=3, percent=True,
                         decimal_comma=self.decimal_comma)

    def rate(self, k: Any, n: Any) -> str:
        """Odsetek k/n z dokładnym przedziałem Clopper–Pearsona i liczebnością; n = 0 → '—'."""
        k, n = fnum(k), fnum(n)
        if k is None or n is None or n <= 0:
            return "—"
        low, high = clopper_pearson(int(round(k)), int(round(n)), self.alpha)
        return f"{self.pct_ci(k / n, low, high)} ({int_text(k)}/{int_text(n)})"

    def pp(self, d: Any, low: Any = None, high: Any = None) -> str:
        """Różnica odsetków w punktach procentowych: '+3.2 pp [+1.0; +5.4]'."""
        d, low, high = fnum(d), fnum(low), fnum(high)
        if d is None:
            return "—"
        text = f"{100 * d:+.1f} pp"
        if low is not None and high is not None:
            text += f" [{100 * low:+.1f}; {100 * high:+.1f}]"
        return self._dc(text)

    def delta(self, metric: str, d: Any, low: Any, high: Any) -> str:
        d, low, high = fnum(d), fnum(low), fnum(high)
        if d is None:
            return "—"
        if metric in PERCENT_METRICS:
            return self.pp(d, low, high)
        text = f"{d:+.4f}"
        if low is not None and high is not None:
            text += f" [{low:+.4f}; {high:+.4f}]"
        return self._dc(text)

    def p(self, value: Any) -> str:
        v = fnum(value)
        if v is None:
            return "—"
        return self._dc("< 0.001" if v < 0.001 else f"{v:.3f}")


def rule_label(rule: str) -> str:
    return RULE_LABELS.get(rule, rule)


# ─────────────────────────────────────────
# Reguły decyzyjne: dopasowanie na VAL
# ─────────────────────────────────────────
@dataclass
class DecisionOptions:
    """Ustawienia warstwy decyzyjnej raportu (z linii poleceń 08)."""

    main_rule: str = "target"
    rules: list[str] = field(default_factory=lambda: list(RULE_ORDER))
    target: float = 0.15
    guarantee: bool = False
    safety_net: bool = True
    population: str = "vc"

    def describe(self) -> str:
        parts = [f"T = {round(100 * self.target, 4):g}%"]
        if self.guarantee:
            parts.append(f"gwarancja: górna granica Clopper–Pearsona ≤ T (α = {DECISION_GUARANTEE_ALPHA:g})")
        parts.append("siatka bezpieczeństwa B2" if self.safety_net else "bez siatki bezpieczeństwa")
        return ", ".join(parts)


@dataclass
class FittedRules:
    """Reguły dopasowane na VAL jednego zestawu prawdopodobieństw (połączenie albo model)."""

    specs: dict[str, dict[str, Any]]
    objects: dict[str, TargetRule | BayesRule]
    population: str
    mask: np.ndarray | None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return jsonable({"population": self.population, "note": self.note, "specs": self.specs,
                         "fitted": {r: o.to_dict() for r, o in self.objects.items()}})


def fit_mask(y: np.ndarray, vc: np.ndarray | None, population: str) -> tuple[np.ndarray | None, str, str | None]:
    """
    Maska wierszy dopasowania reguł: ('vc' | 'all'). Brak kolumny vc albo za mało pacjentów
    ESI 1–2 w vc → cały VAL z notą. Zwraca (maska | None, populacja efektywna, nota | None).
    """
    if population == "all":
        return None, "all", None
    if vc is None:
        return None, "all", "brak kolumny vitals_complete — reguły dopasowane na całym VAL"
    mask = population_mask(vc, population)
    n_high = int(np.isin(np.asarray(y)[mask], HIGH_ACUITY_CLASSES).sum())
    if n_high < MIN_FIT_HIGH:
        return None, "all", (f"w podgrupie vc tylko {n_high} pacjentów ESI 1–2 (< {MIN_FIT_HIGH}) — reguły "
                             f"dopasowane na całym VAL")
    return mask, population, None


def fit_rules(y: np.ndarray, proba: np.ndarray, vc: np.ndarray | None, opts: DecisionOptions) -> FittedRules:
    """
    Dopasowanie reguł z `opts.rules` na VAL (te prawdopodobieństwa, które trafią do backendu:
    po łączeniu i temperaturze). argmax i koszt nie mają parametrów. Błąd dopasowania → ValueError.
    """
    mask, population, note = fit_mask(y, vc, opts.population)
    specs: dict[str, dict[str, Any]] = {}
    objects: dict[str, TargetRule | BayesRule] = {}
    for rule in opts.rules:
        if rule == "target":
            obj = fit_target_rule(y, proba, opts.target, opts.guarantee, DECISION_GUARANTEE_ALPHA, mask=mask,
                                  safety_net=opts.safety_net)
        elif rule == "bayes_target":
            obj = fit_bayes_rule(y, proba, opts.target, mask=mask)
        else:
            specs[rule] = {"rule": rule}
            continue
        objects[rule] = obj
        specs[rule] = obj.spec()
    return FittedRules(specs=specs, objects=objects, population=population, mask=mask, note=note)


def rule_fitter(rule: str, opts: DecisionOptions) -> Callable[..., Any]:
    """Funkcja dopasowania reguły dla `crossfit_rule` (fitter(y, proba, mask=...))."""
    if rule == "target":
        return lambda yy, pp, mask=None: fit_target_rule(yy, pp, opts.target, opts.guarantee,
                                                         DECISION_GUARANTEE_ALPHA, mask=mask,
                                                         safety_net=opts.safety_net)
    if rule == "bayes_target":
        return lambda yy, pp, mask=None: fit_bayes_rule(yy, pp, opts.target, mask=mask)
    raise ValueError(f"Reguła '{rule}' nie ma parametrów do dopasowania")


def predict_rules(proba: np.ndarray, fitted: FittedRules) -> dict[str, np.ndarray]:
    """{reguła: klasy 0–4} dla wszystkich reguł dopasowanych (spec w formacie backendu)."""
    return {rule: decide_rule(proba, spec) for rule, spec in fitted.specs.items()}


def decision_counts(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """QWK i odsetki (z licznikami) jednej decyzji — JSON-owe."""
    from src.evaluation.metrics import quadratic_weighted_kappa

    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    high = np.isin(y, HIGH_ACUITY_CLASSES)
    low = y >= 3
    esi1 = y == 0
    out = {
        "n": int(len(y)),
        "qwk": float(quadratic_weighted_kappa(y, pred)) if len(np.unique(y)) > 1 and len(y) > 1 else None,
        "accuracy": float((y == pred).mean()) if len(y) else None,
        "k_under": int((high & (pred > URGENT_MAX)).sum()), "n_high": int(high.sum()),
        "k_crit": int((esi1 & (pred >= 3)).sum()), "n_esi1": int(esi1.sum()),
        "k_over": int((low & (pred <= URGENT_MAX)).sum()), "n_low": int(low.sum()),
    }
    out["undertriage"] = out["k_under"] / out["n_high"] if out["n_high"] else None
    out["critical_miss"] = out["k_crit"] / out["n_esi1"] if out["n_esi1"] else None
    out["overtriage"] = out["k_over"] / out["n_low"] if out["n_low"] else None
    return out


def val_rule_summary(
    y: np.ndarray, proba: np.ndarray, fitted: FittedRules, opts: DecisionOptions, seed: int = 42, n_folds: int = 2,
) -> dict[str, Any]:
    """
    Wynik reguł na VAL: argmax / koszt wprost; reguły dopasowywane — cross-fitting (reguła z innego
    foldu VAL), czyli uczciwe przybliżenie wyniku na nowych danych. Także na populacji dopasowania.
    """
    out: dict[str, Any] = {}
    for rule, spec in fitted.specs.items():
        if rule in FITTED_RULES:
            try:
                pred = crossfit_rule(y, proba, rule_fitter(rule, opts), n_folds=n_folds, mask=fitted.mask, seed=seed)
                kind = f"cross-fit ({n_folds} foldy)"
            except ValueError as exc:            # za mało pacjentów w foldzie — wynik w próbie
                log.warning(f"Cross-fitting reguły {rule} na VAL niemożliwy ({exc}) — wynik w próbie")
                pred = decide_rule(proba, spec)
                kind = "w próbie (optymistycznie)"
        else:
            pred = decide_rule(proba, spec)
            kind = "bez dopasowania"
        entry = {"kind": kind, "all": decision_counts(y, pred)}
        if fitted.mask is not None:
            entry["fit_population"] = decision_counts(y[fitted.mask], pred[fitted.mask])
        out[rule] = entry
    return out


# ─────────────────────────────────────────
# Bootstrap dla dowolnych reguł (predykcje podane wprost)
# ─────────────────────────────────────────
class _PreparedDecisions(_Prepared):
    """`statistics._Prepared` z macierzą pomyłek z PODANYCH predykcji (reguła = etykieta)."""

    def __init__(self, y: np.ndarray, proba: np.ndarray, decisions: dict[str, np.ndarray],
                 metrics: Sequence[str], n_bins: int = DEFAULT_N_BINS) -> None:
        super().__init__(y, proba, [], metrics, n_bins)
        y = np.asarray(y).astype(int)
        self.rules = list(decisions)
        if any(m not in PROBA_METRICS for m in self.metrics):
            for rule, pred in decisions.items():
                pred = np.asarray(pred).astype(int)
                if pred.shape != y.shape:
                    raise ValueError(f"Predykcje reguły '{rule}' mają kształt {pred.shape}, etykiety {y.shape}")
                if pred.min(initial=0) < 0 or pred.max(initial=0) >= self.k:
                    raise ValueError(f"Predykcje reguły '{rule}' spoza zakresu 0…{self.k - 1}")
                onehot = np.zeros((len(y), self.k * self.k))
                onehot[np.arange(len(y)), y * self.k + pred] = 1.0
                self.cell_onehot[rule] = onehot


def bootstrap_decisions(
    y_true: np.ndarray,
    probas: dict[str, np.ndarray],
    decisions: dict[str, dict[str, np.ndarray]],
    metrics: Sequence[str] | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    stratify: bool = True,
    n_bins: int = DEFAULT_N_BINS,
) -> BootstrapSamples:
    """
    Jak `statistics.bootstrap_samples`, ale decyzje każdego modelu podane wprost:
    decisions[model][reguła] = klasy 0–4. Te same losowania dla wszystkich modeli i reguł
    (sparowane porównania przez `.paired`). Odsetki — Clopper–Pearson z k/n.
    """
    y = np.asarray(y_true).astype(int)
    if len(y) == 0:
        raise ValueError("Pusty zbiór — brak wierszy do oceny")
    if n_boot < 1 or not 0 < alpha < 1:
        raise ValueError("n_boot musi być ≥ 1, a alpha w (0, 1)")
    names = _validate_metrics(metrics)
    if not probas or set(probas) != set(decisions):
        raise ValueError("Te same modele muszą mieć prawdopodobieństwa i decyzje")
    rules = list(next(iter(decisions.values())))
    for name, d in decisions.items():
        if list(d) != rules:
            raise ValueError(f"Model '{name}' ma inne reguły ({list(d)}) niż pozostałe ({rules})")
    preps = {name: _PreparedDecisions(y, probas[name], decisions[name], names, n_bins) for name in probas}
    result = BootstrapSamples(names=list(probas), rules=rules, metrics=names, n=len(y), n_boot=int(n_boot),
                              alpha=float(alpha), seed=int(seed), stratify=bool(stratify), proportion_ci="exact")
    ones = np.ones((1, len(y)))
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


def decision_point_metrics(
    y: np.ndarray, proba: np.ndarray, pred: np.ndarray, metrics: Sequence[str] | None = None,
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, float]:
    """Wartości metryk (definicje jak w bootstrapie) dla podanych predykcji."""
    names = _validate_metrics(metrics)
    prep = _PreparedDecisions(y, proba, {"r": pred}, names, n_bins)
    values = prep.compute(np.ones((1, len(np.asarray(y)))))
    return {m: float(values[("*", m) if m in PROBA_METRICS else ("r", m)][0]) for m in names}


# ─────────────────────────────────────────
# C6 — kompromis undertriage–QWK (VAL, cross-fitting)
# ─────────────────────────────────────────
def chosen_point(opts: DecisionOptions) -> dict[str, Any]:
    """Wiersz krzywej kompromisu odpowiadający regule głównej (do wyróżnienia na wykresie)."""
    if opts.main_rule == "target":
        return {"method": "target_safety" if opts.safety_net else "target", "target": opts.target}
    if opts.main_rule == "bayes_target":
        return {"method": "bayes_target", "target": opts.target}
    if opts.main_rule == "cost":
        return {"method": "cost", "param": UNDERTRIAGE_COST_MULTIPLIER}
    return {"method": "argmax"}


def tradeoff_section(
    y: np.ndarray, proba: np.ndarray, mask: np.ndarray | None, population: str, opts: DecisionOptions,
    fig_path: Path | None = None, seed: int = 42,
) -> dict[str, Any]:
    """C6: krzywa kompromisu na VAL (reguły z celem T dopasowane cross-fittingiem) + wykres."""
    targets = sorted({round(float(t), 6) for t in (*TRADEOFF_TARGETS, opts.target)})
    try:
        table = tradeoff_curve(y, proba, targets, mask=mask, crossfit=True, guarantee=opts.guarantee, seed=seed)
    except ValueError as exc:
        return {"skipped": f"krzywa kompromisu niemożliwa: {exc}"}
    chosen = chosen_point(opts)
    figure = None
    if fig_path is not None:
        from src.evaluation.visualizations import plot_tradeoff

        pop = "podgrupa vc" if population == "vc" else "cały VAL"
        figure = plot_tradeoff(table, fig_path, chosen=chosen,
                               title=f"Kompromis niedotriażowanie – QWK na VAL ({pop}, cross-fitting)")
    return {"table": table, "chosen": chosen, "population": population,
            "figure": Path(figure).name if figure else None}


TRADEOFF_METHOD_LABELS: dict[str, str] = {
    "argmax": "argmax",
    "cost": "koszt (mnożnik k)",
    "target": "progi T (B1)",
    "target_safety": "progi T + siatka (B1 + B2)",
    "bayes_target": "Bayes T (B4)",
}


def render_tradeoff(sec: dict[str, Any], fmt: Fmt, opts: DecisionOptions) -> list[str]:
    lines = ["## Punkt pracy reguły decyzyjnej (VAL, kompromis niedotriażowanie – QWK)", ""]
    if sec.get("skipped"):
        return lines + [f"Pominięte: {sec['skipped']}.", ""]
    pop = "podgrupa vc" if sec["population"] == "vc" else "cały VAL"
    lines += [
        f"Na VAL ({pop}) każda reguła z celem T jest dopasowana na jednym foldzie i oceniona na drugim "
        f"(cross-fitting, 2 foldy), więc liczby przybliżają wynik na nowych danych. Reguła kosztowa "
        f"nie ma parametrów dopasowywanych — tylko mnożnik k. Każde zmniejszenie niedotriażowania "
        f"kosztuje QWK; wybrany punkt pracy (reguła główna `{opts.main_rule}`) jest wyróżniony.",
        "",
    ]
    table: pd.DataFrame = sec["table"]
    targets = set(TRADEOFF_TABLE_TARGETS) | {round(opts.target, 6)}
    rows = []
    chosen = sec["chosen"]
    for rec in table.to_dict(orient="records"):
        method = rec["method"]
        t = fnum(rec.get("target"))
        if method in ("target", "target_safety", "bayes_target") and (t is None or round(t, 6) not in targets):
            continue
        if method == "cost":
            param = f"k = {fnum(rec['param']):g}"
        elif method == "argmax":
            param = "—"
        else:
            param = f"T = {fmt.pct_g(t)}"
        is_chosen = method == chosen.get("method") and (
            (chosen.get("target") is None or (t is not None and abs(t - chosen["target"]) < 1e-9))
            and (chosen.get("param") is None or abs((fnum(rec.get("param")) or -1) - chosen["param"]) < 1e-9))
        label = TRADEOFF_METHOD_LABELS.get(method, method)
        rows.append([f"**{label}**" if is_chosen else label, f"**{param}**" if is_chosen else param,
                     fmt.rate(rec["n_undertriaged"], rec["n_high"]), fmt.num(rec["qwk"]),
                     fmt.rate(rec["n_critical_miss"], rec["n_esi1"]), fmt.pct(rec["overtriage"]),
                     fmt.pct(rec["accuracy"])])
    lines += [md_table(["reguła", "parametr", "niedotriażowanie ESI 1–2", "QWK", "kryt. pominięcie ESI 1",
                        "przetriażowanie", "trafność"], rows), ""]
    if sec.get("figure"):
        lines += [f"Wykres: `figures/{sec['figure']}` (wszystkie cele T: `tables/tradeoff_val.csv`).", ""]
    return lines


# ─────────────────────────────────────────
# Atrybuty pacjentów (join po row_id) i podgrupy
# ─────────────────────────────────────────
def load_attributes(
    meta: pd.DataFrame, split: str, fingerprint: str | None, pred_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame | None, str]:
    """
    Atrybuty pacjentów (wiek, płeć, sposób przybycia, grupy dolegliwości) w kolejności wierszy
    predykcji: z predykcji runu (gdy je zawierają), inaczej z przygotowanego splitu po `row_id` —
    tylko przy zgodnym odcisku danych. Zwraca (ramka | None, źródło albo powód pominięcia).
    """
    if pred_frame is not None:
        cols = [c for c in ATTRIBUTE_COLUMNS if c in pred_frame.columns]
        if cols:
            frame = pred_frame.reindex(meta.index)
            return frame[cols].copy(), "predykcje runów"
    try:
        from src.data import dataset

        current = dataset.data_fingerprint()
        if current == dataset.NO_DATA_FINGERPRINT:
            return None, "brak przygotowanych splitów (data/processed) — atrybuty pacjentów niedostępne"
        if fingerprint and current != fingerprint:
            return None, (f"przygotowane splity mają inny odcisk danych ({current}) niż runy ({fingerprint}) "
                          f"— atrybuty pacjentów pominięte (odmowa łączenia po row_id)")
        import pyarrow.parquet as pq

        available = set(pq.read_schema(dataset.SPLIT_PATHS[split]).names)
        cols = [c for c in ATTRIBUTE_COLUMNS if c in available]
        if not cols or "row_id" not in available:
            return None, f"split '{split}' nie ma kolumn atrybutów pacjentów"
        frame = dataset.load_split(split, columns=["row_id", *cols]).set_index("row_id")
    except Exception as exc:  # noqa: BLE001 — brak splitu / kolumn = sekcja bez atrybutów
        return None, f"atrybuty pacjentów niedostępne ({type(exc).__name__}: {exc})"
    missing = meta.index.difference(frame.index)
    if len(missing):
        return None, f"{len(missing):,} wierszy predykcji nie ma w splicie '{split}' — atrybuty pominięte"
    return frame.loc[meta.index, cols].copy(), f"split '{split}' (data/processed, join po row_id)"


def _flag(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").fillna(0).to_numpy() > 0.5


def subgroup_masks(meta: pd.DataFrame, attrs: pd.DataFrame | None) -> list[dict[str, Any]]:
    """
    Podgrupy: płeć, wiek (18–39 / 40–64 / 65–79 / 80+), sposób przybycia, oddział, komplet
    parametrów życiowych. Pierwsza grupa każdego atrybutu jest grupą odniesienia dla Δ.
    """
    groups: list[dict[str, Any]] = []

    def add(attribute: str, items: list[tuple[str, np.ndarray]]) -> None:
        for i, (label, mask) in enumerate(items):
            groups.append({"attribute": attribute, "group": label, "mask": np.asarray(mask, dtype=bool),
                           "reference": i == 0})

    if attrs is not None and "is_male" in attrs.columns:
        male = pd.to_numeric(attrs["is_male"], errors="coerce").to_numpy()
        add("płeć", [("kobiety", male == 0), ("mężczyźni", male == 1)])
    if attrs is not None and "age" in attrs.columns:
        age = pd.to_numeric(attrs["age"], errors="coerce").to_numpy(dtype=float)
        items = []
        for lo, hi, label in AGE_BANDS:
            if label == "18–39" and np.nanmin(age, initial=np.inf) < 18:
                label = "< 40"
            items.append((label, (age >= lo) & (age < hi)))
        add("wiek", items)
    if attrs is not None and all(c in attrs.columns for c, _ in ARRIVAL_GROUPS):
        add("sposób przybycia", [(label, _flag(attrs, col)) for col, label in ARRIVAL_GROUPS])
    if "site" in meta.columns:
        sites = meta["site"].astype("string").to_numpy()
        values = sorted(str(s) for s in pd.unique(sites[~pd.isna(sites)]))
        add("oddział", [(f"oddział {s}", sites == s) for s in values])
    if "vitals_complete" in meta.columns:
        vc = pd.to_numeric(meta["vitals_complete"], errors="coerce").to_numpy()
        add("parametry życiowe", [("komplet HR/SBP/RR (vc)", vc == 1), ("niekompletne", vc == 0)])
    return groups


def newcombe_diff(k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """
    Różnica dwóch niezależnych odsetków p1 − p2 z przedziałem Newcombe'a (hybryda Wilsona,
    metoda 10). Zwraca (Δ, dół, góra); n = 0 → NaN.
    """
    from scipy.stats import norm

    if n1 <= 0 or n2 <= 0:
        return float("nan"), float("nan"), float("nan")
    z = float(norm.ppf(1 - alpha / 2))

    def wilson(k: int, n: int) -> tuple[float, float]:
        p = k / n
        den = 1 + z * z / n
        center = (p + z * z / (2 * n)) / den
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return center - half, center + half

    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def _paired_mean_ci(d: np.ndarray, strata: np.ndarray, n_boot: int, alpha: float, seed: int) -> tuple[float, float]:
    """Percentylowy przedział bootstrap średniej różnic sparowanych (losowanie w warstwach)."""
    samples = []
    for W in resample_weights(strata, n_boot, seed, stratify=True):
        total = W.sum(axis=1)
        samples.append((W @ d) / np.where(total > 0, total, 1.0))
    s = np.concatenate(samples)
    low, high = np.percentile(s, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def subgroup_section(
    y: np.ndarray, proba: np.ndarray, pred: np.ndarray, groups: list[dict[str, Any]],
    esi: np.ndarray | None = None, admitted: np.ndarray | None = None, pred_argmax: np.ndarray | None = None,
    n_boot: int = 1000, alpha: float = 0.05, seed: int = 42,
) -> dict[str, Any]:
    """
    C2: dla każdej podgrupy — QWK (bootstrap), niedotriażowanie / przetriażowanie / krytyczne
    pominięcie (Clopper–Pearson), Δ niedotriażowania względem grupy odniesienia atrybutu
    (Newcombe) i — gdy znane przyjęcia — undertriage „zakotwiczone w przyjęciu”: odsetek
    PRZYJĘTYCH do szpitala, którym model (reguła główna) / pielęgniarka dali ESI 3–5, z sparowaną Δ.
    """
    if not groups:
        return {"skipped": "brak atrybutów podgrup (płeć, wiek, przybycie, oddział, vc)"}
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    rows: list[dict[str, Any]] = []
    for g in groups:
        mask = g["mask"]
        y_g = y[mask]
        row: dict[str, Any] = {"attribute": g["attribute"], "group": g["group"], "reference": g["reference"],
                               "n": int(mask.sum()), **{k: v for k, v in decision_counts(y_g, pred[mask]).items()
                                                       if k != "n"}}
        if pred_argmax is not None:
            c = decision_counts(y_g, pred_argmax[mask])
            row["k_under_argmax"], row["undertriage_argmax"] = c["k_under"], c["undertriage"]
        if row["n"] < MIN_SUBGROUP_N or len(np.unique(y_g)) < 2:
            row["skipped"] = f"za mała podgrupa (n = {row['n']} < {MIN_SUBGROUP_N}) albo jeden poziom ESI"
            rows.append(row)
            continue
        res = bootstrap_decisions(y_g, {"m": proba[mask]}, {"m": {"r": pred[mask]}}, ["quadratic_weighted_kappa"],
                                  n_boot=n_boot, alpha=alpha, seed=seed)
        ci = res.ci("m", "r").set_index("metric").loc["quadratic_weighted_kappa"]
        row["qwk_low"], row["qwk_high"] = fnum(ci["ci_low"]), fnum(ci["ci_high"])
        if admitted is not None and esi is not None:
            adm = admitted[mask]
            sel = np.isfinite(adm) & (adm == 1)
            n_adm = int(sel.sum())
            row["n_admitted"] = n_adm
            if n_adm >= 1:
                model35 = (pred[mask][sel] > URGENT_MAX).astype(float)
                nurse35 = (np.asarray(esi)[mask][sel] >= 3).astype(float)
                row["adm_model_k"], row["adm_nurse_k"] = int(model35.sum()), int(nurse35.sum())
                row["adm_delta"] = float(model35.mean() - nurse35.mean())
                if n_adm >= MIN_SUBGROUP_N:
                    row["adm_delta_low"], row["adm_delta_high"] = _paired_mean_ci(
                        model35 - nurse35, y_g[sel], n_boot, alpha, seed)
        rows.append(row)
    # Δ niedotriażowania względem grupy odniesienia (pierwszej) każdego atrybutu
    refs = {r["attribute"]: r for r in rows if r["reference"]}
    for row in rows:
        ref = refs.get(row["attribute"])
        if ref is None or row is ref or row.get("skipped") or ref.get("skipped"):
            continue
        d, lo, hi = newcombe_diff(row["k_under"], row["n_high"], ref["k_under"], ref["n_high"], alpha)
        row["under_delta"], row["under_delta_low"], row["under_delta_high"] = d, lo, hi
        row["reference_group"] = ref["group"]
    return {"rows": rows, "min_n": MIN_SUBGROUP_N}


FIT_POPULATION_TEXT: dict[str, str] = {
    "vc": "dopasowaną na VAL w podgrupie vc — komplet HR/SBP/RR",
    "all": "dopasowaną na całym VAL",
}


def render_subgroups(sec: dict[str, Any], fmt: Fmt, main_rule: str, conf: int, source: str | None,
                     population: str | None = None) -> list[str]:
    """`population` — populacja dopasowania reguły głównej ('vc' / 'all'; None = reguła bez dopasowania)."""
    lines = [f"## 8. Podgrupy pacjentów i równość (TEST, reguła {rule_label(main_rule)})", ""]
    if sec.get("skipped"):
        return lines + [f"Pominięte: {sec['skipped']}.", ""]
    fitted = (f"{FIT_POPULATION_TEXT.get(population, f'dopasowaną na VAL ({population})')}, bez zmian dla podgrup"
              if main_rule in FITTED_RULES and population else "bez parametrów dopasowanych do podgrup")
    lines += [
        f"Każdą podgrupę oceniono tą samą regułą ({fitted}). QWK — bootstrap "
        f"w podgrupie; odsetki — Clopper–Pearson z k/n. **Δ** = niedotriażowanie podgrupy − grupy "
        f"odniesienia (pierwszej w atrybucie), {conf}% przedział Newcombe'a; przedział bez zera = "
        "różnica istotna. Część różnic wynika z innego profilu pacjentów (case mix), dlatego obok jest "
        "miara niezależna od etykiety: odsetek **przyjętych do szpitala**, którym dano ESI 3–5.",
        "",
    ]
    if source:
        lines += [f"Atrybuty pacjentów: {source}.", ""]
    rows = []
    for r in sec["rows"]:
        if r.get("skipped"):
            rows.append([r["attribute"], r["group"], int_text(r["n"]), int_text(r["n_high"]), "—", "—", "—", "—",
                         r["skipped"]])
            continue
        ref = "odniesienie" if r["reference"] else fmt.pp(r.get("under_delta"), r.get("under_delta_low"),
                                                          r.get("under_delta_high"))
        rows.append([r["attribute"], r["group"], int_text(r["n"]), int_text(r["n_high"]),
                     fmt.num_ci(r["qwk"], r.get("qwk_low"), r.get("qwk_high")),
                     fmt.rate(r["k_under"], r["n_high"]), ref, fmt.rate(r["k_over"], r["n_low"]),
                     fmt.rate(r["k_crit"], r["n_esi1"])])
    lines += [md_table(["atrybut", "grupa", "n", "ESI 1–2", "QWK", "niedotriażowanie ESI 1–2",
                        "Δ niedotriażowania", "przetriażowanie ESI 4–5", "kryt. pominięcie ESI 1"], rows), ""]
    if any("n_admitted" in r for r in sec["rows"]):
        rows = []
        for r in sec["rows"]:
            if r.get("skipped") or not r.get("n_admitted"):
                continue
            n_adm = r["n_admitted"]
            rows.append([r["attribute"], r["group"], int_text(n_adm), fmt.rate(r["adm_model_k"], n_adm),
                         fmt.rate(r["adm_nurse_k"], n_adm),
                         fmt.pp(r.get("adm_delta"), r.get("adm_delta_low"), r.get("adm_delta_high"))])
        lines += ["Undertriage „zakotwiczone w przyjęciu”: wśród pacjentów PRZYJĘTYCH do szpitala odsetek "
                  "z decyzją ESI 3–5 (niższy = lepiej). Δ = model − pielęgniarka na tych samych pacjentach "
                  f"(sparowany bootstrap, {conf}% CI; dodatnia = model częściej odsyła przyjętych do ESI 3–5).",
                  "", md_table(["atrybut", "grupa", "przyjęci", "model: ESI 3–5", "pielęgniarka: ESI 3–5",
                                "Δ model − pielęgniarka"], rows), ""]
    else:
        lines += ["Brak danych o przyjęciach — kolumna undertriage „zakotwiczonego w przyjęciu” pominięta.", ""]
    return lines


# ─────────────────────────────────────────
# C4 — trafność względem przyjęcia do szpitala (model vs pielęgniarka)
# ─────────────────────────────────────────
def admission_labels(meta: pd.DataFrame, split: str, fingerprint: str | None) -> tuple[np.ndarray | None, str]:
    """
    Kolumna `admitted` w kolejności wierszy predykcji: z predykcji, a gdy jej tam nie ma —
    z przygotowanego splitu (tylko przy zgodnym odcisku danych). Zwraca (wartości | None, źródło/powód).
    """
    if "admitted" in meta.columns:
        values = pd.to_numeric(meta["admitted"], errors="coerce").to_numpy(dtype=float)
        return values, "predykcje runów"
    try:
        from src.data import dataset

        current = dataset.data_fingerprint()
        if current == dataset.NO_DATA_FINGERPRINT:
            return None, "brak kolumny `admitted` w predykcjach i brak przygotowanych splitów (data/processed)"
        if fingerprint and current != fingerprint:
            return None, (f"przygotowane splity mają inny odcisk danych ({current}) niż runy ({fingerprint}) "
                          f"— pomijam")
        frame = dataset.load_split(split, columns=["row_id", "admitted"]).set_index("row_id")
    except Exception as exc:  # noqa: BLE001 — brak splitu / kolumny = pomijamy sekcję
        return None, f"brak kolumny `admitted` w predykcjach i splicie ({type(exc).__name__}: {exc})"
    missing = meta.index.difference(frame.index)
    if len(missing):
        return None, f"{len(missing):,} wierszy predykcji nie ma w splicie '{split}' — pomijam"
    values = pd.to_numeric(frame.loc[meta.index, "admitted"], errors="coerce").to_numpy(dtype=float)
    return values, f"split '{split}' (data/processed)"


def usable_admissions(admitted: np.ndarray | None) -> bool:
    if admitted is None:
        return False
    ok = np.isfinite(admitted)
    return bool(ok.sum() > 0 and len(np.unique(admitted[ok])) == 2)


def nurse_esi(meta: pd.DataFrame) -> np.ndarray:
    """ESI z triażu (1–5): kolumna `esi`, a bez niej target + 1."""
    if "esi" in meta.columns:
        return pd.to_numeric(meta["esi"], errors="coerce").to_numpy(dtype=float)
    return meta["target"].to_numpy(dtype=float) + 1


def admission_section(
    admitted: np.ndarray, esi: np.ndarray, proba: np.ndarray, decisions: dict[str, np.ndarray],
    n_boot: int, alpha: float, seed: int, blend_label: str = "połączenie",
) -> dict[str, Any]:
    """AUC wyników pilności względem przyjęcia (z CI i Δ względem ESI z triażu) + odsetek przyjęć wg poziomu."""
    ok = np.isfinite(admitted)
    adm = admitted[ok].astype(int)
    p = proba[ok]
    e = np.asarray(esi)[ok]
    levels = np.arange(1, N_CLASSES + 1)
    scores: dict[str, np.ndarray] = {
        NURSE: -e.astype(float),
        f"{blend_label}: P(ESI 1–2)": p[:, 0] + p[:, 1],
        f"{blend_label}: oczekiwany poziom (−E[ESI])": -expected_level(p),
    }
    for rule, pred in decisions.items():
        scores[f"{blend_label}: przewidziany ESI ({rule_label(rule)})"] = -(np.asarray(pred)[ok] + 1).astype(float)
    table = bootstrap_auc(adm, scores, reference=NURSE, n_boot=n_boot, alpha=alpha, seed=seed)
    by_level = []
    for level in levels:
        row: dict[str, Any] = {"poziom": f"ESI {level}"}
        mask = e == level
        row["n (ESI z triażu)"] = int(mask.sum())
        row["przyjęci (ESI z triażu)"] = float(adm[mask].mean()) if mask.any() else None
        for rule, pred in decisions.items():
            m = (np.asarray(pred)[ok] + 1) == level
            row[f"n ({rule_label(rule)})"] = int(m.sum())
            row[f"przyjęci ({rule_label(rule)})"] = float(adm[m].mean()) if m.any() else None
        by_level.append(row)
    return {"n": int(ok.sum()), "admission_rate": float(adm.mean()), "auc": table, "by_level": by_level}


def admission_subgroups(
    admitted: np.ndarray, esi: np.ndarray, proba: np.ndarray, groups: list[dict[str, Any]],
    n_boot: int, alpha: float, seed: int,
) -> dict[str, Any]:
    """C4 w podgrupach: AUC przyjęcia dla −E[ESI] modelu i ESI pielęgniarki, sparowana Δ."""
    el = expected_level(proba)
    rows = []
    for g in groups:
        mask = g["mask"] & np.isfinite(admitted)
        adm = admitted[mask].astype(int)
        row = {"attribute": g["attribute"], "group": g["group"], "n": int(mask.sum()),
               "n_admitted": int(adm.sum())}
        if row["n"] < MIN_SUBGROUP_N or len(np.unique(adm)) < 2 or min(adm.sum(), len(adm) - adm.sum()) < 5:
            row["skipped"] = "za mało przyjętych / nieprzyjętych"
            rows.append(row)
            continue
        table = bootstrap_auc(adm, {NURSE: -np.asarray(esi)[mask], "model": -el[mask]}, reference=NURSE,
                              n_boot=n_boot, alpha=alpha, seed=seed).set_index("score")
        for key, name in ((NURSE, "nurse"), ("model", "model"), (f"Δ model − {NURSE}", "delta")):
            rec = table.loc[key]
            row[f"auc_{name}"], row[f"auc_{name}_low"], row[f"auc_{name}_high"] = (
                fnum(rec["value"]), fnum(rec["ci_low"]), fnum(rec["ci_high"]))
        row["p_delta"] = fnum(table.loc[f"Δ model − {NURSE}"].get("p_two_sided"))
        rows.append(row)
    return {"rows": rows}


def discordance_table(admitted: np.ndarray, esi: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    """
    Niezgodność model–pielęgniarka a przyjęcia: dla każdego ESI z triażu odsetek przyjętych,
    gdy model dał poziom PILNIEJSZY / TEN SAM / MNIEJ PILNY (Levin 2018: pacjenci ESI 3 podniesieni
    przez model powinni być przyjmowani częściej niż pozostali ESI 3).
    """
    ok = np.isfinite(admitted)
    adm = admitted[ok].astype(int)
    e = np.asarray(esi)[ok].astype(int)
    p = np.asarray(pred)[ok].astype(int) + 1
    rows = []
    for level in range(1, N_CLASSES + 1):
        sel = e == level
        row: dict[str, Any] = {"esi": level, "n": int(sel.sum())}
        for key, cond in (("up", p < e), ("same", p == e), ("down", p > e)):
            m = sel & cond
            row[f"n_{key}"] = int(m.sum())
            row[f"k_{key}"] = int(adm[m].sum())
        rows.append(row)
    return rows


def render_admission_extra(sub: dict[str, Any] | None, disc: list[dict[str, Any]] | None, fmt: Fmt,
                           main_rule: str, conf: int) -> list[str]:
    lines: list[str] = []
    if sub and sub.get("rows"):
        rows = []
        for r in sub["rows"]:
            if r.get("skipped"):
                rows.append([r["attribute"], r["group"], int_text(r["n"]), int_text(r["n_admitted"]), "—", "—",
                             r["skipped"], ""])
                continue
            rows.append([r["attribute"], r["group"], int_text(r["n"]), int_text(r["n_admitted"]),
                         fmt.num_ci(r["auc_nurse"], r["auc_nurse_low"], r["auc_nurse_high"]),
                         fmt.num_ci(r["auc_model"], r["auc_model_low"], r["auc_model_high"]),
                         fmt.delta("auc", r["auc_delta"], r["auc_delta_low"], r["auc_delta_high"]),
                         fmt.p(r.get("p_delta"))])
        lines += ["### 7.1. Model a pielęgniarka w podgrupach (AUC przyjęcia)", "",
                  "Wynik modelu: −E[ESI] (ciągły oczekiwany poziom); pielęgniarka: ESI z triażu. Δ = model − "
                  f"pielęgniarka, sparowany bootstrap ({conf}% CI).", "",
                  md_table(["atrybut", "grupa", "n", "przyjęci", "AUC pielęgniarka", "AUC model", "Δ AUC", "p"],
                           rows), ""]
    if disc:
        rows = []
        for r in disc:
            if r["n"] == 0:
                continue
            rows.append([f"ESI {r['esi']}", int_text(r["n"]),
                         *[(fmt.rate(r[f"k_{key}"], r[f"n_{key}"]) if r[f"n_{key}"] else "—")
                           for key in ("up", "same", "down")]])
        lines += [f"### 7.2. Niezgodność model–pielęgniarka a przyjęcia (reguła {rule_label(main_rule)})", "",
                  "Odsetek przyjętych do szpitala (k/n) wśród pacjentów danego ESI z triażu, gdy decyzja modelu "
                  "jest pilniejsza, taka sama albo mniej pilna. Jeśli model trafnie „poprawia” triaż, pacjenci "
                  "podniesieni przez model są przyjmowani częściej, a obniżeni — rzadziej niż zgodni.", "",
                  md_table(["ESI z triażu", "n", "model pilniej", "zgodnie", "model mniej pilnie"], rows), ""]
    return lines


# ─────────────────────────────────────────
# C10 — analiza błędów
# ─────────────────────────────────────────
def error_analysis(
    y: np.ndarray, pred: np.ndarray, attrs: pd.DataFrame | None, admitted: np.ndarray | None,
) -> dict[str, Any]:
    """
    Niedotriażowanie (ESI 1–2 → ESI 3–5) według pary poziomów i według grupy dolegliwości
    (cc_group_*; pacjent może mieć kilka grup — udziały nie sumują się do 100%), oraz przyjęcia
    niedotriażowanych pacjentów ESI 2 wobec ESI 2 trafnie pilnych (ślad szumu etykiet).
    """
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    under = np.isin(y, HIGH_ACUITY_CLASSES) & (pred > URGENT_MAX)
    total = int(under.sum())
    pairs = []
    for t in HIGH_ACUITY_CLASSES:
        for p in range(URGENT_MAX + 1, N_CLASSES):
            k = int(((y == t) & (pred == p)).sum())
            pairs.append({"true": t + 1, "pred": p + 1, "n": k, "share": k / total if total else None})
    out: dict[str, Any] = {"n_undertriaged": total, "pairs": pairs,
                           "esi1_as_esi2": int(((y == 0) & (pred == 1)).sum()), "n_esi1": int((y == 0).sum())}
    high = np.isin(y, HIGH_ACUITY_CLASSES)
    cc_cols = [c for c in CC_GROUP_LABELS if attrs is not None and c in attrs.columns]
    if cc_cols:
        flags = {c: _flag(attrs, c) for c in cc_cols}
        any_group = np.zeros(len(y), dtype=bool)
        for f in flags.values():
            any_group |= f
        items = [(CC_GROUP_LABELS[c], flags[c]) for c in cc_cols] + [("brak grupy (pozostałe dolegliwości)",
                                                                       ~any_group)]
        rows = []
        for label, f in items:
            n_high = int((high & f).sum())
            k = int((under & f).sum())
            rows.append({"group": label, "n_high": n_high, "n_under": k,
                         "rate": k / n_high if n_high else None, "share": k / total if total else None})
        out["complaints"] = sorted(rows, key=lambda r: -(r["n_under"]))
    else:
        out["complaints_skipped"] = "brak kolumn cc_group_* (atrybuty pacjentów niedostępne)"
    if usable_admissions(admitted):
        ok = np.isfinite(admitted)
        esi2 = (y == 1) & ok
        missed = esi2 & (pred > URGENT_MAX)
        kept = esi2 & (pred <= URGENT_MAX)
        out["label_noise"] = {
            "missed_n": int(missed.sum()), "missed_k": int(admitted[missed].sum()),
            "kept_n": int(kept.sum()), "kept_k": int(admitted[kept].sum()),
        }
    return out


def render_errors(sec: dict[str, Any], fmt: Fmt, main_rule: str) -> list[str]:
    lines = [f"## 9. Analiza błędów: gdzie model niedotriażuje (TEST, reguła {rule_label(main_rule)})", ""]
    total = sec["n_undertriaged"]
    rows = [[f"ESI {p['true']} → ESI {p['pred']}", int_text(p["n"]), fmt.pct(p["share"])]
            for p in sec["pairs"] if p["n"] > 0]
    lines += [f"Niedotriażowanych pacjentów ESI 1–2: {int_text(total)}. Według pary (prawda → decyzja):", "",
              md_table(["para", "n", "udział w niedotriażowaniu"], rows or [["—", "0", "—"]]), "",
              f"ESI 1 → ESI 2 (nie jest niedotriażowaniem w definicji repo, ale opóźnia resuscytację): "
              f"{int_text(sec['esi1_as_esi2'])} z {int_text(sec['n_esi1'])} pacjentów ESI 1.", ""]
    if sec.get("complaints"):
        rows = [[r["group"], int_text(r["n_high"]), fmt.rate(r["n_under"], r["n_high"]), fmt.pct(r["share"])]
                for r in sec["complaints"]]
        lines += ["Według grupy dolegliwości (cc_group_*; pacjent może należeć do kilku grup):", "",
                  md_table(["grupa dolegliwości", "ESI 1–2 w grupie", "niedotriażowani w grupie",
                            "udział we wszystkich niedotriażowaniach"], rows), ""]
    elif sec.get("complaints_skipped"):
        lines += [f"Grupy dolegliwości pominięte: {sec['complaints_skipped']}.", ""]
    noise = sec.get("label_noise")
    if noise and noise["missed_n"] > 0 and noise["kept_n"] > 0:
        d, lo, hi = newcombe_diff(noise["missed_k"], noise["missed_n"], noise["kept_k"], noise["kept_n"], fmt.alpha)
        conf = int(round(100 * (1 - fmt.alpha)))
        diff = f"Δ = {fmt.pp(d, lo, hi)} ({conf}% przedział Newcombe'a)"
        if hi < 0:
            text = (f"Ślad szumu etykiet: pacjenci ESI 2, których model uznał za mniej pilnych, są przyjmowani do "
                    f"szpitala istotnie rzadziej niż pacjenci ESI 2, których model zostawił jako pilnych ({diff}) — "
                    "część „pomyłek” to pacjenci, których dalszy przebieg był lżejszy niż etykieta.")
        elif lo > 0:
            text = (f"Pacjenci ESI 2, których model uznał za mniej pilnych, są przyjmowani do szpitala istotnie "
                    f"CZĘŚCIEJ niż pacjenci ESI 2 zostawieni jako pilni ({diff}) — tych pomyłek nie tłumaczy szum "
                    "etykiet: model pomija pacjentów, którzy rzeczywiście byli ciężej chorzy.")
        else:
            text = (f"Odsetek przyjęć pacjentów ESI 2, których model uznał za mniej pilnych, nie różni się istotnie "
                    f"od odsetka u pacjentów ESI 2 zostawionych jako pilni ({diff}) — te dane nie rozstrzygają, "
                    "czy pomyłki to szum etykiet.")
        lines += [text, "",
                  md_table(["pacjenci ESI 2", "przyjęci do szpitala"],
                           [["model: ESI 3–5 (niedotriażowani)", fmt.rate(noise["missed_k"], noise["missed_n"])],
                            ["model: ESI 1–2", fmt.rate(noise["kept_k"], noise["kept_n"])]]), ""]
    return lines


# ─────────────────────────────────────────
# C7 — krzywe decyzyjne (net benefit)
# ─────────────────────────────────────────
def _odds(pt: np.ndarray | float) -> np.ndarray:
    pt = np.asarray(pt, dtype=float)
    return pt / (1.0 - pt)


def net_benefit_score(outcome: np.ndarray, score: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """NB(p_t) = TP/n − FP/n · p_t/(1 − p_t) dla „leczymy, gdy wynik ≥ p_t”."""
    outcome = np.asarray(outcome).astype(float)
    score = np.asarray(score, dtype=float)
    n = len(outcome)
    out = []
    for pt in thresholds:
        treat = score >= pt
        tp = float((treat * outcome).sum()) / n
        fp = float((treat * (1 - outcome)).sum()) / n
        out.append(tp - fp * float(_odds(pt)))
    return np.asarray(out)


def net_benefit_fixed(outcome: np.ndarray, treat: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """NB decyzji o stałej klasyfikacji (np. reguła ESI ≤ 2) przy wadze FP zależnej od p_t."""
    outcome = np.asarray(outcome).astype(float)
    treat = np.asarray(treat).astype(float)
    n = len(outcome)
    tp = float((treat * outcome).sum()) / n
    fp = float((treat * (1 - outcome)).sum()) / n
    return tp - fp * _odds(np.asarray(thresholds, dtype=float))


def net_benefit_all(outcome: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    prev = float(np.mean(outcome))
    return prev - (1 - prev) * _odds(np.asarray(thresholds, dtype=float))


def _table_index(thresholds: np.ndarray, values: Sequence[float]) -> list[int]:
    return [int(np.argmin(np.abs(thresholds - v))) for v in values]


def dca_label_section(
    y: np.ndarray, proba: np.ndarray, vc: np.ndarray | None, decisions: dict[str, np.ndarray],
    fig_path: Path | None = None,
) -> dict[str, Any]:
    """
    C7a: wynik = etykieta ESI 1–2, ryzyko = P(ESI 1) + P(ESI 2) (po temperaturze). Decyzje reguł
    (ESI ≤ 2) jako klasyfikacje o stałym NB. Na całym TEST i w podgrupie vc.
    """
    thresholds = DCA_THRESHOLDS
    out: dict[str, Any] = {"thresholds": thresholds}
    for pop, mask in (("all", np.ones(len(y), dtype=bool)), ("vc", vc)):
        if mask is None or not np.asarray(mask).any():
            continue
        mask = np.asarray(mask, dtype=bool)
        outcome = np.isin(np.asarray(y)[mask], HIGH_ACUITY_CLASSES).astype(float)
        risk = proba[mask, 0] + proba[mask, 1]
        out[pop] = {
            "model": net_benefit_score(outcome, risk, thresholds),
            "treat_all": net_benefit_all(outcome, thresholds),
            "rules": {rule: net_benefit_fixed(outcome, np.asarray(pred)[mask] <= URGENT_MAX, thresholds)
                      for rule, pred in decisions.items()},
            "prevalence": float(outcome.mean()), "n": int(mask.sum()),
        }
    figure = None
    if fig_path is not None and "all" in out:
        from src.evaluation.visualizations import plot_decision_curve

        a = out["all"]
        figure = plot_decision_curve(
            {"model: P(ESI 1–2)": (thresholds, a["model"])}, fig_path,
            points={f"reguła {rule_label(r)} (ESI ≤ 2)": (thresholds, v) for r, v in a["rules"].items()},
            treat_all=(thresholds, a["treat_all"]),
            title="Krzywa decyzyjna: wykrywanie ESI 1–2 (TEST)")
    out["figure"] = Path(figure).name if figure else None
    return out


def logistic_fit(x: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None, slope: bool = True,
                 max_iter: int = 50) -> tuple[float, float]:
    """
    Regresja logistyczna y ~ a + b·x (+ offset) metodą Newtona (bez zależności od statsmodels).
    slope=False → tylko wyraz wolny przy b = 1 (kalibracja „in the large” z offsetem).
    Zwraca (a, b); brak zbieżności → ostatnie przybliżenie.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    off = np.zeros_like(x) if offset is None else np.asarray(offset, dtype=float)
    a, b = 0.0, 1.0
    if not slope:
        off = off + x
    for _ in range(max_iter):
        eta = a + (b * x if slope else 0.0) + off
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        w = p * (1 - p) + 1e-12
        r = y - p
        if slope:
            g = np.array([r.sum(), (r * x).sum()])
            h = np.array([[w.sum(), (w * x).sum()], [(w * x).sum(), (w * x * x).sum()]]) + 1e-9 * np.eye(2)
            step = np.linalg.solve(h, g)
            a, b = a + step[0], b + step[1]
            if np.abs(step).max() < 1e-10:
                break
        else:
            step = r.sum() / w.sum()
            a += step
            if abs(step) < 1e-10:
                break
    return float(a), float(b)


def dca_admission_section(
    adm_val: np.ndarray | None, el_val: np.ndarray | None, esi_val: np.ndarray | None,
    adm_test: np.ndarray, el_test: np.ndarray, esi_test: np.ndarray, decisions: dict[str, np.ndarray],
    n_boot: int, alpha: float, seed: int, fig_path: Path | None = None,
) -> dict[str, Any]:
    """
    C7b: wynik = przyjęcie do szpitala (niezależne od etykiety). Ryzyko modelu = logistyczna
    rekalibracja −E[ESI] → P(przyjęcia) dopasowana na VAL (wynik pilności nie jest
    prawdopodobieństwem przyjęcia); ryzyko pielęgniarki = odsetek przyjęć danego ESI na VAL.
    Porównanie z regułą pielęgniarki „ESI ≤ 2 = pilny” (NB o stałej klasyfikacji) i Δ NB z
    przedziałem bootstrap przy wybranych p_t.
    """
    if not usable_admissions(adm_val) or el_val is None or esi_val is None:
        return {"skipped": "brak przyjęć na VAL — rekalibracja wyniku modelu do P(przyjęcia) niemożliwa"}
    okv = np.isfinite(adm_val)
    a, b = logistic_fit(-el_val[okv], adm_val[okv])
    rates_val = {lvl: float(adm_val[okv][esi_val[okv] == lvl].mean()) if (esi_val[okv] == lvl).any() else None
                 for lvl in range(1, N_CLASSES + 1)}
    ok = np.isfinite(adm_test)
    outcome = adm_test[ok].astype(float)
    risk_model = 1.0 / (1.0 + np.exp(-(a + b * (-el_test[ok]))))
    e = np.asarray(esi_test)[ok]
    overall = float(np.nanmean(adm_val[okv]))
    risk_nurse = np.array([rates_val.get(int(v)) if rates_val.get(int(v)) is not None else overall for v in e])
    nurse_rule = e <= 2
    thresholds = DCA_THRESHOLDS
    out: dict[str, Any] = {
        "thresholds": thresholds, "n": int(ok.sum()), "prevalence": float(outcome.mean()),
        "recalibration": {"a": a, "b": b, "fit": "VAL, logit P(przyjęcia) = a + b·(−E[ESI])"},
        "nurse_rates_val": rates_val,
        "model": net_benefit_score(outcome, risk_model, thresholds),
        "nurse_risk": net_benefit_score(outcome, risk_nurse, thresholds),
        "nurse_rule": net_benefit_fixed(outcome, nurse_rule, thresholds),
        "treat_all": net_benefit_all(outcome, thresholds),
        "rules": {rule: net_benefit_fixed(outcome, np.asarray(pred)[ok] <= URGENT_MAX, thresholds)
                  for rule, pred in decisions.items()},
    }
    # Δ NB (model − reguła pielęgniarki ESI ≤ 2) przy wybranych progach — sparowany bootstrap
    pts = np.asarray(DCA_DELTA_THRESHOLDS, dtype=float)
    w = _odds(pts)
    treat_m = (risk_model[:, None] >= pts[None, :]).astype(float)
    c = (outcome[:, None] - (1 - outcome[:, None]) * w[None, :]) * (treat_m - nurse_rule[:, None].astype(float))
    point = c.mean(axis=0)
    samples = []
    for W in resample_weights(outcome.astype(int), n_boot, seed, stratify=True):
        samples.append((W @ c) / W.sum(axis=1, keepdims=True))
    s = np.concatenate(samples)
    low, high = np.percentile(s, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    out["delta_vs_nurse_rule"] = [{"pt": float(pt), "delta": float(d), "ci_low": float(lo), "ci_high": float(hi)}
                                  for pt, d, lo, hi in zip(pts, point, low, high)]
    figure = None
    if fig_path is not None:
        from src.evaluation.visualizations import plot_decision_curve

        figure = plot_decision_curve(
            {"model: P(przyjęcia) z −E[ESI]": (thresholds, out["model"]),
             "pielęgniarka: odsetek przyjęć jej ESI (VAL)": (thresholds, out["nurse_risk"])}, fig_path,
            points={"pielęgniarka: ESI ≤ 2": (thresholds, out["nurse_rule"]),
                    **{f"reguła {rule_label(r)} (ESI ≤ 2)": (thresholds, v) for r, v in out["rules"].items()}},
            treat_all=(thresholds, out["treat_all"]), title="Krzywa decyzyjna: przyjęcie do szpitala (TEST)")
    out["figure"] = Path(figure).name if figure else None
    return out


def render_dca(label: dict[str, Any], adm: dict[str, Any], fmt: Fmt, main_rule: str, conf: int) -> list[str]:
    lines = ["## 10. Krzywe decyzyjne — użyteczność kliniczna (TEST)", "",
             "**Net benefit** (NB; Vickers i Elkin 2006) = odsetek prawdziwie pozytywnych − odsetek fałszywie "
             "pozytywnych × p_t/(1 − p_t). Próg p_t wyraża, ilu fałszywych alarmów wart jest jeden "
             "trafiony pacjent (p_t = 0,25 ↔ 3 fałszywe alarmy na 1 trafienie; reguła kosztowa k = 3 "
             "odpowiada mniej więcej p_t ≈ 1/(1 + k)). Model jest użyteczny przy p_t, przy którym jego NB "
             "przewyższa „wszyscy” i „nikt” (NB = 0).", ""]
    if "all" in label:
        th = label["thresholds"]
        idx = _table_index(th, DCA_TABLE_THRESHOLDS)
        a = label["all"]
        header = ["p_t", "model: P(ESI 1–2)", *([] if "vc" not in label else ["model, vc"]),
                  *[f"reguła {rule_label(r)}" for r in a["rules"]], "wszyscy pilni"]
        rows = []
        for i in idx:
            row = [fmt.num(th[i], 2), fmt.num(a["model"][i])]
            if "vc" in label:
                row.append(fmt.num(label["vc"]["model"][i]))
            row += [fmt.num(v[i]) for v in a["rules"].values()]
            row.append(fmt.num(a["treat_all"][i]))
            rows.append(row)
        lines += [f"### 10.1. Wykrywanie ESI 1–2 (etykieta; częstość {fmt.pct(a['prevalence'])})", "",
                  md_table(header, rows), "",
                  "Analiza na etykiecie ESI jest częściowo „okrężna” (model uczy się tej samej etykiety) — "
                  "główną analizą jest 10.2 (przyjęcie do szpitala)." +
                  (f" Wykres: `figures/{label['figure']}`." if label.get("figure") else ""), ""]
    lines += ["### 10.2. Przyjęcie do szpitala: model a pielęgniarka", ""]
    if adm.get("skipped"):
        return lines + [f"Pominięte: {adm['skipped']}.", ""]
    th = adm["thresholds"]
    idx = _table_index(th, DCA_TABLE_THRESHOLDS)
    rows = [[fmt.num(th[i], 2), fmt.num(adm["model"][i]), fmt.num(adm["nurse_risk"][i]),
             fmt.num(adm["nurse_rule"][i]), *[fmt.num(v[i]) for v in adm["rules"].values()],
             fmt.num(adm["treat_all"][i])] for i in idx]
    rec = adm["recalibration"]
    lines += [f"Przyjęto {fmt.pct(adm['prevalence'])} pacjentów (n = {int_text(adm['n'])}). Ryzyko modelu = "
              f"rekalibracja logistyczna −E[ESI] → P(przyjęcia) dopasowana na VAL (a = {fmt.num(rec['a'])}, "
              f"b = {fmt.num(rec['b'])}); ryzyko pielęgniarki = odsetek przyjęć jej poziomu ESI na VAL.", "",
              md_table(["p_t", "model: P(przyjęcia)", "pielęgniarka: ryzyko z ESI", "pielęgniarka: ESI ≤ 2",
                        *[f"reguła {rule_label(r)} (ESI ≤ 2)" for r in adm["rules"]], "wszyscy"], rows), ""]
    rows = [[fmt.num(d["pt"], 2), fmt.delta("net_benefit", d["delta"], d["ci_low"], d["ci_high"])]
            for d in adm["delta_vs_nurse_rule"]]
    lines += [f"Δ NB = model − reguła pielęgniarki „ESI ≤ 2” (sparowany bootstrap, {conf}% CI; dodatnia = model "
              "użyteczniejszy przy tym progu):", "", md_table(["p_t", "Δ NB [CI]"], rows), "",
              "Zastrzeżenie: przyjęcie częściowo wynika z samego triażu (korzyść dla pielęgniarki) i nie jest "
              "tym samym co pilność; Yale nie ma danych o OIOM ani zgonach." +
              (f" Wykres: `figures/{adm['figure']}`." if adm.get("figure") else ""), ""]
    return lines


# ─────────────────────────────────────────
# C8 — kalibracja per poziom ESI i per oddział
# ─────────────────────────────────────────
def calibration_groups(y: np.ndarray, proba: np.ndarray, groups: list[tuple[str, np.ndarray]],
                       alpha: float = 0.05) -> dict[str, Any]:
    """
    O/E (obserwowane / oczekiwane = Σ p) każdego poziomu ESI w każdej grupie, z przedziałem
    z Clopper–Pearsona dla liczby obserwowanej (oczekiwane traktowane jako stałe), oraz dla
    P(ESI 1–2): kalibracja „in the large” (wyraz wolny przy offsecie logit p; 0 = idealnie;
    > 0 = model zaniża ryzyko) i nachylenie (1 = idealnie; < 1 = predykcje zbyt skrajne).
    """
    y = np.asarray(y).astype(int)
    classes, esi12 = [], []
    for label, mask in groups:
        mask = np.asarray(mask, dtype=bool)
        n = int(mask.sum())
        if n == 0:
            continue
        p = proba[mask]
        yy = y[mask]
        row: dict[str, Any] = {"group": label, "n": n}
        for k in range(N_CLASSES):
            obs = int((yy == k).sum())
            exp = float(p[:, k].sum())
            low, high = clopper_pearson(obs, n, alpha)
            row[f"obs_{k}"], row[f"exp_{k}"] = obs, exp
            row[f"oe_{k}"] = obs / exp if exp > 0 else None
            row[f"oe_{k}_low"] = low * n / exp if exp > 0 else None
            row[f"oe_{k}_high"] = high * n / exp if exp > 0 else None
        classes.append(row)
        outcome = np.isin(yy, HIGH_ACUITY_CLASSES).astype(float)
        risk = np.clip(p[:, 0] + p[:, 1], 1e-6, 1 - 1e-6)
        rec: dict[str, Any] = {"group": label, "n": n, "observed": float(outcome.mean()), "expected": float(risk.mean())}
        rec["oe"] = rec["observed"] / rec["expected"] if rec["expected"] > 0 else None
        if 0 < outcome.sum() < n and n >= MIN_SUBGROUP_N:
            logit = np.log(risk / (1 - risk))
            rec["citl"] = logistic_fit(logit, outcome, slope=False)[0]
            rec["intercept"], rec["slope"] = logistic_fit(logit, outcome, slope=True)
        esi12.append(rec)
    return {"classes": classes, "esi12": esi12}


def render_calibration_groups(sec: dict[str, Any], fmt: Fmt) -> list[str]:
    lines = ["## 11. Kalibracja według poziomu ESI i oddziału (TEST, połączenie po temperaturze)", "",
             "**O/E** = liczba obserwowanych pacjentów danego poziomu / suma przewidzianych prawdopodobieństw "
             "(1 = idealnie; > 1 = model ZANIŻA częstość poziomu, < 1 = zawyża). Przedział z Clopper–Pearsona "
             "dla liczby obserwowanej. Temperatura nie naprawi różnic między oddziałami — to przesunięcie "
             "rozkładu klas (sekcja 13.2: leave-site-out z korektą rozkładu).", ""]
    rows = []
    for r in sec["classes"]:
        rows.append([r["group"], int_text(r["n"]),
                     *[(fmt.num_ci(r[f"oe_{k}"], r[f"oe_{k}_low"], r[f"oe_{k}_high"], 2)
                        if r.get(f"oe_{k}") is not None else "—") for k in range(N_CLASSES)]])
    lines += [md_table(["grupa", "n", *[f"O/E ESI {k + 1}" for k in range(N_CLASSES)]], rows), ""]
    rows = [[r["group"], int_text(r["n"]), fmt.pct(r["observed"]), fmt.pct(r["expected"]), fmt.num(r.get("oe"), 2),
             fmt.signed(r.get("citl"), 2), fmt.num(r.get("slope"), 2)] for r in sec["esi12"]]
    lines += ["P(ESI 1–2): kalibracja „in the large” (wyraz wolny przy offsecie logit p; 0 = idealnie, > 0 = "
              "model zaniża ryzyko) i nachylenie kalibracji (1 = idealnie, < 1 = predykcje zbyt skrajne; Van "
              "Calster i in. 2019).", "",
              md_table(["grupa", "n", "obserwowane ESI 1–2", "średnie P(ESI 1–2)", "O/E", "wyraz wolny",
                        "nachylenie"], rows), ""]
    return lines


# ─────────────────────────────────────────
# C13 — odsyłanie niepewnych i zbiory conformal warunkowe po ESI
# ─────────────────────────────────────────
def uncertainty_score(proba: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    """
    Niepewność decyzji danej reguły (większa = bardziej niepewna) — ta sama statystyka, której
    reguła używa: progi T → wariancja oczekiwanego poziomu; koszt / Bayes T → najmniejszy oczekiwany
    koszt (ryzyko Bayesa); argmax → 1 − max p.
    """
    proba = np.asarray(proba, dtype=float)
    rule = spec.get("rule", "argmax")
    if rule == "target":
        el = expected_level(proba)
        levels = np.arange(proba.shape[1])[None, :]
        return (proba * (levels - el[:, None]) ** 2).sum(axis=1)
    if rule in ("cost", "bayes_target"):
        costs = cost_matrix() if rule == "cost" else bayes_cost_matrix(spec["lam"])
        return (proba @ costs).min(axis=1)
    return 1.0 - proba.max(axis=1)


def deferral_section(
    p_val: np.ndarray, val_mask: np.ndarray | None, y_test: np.ndarray, p_test: np.ndarray,
    vc_test: np.ndarray | None, pred_test: np.ndarray, spec: dict[str, Any],
    fractions: Sequence[float] = DEFERRAL_FRACTIONS,
) -> dict[str, Any]:
    """
    C13a: model odsyła do pielęgniarki odsetek f najbardziej niepewnych pacjentów (próg z VAL,
    populacja dopasowania), o reszcie decyduje reguła główna. Wynik na TEST: odsetek
    rozstrzyganych, QWK i niedotriażowanie wśród rozstrzyganych, odesłani ESI 1–2 oraz
    „niedotriażowani i nieodesłani” na 1000 wszystkich pacjentów.
    """
    u_val = uncertainty_score(p_val if val_mask is None else p_val[val_mask], spec)
    u_test = uncertainty_score(p_test, spec)
    y_test = np.asarray(y_test).astype(int)
    pops = {"all": np.ones(len(y_test), dtype=bool)}
    if vc_test is not None and np.asarray(vc_test).any():
        pops["vc"] = np.asarray(vc_test, dtype=bool)
    rows = []
    for f in fractions:
        thr = np.inf if f <= 0 else float(np.quantile(u_val, 1 - f))
        decided = u_test <= thr
        for pop, mask in pops.items():
            m = mask & decided
            c = decision_counts(y_test[m], pred_test[m])
            high = np.isin(y_test[mask], HIGH_ACUITY_CLASSES)
            rows.append({"fraction": float(f), "population": pop, "threshold": thr if np.isfinite(thr) else None,
                         "n": int(mask.sum()), "n_decided": int(m.sum()),
                         "decided_share": float(m.sum() / max(mask.sum(), 1)),
                         "qwk": c["qwk"], "k_under": c["k_under"], "n_high": c["n_high"],
                         "k_crit": c["k_crit"], "n_esi1": c["n_esi1"],
                         "high_deferred": float((~decided[mask])[high].mean()) if high.any() else None,
                         "under_not_deferred_per_1000": 1000.0 * c["k_under"] / max(int(mask.sum()), 1)})
    return {"rows": rows, "score": spec.get("rule", "argmax")}


def _qhat(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return math.inf
    k = int(math.ceil((n + 1) * (1 - alpha)))
    return math.inf if k > n else float(np.sort(scores)[k - 1])


def conformal_section(
    y_val: np.ndarray, p_val: np.ndarray, y_test: np.ndarray, p_test: np.ndarray, vc_test: np.ndarray | None,
    alpha: float = CONFORMAL_ALPHA,
) -> dict[str, Any]:
    """
    C13b: split conformal (wynik 1 − p_y) skalibrowany na VAL — zwykły LAC i warunkowy po
    prawdziwym ESI (Mondrian: osobny próg dla każdego poziomu). Pokrycie per poziom, średni
    rozmiar zbioru i odsetek zbiorów jednoelementowych na TEST (cała populacja i vc).
    """
    y_val = np.asarray(y_val).astype(int)
    y_test = np.asarray(y_test).astype(int)
    q_lac = _qhat(1 - p_val[np.arange(len(y_val)), y_val], alpha)
    q_cls = np.array([_qhat(1 - p_val[y_val == k, k], alpha) for k in range(N_CLASSES)])
    methods = {"LAC": (1 - p_test) <= q_lac, "LAC warunkowy po ESI": (1 - p_test) <= q_cls[None, :]}
    pops = {"all": np.ones(len(y_test), dtype=bool)}
    if vc_test is not None and np.asarray(vc_test).any():
        pops["vc"] = np.asarray(vc_test, dtype=bool)
    rows = []
    for name, sets in methods.items():
        for pop, mask in pops.items():
            s = sets[mask]
            yy = y_test[mask]
            cov = s[np.arange(len(yy)), yy]
            size = s.sum(axis=1)
            row = {"method": name, "population": pop, "n": int(mask.sum()), "coverage": float(cov.mean()),
                   "mean_size": float(size.mean()), "singleton": float((size == 1).mean())}
            for k in range(N_CLASSES):
                sel = yy == k
                row[f"cov_{k}"] = float(cov[sel].mean()) if sel.any() else None
                row[f"n_{k}"] = int(sel.sum())
            rows.append(row)
    return {"alpha": alpha, "q_lac": q_lac, "q_class": q_cls, "rows": rows}


def render_deferral(defer: dict[str, Any], conf_sets: dict[str, Any], fmt: Fmt, main_rule: str) -> list[str]:
    lines = ["## 12. Odsyłanie niepewnych przypadków i zbiory conformal (TEST)", ""]
    score_label = {"target": "wariancja oczekiwanego poziomu", "cost": "oczekiwany koszt reguły kosztowej",
                   "bayes_target": "oczekiwany koszt decyzji Bayesa", "argmax": "1 − max p"}[defer["score"]]
    rows = []
    for r in defer["rows"]:
        pop = "cała populacja" if r["population"] == "all" else "vc"
        rows.append([fmt.pct(r["fraction"], 0), pop, fmt.pct(r["decided_share"]), fmt.num(r["qwk"]),
                     fmt.rate(r["k_under"], r["n_high"]), fmt.rate(r["k_crit"], r["n_esi1"]),
                     fmt.pct(r["high_deferred"]), fmt.num(r["under_not_deferred_per_1000"], 1)])
    lines += [f"### 12.1. Odsyłanie do pielęgniarki (reguła {rule_label(main_rule)}, miara niepewności: "
              f"{score_label})", "",
              "Próg niepewności dobrany na VAL (w populacji dopasowania reguły) tak, by odesłać zadany odsetek "
              "pacjentów; w innej populacji odsetek odesłanych jest inny (kolumna „rozstrzyga model”). QWK, niedotriażowanie i krytyczne pominięcie liczone **tylko wśród rozstrzyganych przez "
              "model** — to nie jest wynik całego systemu (błędów pielęgniarki w przypadkach odesłanych nie da "
              "się zasymulować, bo etykiety pochodzą od niej).", "",
              md_table(["odesłani (cel)", "populacja", "rozstrzyga model", "QWK", "niedotriażowanie",
                        "kryt. pominięcie", "odesłani wśród ESI 1–2", "niedotriażowani i nieodesłani / 1000"],
                       rows), ""]
    rows = []
    for r in conf_sets["rows"]:
        pop = "cała populacja" if r["population"] == "all" else "vc"
        rows.append([r["method"], pop, fmt.pct(r["coverage"]),
                     *[fmt.pct(r[f"cov_{k}"]) for k in range(N_CLASSES)], fmt.num(r["mean_size"], 2),
                     fmt.pct(r["singleton"])])
    level = int(round(100 * (1 - conf_sets["alpha"])))
    lines += [f"### 12.2. Zbiory conformal {level}% (kalibracja na VAL)", "",
              "Zbiór poziomów ESI, który zawiera prawdziwy poziom z prawdopodobieństwem ≥ "
              f"{level}%. Zwykły LAC gwarantuje to średnio, ale nie dla rzadkiego ESI 1; wersja warunkowa po "
              "ESI kalibruje każdy poziom osobno (większe zbiory). Jako reguła decyzyjna conformal przegrywa "
              "z progami T (ULEPSZENIA.md, C13) — tu tylko jako miara niepewności.", "",
              md_table(["metoda", "populacja", "pokrycie", *[f"ESI {k + 1}" for k in range(N_CLASSES)],
                        "śr. rozmiar zbioru", "zbiory 1-elementowe"], rows), ""]
    return lines


# ─────────────────────────────────────────
# C1 / C3 / C9 — wyniki runs/analyses/
# ─────────────────────────────────────────
ANALYSIS_KINDS: tuple[str, ...] = ("seeds", "site_out", "learning_curve")


STALE_HINT = "uruchom `python scripts/run_all.py --stage analyses`"


def _analysis_models(kind: str, data: dict[str, Any]) -> list[str]:
    if kind == "site_out":
        return list(dict.fromkeys(m for models in (data.get("sites") or {}).values() for m in (models or {})))
    return list(data.get("models") or {})


def _drop_models(kind: str, data: dict[str, Any], drop: set[str]) -> None:
    if kind == "site_out":
        for models in (data.get("sites") or {}).values():
            for m in drop:
                (models or {}).pop(m, None)
    else:
        for m in drop:
            (data.get("models") or {}).pop(m, None)


def load_analyses(feature_set: str, tag: str | None, runs_dir: Path | str | None = None, *,
                  fingerprint: str | None = None, run_names: dict[str, str] | None = None) -> dict[str, Any]:
    """
    {rodzaj: {"path", "results"[, "stale_models"]} | {"skipped": powód}} dla
    runs/analyses/<rodzaj>__<cechy>__<tag>/.

    Kontrola aktualności (wyniki z innych runów / danych nie trafiają do raportu bez słowa):
        - inna wersja metryk analiz (src.analysis.common.METRICS_VERSION) → pominięte,
        - `fingerprint` podany i ≠ data_fingerprint analizy → pominięte (inne dane),
        - `run_names` {model: run raportu}: model, którego analiza pochodzi z INNEGO runa niż run
          raportu, jest usuwany z wyników (lista w "stale_models"); bez modeli → pominięte.
    """
    from src.analysis.common import METRICS_VERSION
    from src.utils import config

    root = Path(runs_dir if runs_dir is not None else config.RUNS_DIR) / "analyses"
    out: dict[str, Any] = {}
    for kind in ANALYSIS_KINDS:
        if not tag:
            out[kind] = {"skipped": "runy bez wspólnego tagu — nie wiadomo, których analiz szukać"}
            continue
        path = root / f"{kind}__{feature_set}__{tag}" / "results.json"
        if not path.exists():
            out[kind] = {"skipped": f"brak `{path.parent.name}/results.json` (uruchom "
                                    f"`python scripts/run_all.py --stage analyses`)"}
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            out[kind] = {"skipped": f"nieczytelny {path} ({exc})"}
            continue
        if not isinstance(data, dict):
            out[kind] = {"skipped": f"nieczytelny {path}"}
            continue
        if data.get("metrics_version") != METRICS_VERSION:
            out[kind] = {"skipped": f"nieaktualne: `{path.parent.name}` ze starszej wersji metryk analiz "
                                    f"({data.get('metrics_version')} ≠ {METRICS_VERSION}) — {STALE_HINT}"}
            continue
        if fingerprint and data.get("data_fingerprint") and data["data_fingerprint"] != fingerprint:
            out[kind] = {"skipped": f"nieaktualne: `{path.parent.name}` z innych danych (odcisk "
                                    f"{data['data_fingerprint']} ≠ {fingerprint}) — {STALE_HINT}"}
            continue
        entry: dict[str, Any] = {"path": str(path), "results": data}
        if run_names:
            sources = data.get("source_runs") or {}
            stale = {m: sources.get(m) for m in _analysis_models(kind, data)
                     if m in run_names and sources.get(m) != run_names[m]}
            if stale:
                _drop_models(kind, data, set(stale))
                entry["stale_models"] = {m: {"analysis": src, "report": run_names[m]} for m, src in stale.items()}
                if not _analysis_models(kind, data):
                    out[kind] = {"skipped": f"nieaktualne: `{path.parent.name}` liczone dla innych runów niż w "
                                            f"raporcie ({', '.join(sorted(stale))}) — {STALE_HINT}"}
                    continue
        out[kind] = entry
    return out


def _stale_note(entry: dict[str, Any]) -> list[str]:
    stale = entry.get("stale_models") or {}
    if not stale:
        return []
    items = "; ".join(f"{m} (analiza: {v['analysis'] or '—'}, raport: {v['report']})" for m, v in stale.items())
    return [f"⚠ Pominięto nieaktualne wyniki modeli liczone z innego runa niż run raportu: {items} — {STALE_HINT}.",
            ""]


def seed_key(metric: str, rule: str) -> str | None:
    """Klucz metryki w results.json seedów dla (metryka 08, reguła); None = metryka nieobecna."""
    if metric in SEED_PROBA_KEYS:
        return SEED_PROBA_KEYS[metric]
    if metric in SEED_METRIC_KEYS and rule in SEED_RULES:
        return f"{SEED_METRIC_KEYS[metric]}_{rule}"
    return None


def seed_sd(seeds: dict[str, Any] | None, models: Sequence[str], metric: str, rule: str,
            split: str = "test", population: str = "all") -> float | None:
    """Największe SD seedów wśród `models` (None, gdy brak wyników / metryki)."""
    if not seeds or "results" not in seeds:
        return None
    key = seed_key(metric, rule)
    if key is None:
        return None
    if population == "vc":
        key = f"vc_{key}"
    sds = []
    for m in models:
        entry = (seeds["results"].get("models") or {}).get(m) or {}
        stat = (entry.get(split) or {}).get(key) or {}
        sd = fnum(stat.get("sd"))
        if sd is not None and (stat.get("n") or 0) >= 2:
            sds.append(sd)
    return max(sds) if sds else None


def superiority(metric: str, delta: Any, ci_low: Any, ci_high: Any, sd: float | None,
                name_a: str = "A", name_b: str = "B") -> str:
    """
    Reguła C1: wyższość tylko, gdy przedział Δ = A − B wyklucza 0 I |Δ| > MID I |Δ| > 2 × SD seedów.
    Werdykt pozytywny NAZYWA zwycięzcę wg kierunku metryki („TAK — <nazwa>”): dla QWK / AUC lepsza
    dodatnia Δ (wygrywa A), dla niedotriażowania / log loss (statistics.LOWER_IS_BETTER) — ujemna.
    Zwraca krótki werdykt po polsku.
    """
    d, lo, hi = fnum(delta), fnum(ci_low), fnum(ci_high)
    if d is None or lo is None or hi is None:
        return "—"
    if lo <= 0 <= hi:
        return "nie (CI zawiera 0)"
    mid = MINIMAL_IMPORTANT_DIFFERENCE.get(metric)
    if mid is not None and abs(d) <= mid:
        return f"nie (Δ poniżej MID {mid:g})"
    if sd is None:
        return "nierozstrzygnięte (brak SD seedów)" if mid is not None else "brak MID i SD seedów"
    if abs(d) <= SEED_SD_FACTOR * sd:
        return f"nie (Δ poniżej 2 × SD seedów = {SEED_SD_FACTOR * sd:.4f})"
    winner = name_a if (d > 0) != (metric in LOWER_IS_BETTER) else name_b
    return f"TAK — {winner}" if mid is not None else f"tak (bez MID) — {winner}"


OE_DISPLAY_MAX: float = 999.0


def _oe_text(value: Any, fmt: Fmt) -> str:
    """O/E do tabeli; skrajne wartości (oczekiwane ≈ 0, np. EM wyzerowało klasę) jako '> 999'."""
    v = fnum(value)
    if v is not None and v > OE_DISPLAY_MAX:
        return f"> {OE_DISPLAY_MAX:g}"
    return fmt.num(v, 2)


def _ms(stat: dict[str, Any] | None, fmt: Fmt, digits: int = 3, percent: bool = False) -> str:
    if not stat or fnum(stat.get("mean")) is None:
        return "—"
    mean, sd = fnum(stat["mean"]), fnum(stat.get("sd")) or 0.0
    if percent:
        return fmt._dc(f"{100 * mean:.1f}% ± {100 * sd:.1f}")
    return fmt._dc(f"{mean:.{digits}f} ± {sd:.{digits}f}")


LC_METRIC_LABELS: dict[str, str] = {"log_loss": "log loss", "qwk_argmax": "QWK argmax", "qwk_cost": "QWK koszt",
                                    "auc_macro": "AUC makro", "undertriage_cost": "niedotriażowanie koszt"}
SEED_TABLE: list[tuple[str, str, bool]] = [
    ("qwk_argmax", "QWK argmax", False), ("qwk_cost", "QWK koszt", False), ("qwk_target", "QWK progi T", False),
    ("undertriage_target", "niedotriażowanie progi T", True), ("critical_miss_target", "kryt. pominięcie progi T", True),
    ("log_loss", "log loss", False), ("auc_macro", "AUC makro", False), ("admission_auc", "AUC przyjęcia", False),
]


def render_analyses(analyses: dict[str, Any], fmt: Fmt, report_opts: DecisionOptions,
                    fig_names: dict[str, str] | None = None) -> list[str]:
    """Sekcja 13: podsumowanie seedów (C1), leave-site-out (C3) i krzywej uczenia (C9)."""
    lines = ["## 13. Analizy dodatkowe (runs/analyses/, refity bez tuningu)", ""]
    # C1
    seeds = analyses.get("seeds") or {}
    lines += ["### 13.1. Zmienność między seedami (C1)", ""]
    if seeds.get("skipped"):
        lines += [f"Pominięte: {seeds['skipped']}.", ""]
    else:
        res = seeds["results"]
        lines += _stale_note(seeds)
        for split, title in (("test", "TEST"), ("val", "VAL")):
            rows = []
            for model, entry in (res.get("models") or {}).items():
                stats = entry.get(split) or {}
                n = max([int(s.get("n") or 0) for s in stats.values() if isinstance(s, dict)] or [0])
                rows.append([model, str(n), *[_ms(stats.get(k), fmt, percent=pct) for k, _, pct in SEED_TABLE]])
            if rows:
                lines += [f"{title} — średnia ± SD (n refitów z innym seedem{' + run źródłowy' if res.get('includes_source') else ''}):",
                          "", md_table(["model", "n", *[label for _, label, _ in SEED_TABLE]], rows), ""]
        note = res.get("target_rule_note")
        lines += [f"Reguła „progi T” w analizie seedów: {note}." if note else "",
                  "Regresja logistyczna ma deterministyczny solver — SD = 0 oznacza brak zmienności treningu, "
                  "nie brak niepewności. **Wyższość jednego modelu nad drugim ogłaszaj tylko, gdy przedział "
                  "sparowanej Δ wyklucza 0 i |Δ| > minimalna istotna różnica (MID: QWK 0,01; niedotriażowanie "
                  "1 pp; AUC 0,005; log loss 0,005) i |Δ| > 2 × SD seedów** — kolumna „wyższość (C1)” w sekcji 3.",
                  ""]
        if report_opts.main_rule == "target" and abs(report_opts.target - 0.15) > 1e-9:
            lines += ["Uwaga: reguła raportu ma inny cel T niż reguła w analizie seedów — SD dla „progi T” jest "
                      "przybliżeniem.", ""]
    # C3
    site = analyses.get("site_out") or {}
    lines += ["### 13.2. Walidacja z wyłączeniem oddziału (leave-site-out, C3)", ""]
    if site.get("skipped"):
        lines += [f"Pominięte: {site['skipped']}.", ""]
    else:
        res = site["results"]
        lines += _stale_note(site)
        lines += ["Model trenowany na wierszach TRAIN pozostałych oddziałów, oceniany na wierszach TRAIN + VAL "
                  "wyłączonego oddziału (nigdy TEST). Warianty: *surowy*; *znany rozkład* — korekta "
                  "prawdopodobieństw do lokalnego rozkładu ESI (Saerens i in. 2002); *EM* — rozkład "
                  "estymowany bez etykiet; *w rozkładzie* — model z tymi samymi parametrami trenowany na "
                  "wszystkich oddziałach, oceniany na VAL tego oddziału (punkt odniesienia). „progi T "
                  "przeniesione” = reguła dopasowana na VAL pozostałych oddziałów; „lokalne” = dopasowana na "
                  "połowie wierszy oddziału, oceniona na drugiej połowie. ⚠ = EM niestabilne (nie interpretuj). "
                  "**O/E ESI 1** = obserwowane / oczekiwane (Σ P(ESI 1)) na wyłączonym oddziale — ta sama "
                  "definicja co w sekcji 11: > 1 = model ZANIŻA częstość ESI 1, < 1 = zawyża.",
                  ""]
        variants = [("raw", "surowy"), ("prior_known", "znany rozkład"), ("prior_em", "EM"),
                    ("in_distribution", "w rozkładzie")]
        for s, models in (res.get("sites") or {}).items():
            rows = []
            for model, entry in models.items():
                for key, label in variants:
                    m = entry.get(key)
                    if not isinstance(m, dict):
                        continue
                    flag = " ⚠" if m.get("em_suspect") else ""
                    rows.append([model, label + flag, int_text(m.get("n") or 0), fmt.num(m.get("qwk_argmax")),
                                 fmt.pct(m.get("undertriage_argmax")), fmt.num(m.get("qwk_target_transfer")),
                                 fmt.pct(m.get("undertriage_target_transfer")), fmt.num(m.get("target_local_qwk")),
                                 fmt.pct(m.get("target_local_undertriage")), fmt.num(m.get("log_loss")),
                                 fmt.num(m.get("admission_auc")), _oe_text(m.get("oe_esi1"), fmt)])
                nurse = fnum(entry.get("nurse_admission_auc"))
                if nurse is not None:
                    rows.append([model, "pielęgniarka (ESI z triażu)", int_text(entry.get("n") or 0), "—", "—", "—",
                                 "—", "—", "—", "—", fmt.num(nurse), "—"])
            if rows:
                lines += [f"Wyłączony oddział {s}:", "",
                          md_table(["model", "wariant", "n", "QWK argmax", "niedotr. argmax", "QWK progi T przen.",
                                    "niedotr. progi T przen.", "QWK progi T lokalne", "niedotr. progi T lokalne",
                                    "log loss", "AUC przyjęcia", "O/E ESI 1"], rows), ""]
        weighted = sorted({f"{m} ({(e.get('priors') or {}).get('train_weights')})"
                           for models in (res.get("sites") or {}).values() for m, e in (models or {}).items()
                           if isinstance(e, dict) and (e.get("priors") or {}).get("train_weights")})
        if weighted:
            lines += ["Modele trenowane z wagami klas: " + ", ".join(weighted) + " — korekta rozkładu liczona "
                      "względem rozkładu treningowego przemnożonego przez wagi (taki rozkład „widział” model).", ""]
        lines += ["Ranking (AUC przyjęcia) zwykle się przenosi, rozkład klas — nie: to argument, że progi "
                  "warstwy decyzyjnej trzeba dobrać na nowo na polskim rozkładzie ESI (TOPSOR).", ""]
    # C9
    lc = analyses.get("learning_curve") or {}
    lines += ["### 13.3. Krzywa uczenia (C9, uzasadnienie liczebności — TRIPOD+AI pkt 10)", ""]
    if lc.get("skipped"):
        lines += [f"Pominięte: {lc['skipped']}.", ""]
    else:
        res = lc["results"]
        lines += _stale_note(lc)
        for model, entry in (res.get("models") or {}).items():
            rows = []
            for pt in entry.get("points") or []:
                v = pt.get("val") or {}
                rows.append([fmt.num(pt.get("fraction"), 2), int_text(pt.get("n_fit") or 0), fmt.num(v.get("log_loss")),
                             fmt.num(v.get("qwk_argmax")), fmt.num(v.get("qwk_cost")), fmt.pct(v.get("undertriage_cost")),
                             fmt.num(v.get("auc_macro"))])
            lines += [f"{model} (VAL):", "",
                      md_table(["ułamek X_fit", "n", "log loss", "QWK argmax", "QWK koszt", "niedotr. koszt",
                                "AUC makro"], rows), ""]
            ext = []
            for metric, fit in (entry.get("power_law") or {}).items():
                pred = fit.get("predicted") or {}
                values = ", ".join(f"n = {int_text(float(n))}: {fmt.num(v)}" for n, v in pred.items())
                reasons = "; ".join(fit.get("reasons") or [])
                verdict = ("wiarygodna" if fit.get("reliable")
                           else "**niewiarygodna** — nie raportuj" + (f" ({reasons})" if reasons else ""))
                ext.append([LC_METRIC_LABELS.get(metric, metric), fmt.num(fit.get("a")), fmt.num(fit.get("c"), 2),
                            values or "—", verdict])
            if ext:
                lines += ["Dopasowanie potęgowe metric(n) = a + b·n^(−c) i ekstrapolacja:", "",
                          md_table(["metryka", "a (asymptota)", "c", "ekstrapolacja", "ocena"], ext), ""]
        if fig_names and fig_names.get("learning_curve"):
            lines += [f"Wykres: `figures/{fig_names['learning_curve']}`.", ""]
    return lines


__all__ = [
    "COUNT_METRICS",
    "DecisionOptions",
    "FittedRules",
    "Fmt",
    "MINIMAL_IMPORTANT_DIFFERENCE",
    "NURSE",
    "PERCENT_METRICS",
    "RULE_LABELS",
    "RULE_ORDER",
    "admission_labels",
    "admission_section",
    "admission_subgroups",
    "bootstrap_decisions",
    "calibration_groups",
    "conformal_section",
    "dca_admission_section",
    "dca_label_section",
    "decision_counts",
    "decision_point_metrics",
    "deferral_section",
    "discordance_table",
    "error_analysis",
    "fit_mask",
    "fit_rules",
    "jsonable",
    "load_analyses",
    "load_attributes",
    "logistic_fit",
    "md_table",
    "net_benefit_all",
    "net_benefit_fixed",
    "net_benefit_score",
    "newcombe_diff",
    "predict_rules",
    "seed_sd",
    "subgroup_masks",
    "subgroup_section",
    "superiority",
    "tradeoff_section",
    "uncertainty_score",
    "val_rule_summary",
]
