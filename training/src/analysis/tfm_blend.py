"""
Eksperyment B (TFM) — czy dołączenie wstępnie wytrenowanych modeli fundamentalnych (TabPFN-3.5,
TabICLv2) do LightGBM poprawia prawdopodobieństwa ESI na PEŁNYCH danych. Wyłącznie VAL: zbiór
testowy ocenia raz `scripts/08_final_report.py` z recepturą wskazaną tutaj (recommendation.json).

Wejście: najnowsze ukończone, dostrojone runy `lightgbm`, `tabpfn`, `tabicl` z tagiem `tag` (albo jawne
`runs`) — wyłącznie ich zapisane predykcje VAL (bez GPU, bez ponownej predykcji). Odniesienie (gdy runy
istnieją): zestaw modeli klasycznych z tym samym tagiem („LightGBM sam vs zestaw klasyczny”).

Zestawy (najwyżej 2 TFM): {lgbm} (punkt odniesienia), {lgbm, tabpfn}, {lgbm, tabicl},
{lgbm, tabpfn, tabicl}; {tabpfn}, {tabicl} same — tylko odniesienie. Metody łączenia: średnia i średnia
geometryczna (`src.ensemble.blend`); dla każdego zestawu wybierana ta z niższym log loss.

Uczciwa ocena na VAL — powtarzany (`repeats`, domyślnie 5) dwukrotny cross-fitting, foldy warstwowane
ESI × oddział: na połowie dopasowania wagi (`optimize_weights`, min log loss), temperatura
(`fit_temperature`) i reguły decyzyjne jak w 08 (`report_sections.fit_rules`: argmax, koszt, progi T
z siatką bezpieczeństwa, populacja dopasowania vc); ocena na drugiej połowie. LightGBM przechodzi tę
samą procedurę (własna temperatura, własne reguły). Metryki: log loss, Brier, ECE (top-label oraz
ESI 1 / ESI 5 jeden-przeciw-reszcie), AUC makro, QWK / niedotriażowanie / krytyczne pominięcia /
czułość ESI 1 dla reguł argmax, koszt i progi T. Δ = zestaw − LightGBM z różnic na TYCH SAMYCH
pacjentach (dla log loss / Brier — średnia różnic per wiersz) uśrednionych po powtórzeniach; 95% CI
ze sparowanego bootstrapu warstwowanego względem ESI (`statistics.resample_weights`, te same losowania
dla wszystkich zestawów i powtórzeń; `report_sections.bootstrap_decisions`).

Bramka (ustalona z góry, wypisywana w raporcie PRZED liczbami): Δ log loss ≤ −MID (0,005) I górna
granica CI < 0 I |Δ| > 2 × SD seedów LightGBM (log loss na VAL z analizy seedów; brak → kryterium
niesprawdzalne → „nierozstrzygnięte”, zestaw nierekomendowany) I niedotriażowanie (progi T) nie gorsze
o > 1 pp I ECE ESI 1 nie gorsze o > 0,01. Oszczędność: para TFM tylko, gdy bije najlepszy przechodzący
zestaw z jednym TFM o ≥ MID.

Wyniki: runs/analyses/tfm_blend__<cechy>__<tag>/ — plan.json, units/repeat<k>.json (UnitStore; klucz
source_run = runy wejściowe), preds/repeat<k>.npz (predykcje cross-fitted), results.json,
recommendation.json (runy + dokładne polecenie 08), report.md (po polsku, do pracy), summary.md, figures/.
Wznawialne: gotowe powtórzenia są pomijane (`force` liczy od nowa).
"""

from __future__ import annotations

import hashlib
import json
import math
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.common import (
    METRICS_VERSION,
    AnalysisArgumentError,
    UnitStore,
    analyses_root,
    default_tag,
    execute_units,
    format_duration,
    now_iso,
    proba_metrics,
    rule_metrics,
    write_json,
)
from src.utils import config
from src.utils.fileio import replace_with_retry, write_text_atomic
from src.utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────
# Stałe (ustalone z góry — zmiana = inny eksperyment)
# ─────────────────────────────────────────
ANALYSIS = "tfm_blend"
# Wersja semantyki jednostek — PODBIJ przy zmianie sposobu liczenia powtórzenia (stare zostaną przeliczone)
TFM_BLEND_VERSION = 1
BASE_MODEL = "lightgbm"
BASELINE = "lightgbm"                     # identyfikator wariantu odniesienia (LightGBM sam)
CLASSICAL_SET = "classical"               # odniesienie: zestaw modeli klasycznych (jeśli są runy)
N_FOLDS = 2
DEFAULT_REPEATS = 5
DEFAULT_BOOTSTRAP = 1_000
METHODS: tuple[str, ...] = ("mean", "geometric")
MAX_TFMS = 2
ALPHA = 0.05
RULES: tuple[str, ...] = ("argmax", "cost", "target")
MAIN_RULE = "target"
N_BINS = 15                                # biny ECE (jak statistics.DEFAULT_N_BINS i 08)
PREDS_SUBDIR = "preds"
FIGURES_SUBDIR = "figures"

# Bramka — progi (pre-rejestracja)
GATE_MAX_UNDERTRIAGE_WORSE = 0.01          # niedotriażowanie reguły progi T: najwyżej +1 pp
GATE_MAX_ECE_ESI1_WORSE = 0.01             # ECE ESI 1 (jeden-przeciw-reszcie): najwyżej +0,01
PASSED, FAILED, INCONCLUSIVE = "passed", "failed", "inconclusive"
PASS, FAIL, NOT_CHECKED = "pass", "fail", "not_checked"
STATUS_PL: dict[str, str] = {PASSED: "PRZESZEDŁ", FAILED: "nie przeszedł", INCONCLUSIVE: "nierozstrzygnięte"}
CRITERION_PL: dict[str, str] = {PASS: "✓", FAIL: "✗", NOT_CHECKED: "?"}

DISPLAY: dict[str, str] = {
    "lightgbm": "LightGBM", "tabpfn": "TabPFN-3.5", "tabicl": "TabICLv2", "logreg": "regresja logistyczna",
    "xgboost": "XGBoost", "catboost": "CatBoost", "random_forest": "las losowy", "ebm": "EBM",
}
METHOD_PL: dict[str, str] = {"mean": "średnia", "geometric": "geometryczna", "single": "—"}
RULE_PL: dict[str, str] = {"argmax": "argmax", "cost": "koszt", "target": "progi T"}
# Kolory (paleta referencyjna, kolejność stała; kolor = byt, nie ranga) i znaczniki (kodowanie bez koloru)
SET_COLORS: dict[str, str] = {
    "lightgbm": "#2a78d6", "tabpfn": "#eb6834", "tabicl": "#1baf7a", "lightgbm+tabpfn": "#eb6834",
    "lightgbm+tabicl": "#1baf7a", "lightgbm+tabpfn+tabicl": "#6250d6", CLASSICAL_SET: "#2a78d6",
}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

# Metryki: (klucz wyników, metryka statistics, reguła | None = z prawdopodobieństw)
_STAT_NAMES: dict[str, str] = {"quadratic_weighted_kappa": "qwk", "undertriage_rate": "undertriage",
                               "critical_miss_rate": "critical_miss", "recall_esi1": "recall_esi1"}
# AUC bez bootstrapu (wartość punktowa): rangowe AUC w losowaniach to ~95% czasu bootstrapu, a AUC nie
# wchodzi do bramki
BOOT_METRICS: tuple[str, ...] = ("log_loss", "brier", "ece", "quadratic_weighted_kappa",
                                 "undertriage_rate", "critical_miss_rate", "recall_esi1")
PROBA_KEYS: tuple[str, ...] = ("log_loss", "brier", "ece", "ece_esi1", "ece_esi5", "auc_macro")
RULE_KEYS: tuple[str, ...] = tuple(f"{m}_{r}" for r in RULES for m in ("qwk", "undertriage", "critical_miss",
                                                                         "recall_esi1"))
METRIC_KEYS: tuple[str, ...] = PROBA_KEYS + RULE_KEYS
HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {"auc_macro"} | {f"{m}_{r}" for r in RULES for m in ("qwk", "recall_esi1")})
CLASSWISE_ECE: dict[str, int] = {"ece_esi1": 0, "ece_esi5": 4}


class BlendSkipped(RuntimeError):  # noqa: N818 — jak common.NoSourceRuns (warunek, nie błąd)
    """Brak warunków eksperymentu B (brak runu lightgbm albo żadnego runu TFM) — skrypt: `skipped: …`, kod 0."""


# ─────────────────────────────────────────
# Konfiguracja
# ─────────────────────────────────────────
@dataclass
class BlendConfig:
    feature_set: str = "triage"
    tag: str | None = None
    models: str | Sequence[str] | None = "tfm"      # TFM do sprawdzenia (lightgbm zawsze)
    runs: Sequence[str] | None = None                 # jawne runy (lightgbm + ≤ 2 TFM)
    repeats: int = DEFAULT_REPEATS
    n_boot: int = DEFAULT_BOOTSTRAP
    methods: str | Sequence[str] = METHODS
    seed: int = config.RANDOM_SEED
    target: float = config.DECISION_TARGET_UNDERTRIAGE
    safety_net: bool = True
    fit_population: str = config.DECISION_FIT_POPULATION
    reference: bool = True                            # zestaw klasyczny jako odniesienie
    plots: bool = True
    force: bool = False
    runs_dir: Path | str | None = None


@dataclass(frozen=True)
class Variant:
    """Wariant = zestaw członków + metoda łączenia; `kind`: baseline | candidate | tfm_only | reference."""

    id: str
    set_id: str
    members: tuple[str, ...]
    method: str
    kind: str

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "set": self.set_id, "members": list(self.members), "method": self.method,
                "kind": self.kind}


@dataclass
class ValData:
    """Wyrównane predykcje VAL wszystkich runów (kolejność wierszy = rosnące row_id)."""

    y: np.ndarray
    probas: dict[str, np.ndarray]
    vc: np.ndarray | None
    site: np.ndarray | None
    row_ids: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.y))


@dataclass
class Selection:
    """Runy eksperymentu: lightgbm, TFM (≤ 2) i odniesienia klasyczne."""

    lightgbm: Any
    tfms: dict[str, Any]
    references: dict[str, Any] = field(default_factory=dict)
    tag: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def members(self) -> dict[str, Any]:
        return {BASE_MODEL: self.lightgbm, **self.tfms}

    def source_key(self) -> str:
        return "+".join(r.name for r in [*self.members.values(), *self.references.values()])


def parse_methods(spec: str | Sequence[str]) -> list[str]:
    items = [s.strip().lower() for s in (spec.split(",") if isinstance(spec, str) else spec) if str(s).strip()]
    bad = [m for m in items if m not in METHODS]
    if bad or not items:
        raise AnalysisArgumentError(f"--methods: dozwolone {', '.join(METHODS)} (podano {spec!r})")
    return list(dict.fromkeys(items))


def parse_tfm_models(spec: str | Sequence[str] | None) -> list[str]:
    """TFM z listy modeli (nazwy / aliasy / grupy; lightgbm pomijany — zawsze w eksperymencie)."""
    from src.models import resolve_model_list

    if spec is None:
        return list(config.TFM_MODELS)
    text = spec if isinstance(spec, str) else ",".join(spec)
    if text.strip().lower() in ("", "none"):
        return []
    try:
        names = resolve_model_list(text)
    except ValueError as exc:
        raise AnalysisArgumentError(f"--models: {exc}") from exc
    ignored = [m for m in names if m not in config.TFM_MODELS and m != BASE_MODEL]
    if ignored:
        log.warning(f"--models: pomijam modele klasyczne {', '.join(ignored)} — eksperyment B łączy LightGBM "
                    f"z modelami fundamentalnymi (zestaw klasyczny jest liczony jako odniesienie)")
    return [m for m in config.TFM_MODELS if m in names]


def analysis_path(feature_set: str, tag: str, runs_dir: Path | str | None = None) -> Path:
    """runs/analyses/tfm_blend__<cechy>__<tag>/ (poza common.ANALYSIS_KINDS — ścieżka budowana tutaj)."""
    return analyses_root(runs_dir) / f"{ANALYSIS}__{feature_set}__{tag}"


def decision_options(cfg: BlendConfig) -> Any:
    from src.evaluation.report_sections import DecisionOptions

    return DecisionOptions(main_rule=MAIN_RULE, rules=list(RULES), target=float(cfg.target), guarantee=False,
                           safety_net=bool(cfg.safety_net), population=cfg.fit_population)


def validate_config(cfg: BlendConfig) -> BlendConfig:
    if int(cfg.repeats) < 1:
        raise AnalysisArgumentError(f"--repeats musi być ≥ 1, podano {cfg.repeats}")
    if int(cfg.n_boot) < 1:
        raise AnalysisArgumentError(f"--bootstrap musi być ≥ 1 (bramka wymaga przedziału ufności), podano {cfg.n_boot}")
    if not 0 < float(cfg.target) < 1:
        raise AnalysisArgumentError(f"--target-undertriage musi być w (0, 1), podano {cfg.target}")
    if cfg.fit_population not in ("vc", "all"):
        raise AnalysisArgumentError(f"--fit-population: vc | all, podano {cfg.fit_population!r}")
    cfg.methods = parse_methods(cfg.methods)
    return cfg


# ─────────────────────────────────────────
# Wybór runów
# ─────────────────────────────────────────
def _is_weighted(weights: Any) -> bool:
    return weights not in (None, "none") and not (isinstance(weights, float) and math.isnan(weights))


def latest_source_runs(feature_set: str, tag: str, models: Sequence[str],
                       runs_dir: Path | str | None = None) -> tuple[dict[str, Any], list[str]]:
    """
    {model: Run} — najnowszy ukończony, dostrojony, nieważony run każdego modelu z tagiem `tag` (bez refitów
    i runów niestandardowych; runy na próbkach dozwolone — jak w analizach 09–11). Odcisk danych: gdy dane
    są przygotowane, tylko runy z bieżącym odciskiem (inne → nota); bez danych odcisk nie jest sprawdzany
    (eksperyment czyta wyłącznie zapisane predykcje, a ich etykiety porównuje `aligned_probas`).
    """
    from src.data.dataset import NO_DATA_FINGERPRINT, data_fingerprint
    from src.ensemble.runs import list_runs, load_run, resolve_feature_set

    fs = resolve_feature_set(feature_set)
    table = list_runs(runs_dir)
    notes: list[str] = []
    if table.empty:
        return {}, notes
    table = table[(table["feature_set"] == fs) & (table["tag"] == tag) & table["done"]
                  & (table["data_version"] == config.DATA_VERSION) & table["model"].isin(list(models))]
    current = data_fingerprint()
    chosen: dict[str, Any] = {}
    stale: list[str] = []
    for name in table["run"]:                      # kolejność nazw = kolejność czasu → ostatni wygrywa
        run = load_run(name, runs_dir)
        info = run.info
        if info.get("custom") or info.get("refit_of") or not info.get("tuned"):
            continue
        if _is_weighted(info.get("weights_strategy", info.get("sample_weight_strategy"))):
            continue
        if not (run.path / "predictions_val.parquet").exists():
            continue
        if current != NO_DATA_FINGERPRINT and info.get("data_fingerprint") != current:
            stale.append(name)
            continue
        chosen[run.model] = run
    if stale:
        notes.append(f"pominięto runy z innych danych (odcisk ≠ {current}): {', '.join(stale)}")
    if current == NO_DATA_FINGERPRINT:
        notes.append("brak przygotowanych danych — odcisk runów nie był porównany z bieżącymi danymi")
    return {m: chosen[m] for m in models if m in chosen}, notes


def select_runs(cfg: BlendConfig) -> Selection:
    """Runy eksperymentu albo `BlendSkipped` (brak lightgbm / TFM) / `AnalysisArgumentError` (złe --runs)."""
    from src.ensemble.runs import load_run

    tfm_wanted = parse_tfm_models(cfg.models)
    if not tfm_wanted:
        raise BlendSkipped("--models nie zawiera modelu fundamentalnego (tabpfn / tabicl)")
    notes: list[str] = []
    if cfg.runs:
        try:
            runs = [load_run(r, cfg.runs_dir) for r in cfg.runs]
        except FileNotFoundError as exc:
            raise AnalysisArgumentError(str(exc)) from exc
        models = [r.model for r in runs]
        duplicates = sorted({m for m in models if models.count(m) > 1})
        if duplicates:
            raise AnalysisArgumentError(f"--runs: model występuje kilka razy: {duplicates} — jeden run na model")
        other = [r.name for r in runs if r.model != BASE_MODEL and r.model not in config.TFM_MODELS]
        if other:
            raise AnalysisArgumentError(f"--runs: tylko lightgbm i modele fundamentalne ({', '.join(config.TFM_MODELS)}); "
                                        f"poza nimi: {', '.join(other)}")
        found = {r.model: r for r in runs}
        if BASE_MODEL not in found:
            raise BlendSkipped("--runs bez runu lightgbm (punkt odniesienia eksperymentu B)")
        tag = str(cfg.tag or found[BASE_MODEL].info.get("tag") or "custom")
    else:
        tag = cfg.tag or default_tag()
        found, notes = latest_source_runs(cfg.feature_set, tag, [BASE_MODEL, *tfm_wanted], cfg.runs_dir)
        if BASE_MODEL not in found:
            raise BlendSkipped(f"brak ukończonego runu lightgbm z tagiem '{tag}' (z tuningiem, bez wag klas, "
                               f"bieżące dane) — najpierw: python scripts/run_all.py --profile {tag.split('-')[0]}")
    tfms = {m: found[m] for m in config.TFM_MODELS if m in found and m in tfm_wanted}
    if not tfms:
        raise BlendSkipped(f"brak ukończonego runu TFM ({', '.join(tfm_wanted)}) z tagiem '{tag}' — najpierw: "
                           f"python scripts/run_all.py --profile {tag.split('-')[0]} --models tfm")
    if len(tfms) > MAX_TFMS:
        raise AnalysisArgumentError(f"Najwyżej {MAX_TFMS} modele fundamentalne w zestawie")
    selection = Selection(lightgbm=found[BASE_MODEL], tfms=tfms, tag=tag, notes=notes)
    if cfg.reference:
        classical = [m for m in config.CLASSICAL_MODELS if m != BASE_MODEL]
        refs, _ = latest_source_runs(cfg.feature_set, tag, classical, cfg.runs_dir)
        selection.references = refs
    return selection


# ─────────────────────────────────────────
# Dane VAL i warianty
# ─────────────────────────────────────────
def load_val(selection: Selection) -> tuple[ValData, Selection]:
    """Predykcje VAL członków (+ odniesień, jeśli mają te same wiersze — inaczej odniesienia pominięte z notą)."""
    from src.ensemble.runs import aligned_meta, aligned_probas

    members = list(selection.members.values())

    def aligned(runs: list[Any]) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None, Any]:
        y, probas, vc = aligned_probas(runs, "val", return_vc=True)
        return y, probas, vc, aligned_meta(runs, "val")

    try:
        y, probas, vc, meta = aligned(members)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise AnalysisArgumentError(f"Predykcje VAL runów eksperymentu nie są wyrównane: {exc}") from exc
    if selection.references:
        try:
            y2, probas2, vc2, meta2 = aligned(members + list(selection.references.values()))
            y, probas, vc, meta = y2, probas2, vc2, meta2
        except (ValueError, FileNotFoundError, KeyError) as exc:
            selection.notes.append(f"odniesienie klasyczne pominięte — runy mają inne wiersze VAL ({exc})")
            selection.references = {}
    site = meta["site"].astype(str).to_numpy() if "site" in meta.columns else None
    return ValData(y=np.asarray(y).astype(int), probas=probas, vc=vc, site=site,
                   row_ids=meta.index.to_numpy()), selection


def set_id(members: Sequence[str]) -> str:
    return "+".join(members)


def build_variants(selection: Selection, methods: Sequence[str]) -> list[Variant]:
    """{lgbm}; {lgbm}+podzbiory TFM (każda metoda); TFM same; zestaw klasyczny (średnia) — gdy ≥ 2 modele."""
    tfms = [m for m in config.TFM_MODELS if m in selection.tfms]
    variants = [Variant(BASELINE, BASELINE, (BASE_MODEL,), "single", "baseline")]
    subsets = [[t] for t in tfms] + ([tfms] if len(tfms) == 2 else [])
    for subset in subsets:
        members = (BASE_MODEL, *subset)
        for method in methods:
            variants.append(Variant(f"{set_id(members)}|{method}", set_id(members), members, method, "candidate"))
    for t in tfms:
        variants.append(Variant(t, t, (t,), "single", "tfm_only"))
    classical = [m for m in config.CLASSICAL_MODELS if m == BASE_MODEL or m in selection.references]
    if len(classical) >= 2:
        variants.append(Variant(f"{CLASSICAL_SET}|mean", CLASSICAL_SET, tuple(classical), "mean", "reference"))
    return variants


def variant_label(v: Variant | Mapping[str, Any]) -> str:
    members = v.members if isinstance(v, Variant) else tuple(v["members"])
    method = v.method if isinstance(v, Variant) else v["method"]
    sid = v.set_id if isinstance(v, Variant) else v["set"]
    if sid == CLASSICAL_SET:
        name = f"zestaw klasyczny ({len(members)} modeli)"
    else:
        name = " + ".join(DISPLAY.get(m, m) for m in members)
    return name + (f" ({METHOD_PL[method]})" if method not in ("single",) else "")


def set_label(sid: str) -> str:
    if sid == CLASSICAL_SET:
        return "zestaw klasyczny"
    return " + ".join(DISPLAY.get(m, m) for m in sid.split("+"))


def fold_strata(y: np.ndarray, site: np.ndarray | None) -> np.ndarray:
    """Etykiety warstw foldów: ESI × oddział; warstwa rzadsza niż liczba foldów → samo ESI."""
    base = np.asarray(y).astype(int).astype(str)
    if site is None:
        return base
    labels = np.char.add(np.char.add(base, "_"), np.asarray(site).astype(str))
    values, counts = np.unique(labels, return_counts=True)
    rare = values[counts < N_FOLDS]
    return np.where(np.isin(labels, rare), base, labels) if len(rare) else labels


def repeat_seed(seed: int, repeat: int) -> int:
    return int(seed) + 1_000 * int(repeat)


# ─────────────────────────────────────────
# Cross-fitting jednego powtórzenia
# ─────────────────────────────────────────
@dataclass
class RepeatPreds:
    """Predykcje cross-fitted jednego powtórzenia: proba[wariant] (n, 5), decisions[wariant][reguła] (n,)."""

    proba: dict[str, np.ndarray]
    decisions: dict[str, dict[str, np.ndarray]]


def crossfit_repeat(data: ValData, variants: Sequence[Variant], seed: int, dopts: Any,
                    ) -> tuple[RepeatPreds, list[dict[str, Any]]]:
    """
    Dwukrotny cross-fitting: dla każdej połowy — wagi, temperatura i reguły dopasowane na drugiej połowie.
    Zwraca predykcje i decyzje każdego wariantu na WSZYSTKICH wierszach VAL (każdy wiersz oceniony przez
    receptury, które go nie widziały) oraz rekordy foldów (wagi, T, parametry reguły progi T).
    """
    from src.ensemble import apply_temperature, blend, fit_temperature, optimize_weights
    from src.evaluation import report_sections as rs
    from src.evaluation.statistics import stratified_folds

    folds = stratified_folds(fold_strata(data.y, data.site), N_FOLDS, seed)
    n = data.n
    proba = {v.id: np.zeros((n, config.N_CLASSES)) for v in variants}
    decisions = {v.id: {r: np.full(n, -1, dtype=np.int64) for r in RULES} for v in variants}
    records: list[dict[str, Any]] = []
    for f in range(N_FOLDS):
        held = folds == f
        fit = ~held
        y_fit = data.y[fit]
        vc_fit = data.vc[fit] if data.vc is not None else None
        for v in variants:
            p_fit = {m: data.probas[m][fit] for m in v.members}
            if len(v.members) == 1:
                weights = {v.members[0]: 1.0}
                method = "mean"
            else:
                method = v.method
                weights = optimize_weights(p_fit, y_fit, method=method)
            blended = blend(p_fit, weights, method)
            temperature = fit_temperature(blended, y_fit)
            try:
                fitted = rs.fit_rules(y_fit, apply_temperature(blended, temperature), vc_fit, dopts)
            except ValueError as exc:
                raise RuntimeError(f"{v.id}, fold {f}: reguł decyzyjnych nie da się dopasować ({exc})") from exc
            p_held = apply_temperature(blend({m: data.probas[m][held] for m in v.members}, weights, method),
                                       temperature)
            proba[v.id][held] = p_held
            preds = rs.predict_rules(p_held, fitted)
            for rule in RULES:
                decisions[v.id][rule][held] = preds[rule]
            records.append({"variant": v.id, "fold": f, "n_fit": int(fit.sum()), "n_eval": int(held.sum()),
                            "weights": {m: float(w) for m, w in weights.items()},
                            "temperature": float(temperature), "target_rule": fitted.specs.get(MAIN_RULE),
                            "fit_population": fitted.population, "fit_note": fitted.note})
    return RepeatPreds(proba=proba, decisions=decisions), records


def save_preds(path: Path, preds: RepeatPreds, variants: Sequence[Variant], row_ids: np.ndarray) -> Path:
    """preds/repeat<k>.npz (float32 prawdopodobieństwa, int8 decyzje) — zapis atomowy."""
    ids = [v.id for v in variants]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh, variants=np.array(json.dumps(ids)), row_ids=np.asarray(row_ids),
            proba=np.stack([preds.proba[i] for i in ids]).astype(np.float32),
            decisions=np.stack([np.stack([preds.decisions[i][r] for r in RULES]) for i in ids]).astype(np.int8))
    replace_with_retry(tmp, path)
    return path


def load_preds(path: Path, variants: Sequence[Variant], row_ids: np.ndarray) -> RepeatPreds | None:
    """Predykcje zapisanego powtórzenia albo None (brak pliku / inne warianty / inne wiersze)."""
    try:
        with np.load(path, allow_pickle=False) as npz:
            ids = json.loads(str(npz["variants"]))
            if ids != [v.id for v in variants] or not np.array_equal(npz["row_ids"], np.asarray(row_ids)):
                return None
            proba = npz["proba"].astype(np.float64)
            decisions = npz["decisions"].astype(np.int64)
    except (OSError, ValueError, KeyError):
        return None
    proba /= proba.sum(axis=2, keepdims=True)
    return RepeatPreds(proba={i: proba[k] for k, i in enumerate(ids)},
                       decisions={i: {r: decisions[k, j] for j, r in enumerate(RULES)} for k, i in enumerate(ids)})


# ─────────────────────────────────────────
# Metryki, bootstrap i Δ
# ─────────────────────────────────────────
def _classwise_terms(y: np.ndarray, proba: np.ndarray, k: int, n_bins: int = N_BINS) -> Any:
    """Rzadka macierz (n × biny): p_k − 1[y = k] w binie p_k — ECE klasy k = Σ_b |W·kolumna_b| / ΣW."""
    from scipy import sparse

    score = np.asarray(proba[:, k], dtype=float)
    idx = np.clip(np.floor(score * n_bins).astype(int), 0, n_bins - 1)
    values = score - (np.asarray(y) == k).astype(float)
    return sparse.csr_matrix((values, (np.arange(len(score)), idx)), shape=(len(score), n_bins))


def classwise_ece_samples(y: np.ndarray, probas: Mapping[str, np.ndarray], n_boot: int, seed: int,
                          ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]]]:
    """
    ECE ESI 1 / ESI 5 (jeden-przeciw-reszcie, biny równe — jak `statistics.ece_classwise`): wartości na
    całym zbiorze i w losowaniach `statistics.resample_weights` (te same losowania co `bootstrap_decisions`).
    """
    from src.evaluation.statistics import resample_weights

    terms = {name: {key: _classwise_terms(y, p, k) for key, k in CLASSWISE_ECE.items()} for name, p in probas.items()}
    point = {name: {key: float(np.abs(np.asarray(t.sum(axis=0)).ravel()).sum() / len(y)) for key, t in d.items()}
             for name, d in terms.items()}
    parts: dict[str, dict[str, list[np.ndarray]]] = {name: {key: [] for key in CLASSWISE_ECE} for name in probas}
    for counts in resample_weights(y, n_boot, seed, stratify=True):      # krotności wierszy (b × n)
        total = counts.sum(axis=1)
        for name, d in terms.items():
            for key, t in d.items():
                gaps = np.asarray((t.T @ counts.T).T)            # (b × biny)
                parts[name][key].append(np.abs(gaps).sum(axis=1) / total)
    samples = {name: {key: np.concatenate(v) for key, v in d.items()} for name, d in parts.items()}
    return point, samples


def _flat_key(rule: str, metric: str) -> str:
    return metric if rule == "*" else f"{_STAT_NAMES[metric]}_{rule}"


def repeat_bootstrap(y: np.ndarray, preds: RepeatPreds, variants: Sequence[Variant], n_boot: int, seed: int,
                     alpha: float = ALPHA) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]]]:
    """Wartości (cały VAL) i losowania bootstrap każdej metryki każdego wariantu w jednym powtórzeniu."""
    from src.evaluation import report_sections as rs

    ids = [v.id for v in variants]
    res = rs.bootstrap_decisions(y, {i: preds.proba[i] for i in ids}, {i: preds.decisions[i] for i in ids},
                                 BOOT_METRICS, n_boot=n_boot, alpha=alpha, seed=seed, n_bins=N_BINS)
    point: dict[str, dict[str, float]] = {}
    samples: dict[str, dict[str, np.ndarray]] = {}
    for i in ids:
        point[i] = {_flat_key(r, m): float(v) for (r, m), v in res.point[i].items()}
        samples[i] = {_flat_key(r, m): np.asarray(v, dtype=float) for (r, m), v in res.samples[i].items()}
    cw_point, cw_samples = classwise_ece_samples(y, {i: preds.proba[i] for i in ids}, n_boot, seed)
    for i in ids:
        point[i].update(cw_point[i])
        samples[i].update(cw_samples[i])
        point[i]["auc_macro"] = float(proba_metrics(y, preds.proba[i])["auc_macro"])
    return point, samples


def subgroup_point(y: np.ndarray, preds: RepeatPreds, variant_id: str, mask: np.ndarray) -> dict[str, float]:
    """Metryki podgrupy (np. vc): reguły + log loss / AUC (bez bootstrapu)."""
    out: dict[str, float] = {}
    for rule in RULES:
        for name, value in rule_metrics(y[mask], preds.decisions[variant_id][rule][mask]).items():
            out[f"{name}_{rule}"] = value
    out.update(proba_metrics(y[mask], preds.proba[variant_id][mask]))
    return out


def delta_summary(delta: float | None, samples: np.ndarray, alpha: float, higher_is_better: bool) -> dict[str, Any]:
    """
    Przedział i p z tych samych ogonów rozkładu Δ (definicja `statistics._paired_summary`: statystyki
    pozycyjne c = ⌈B·α/2⌉, p = min(1, 2·min(P(Δ ≤ 0), P(Δ ≥ 0))), najmniej 1/B). `better`: 'zestaw' /
    'LightGBM' / 'brak istotnej różnicy' z uwzględnieniem kierunku metryki.
    """
    valid = np.asarray(samples, dtype=float)
    valid = valid[np.isfinite(valid)]
    if delta is None or not np.isfinite(delta) or len(valid) == 0:
        return {"delta": None if delta is None or not np.isfinite(delta) else float(delta), "ci_low": None,
                "ci_high": None, "p_two_sided": None, "n_valid": int(len(valid)), "better": "nieokreślone"}
    b = len(valid)
    ordered = np.sort(valid)
    c = max(1, int(np.ceil(round(b * alpha / 2, 9))))
    low, high = float(ordered[c - 1]), float(ordered[b - c])
    n_le, n_ge = int(np.sum(valid <= 0)), int(np.sum(valid >= 0))
    p = float(min(1.0, max(2.0 * min(n_le, n_ge) / b, 1.0 / b)))
    if min(n_le, n_ge) >= c:
        better = "brak istotnej różnicy"
    else:
        a_higher = n_le < c
        better = "zestaw" if a_higher == higher_is_better else "LightGBM"
    return {"delta": float(delta), "ci_low": low, "ci_high": high, "p_two_sided": p, "n_valid": int(b),
            "better": better}


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def _sd(values: Sequence[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else None


def aggregate(points: Sequence[Mapping[str, Mapping[str, float]]], samples: Sequence[Mapping[str, Mapping[str, np.ndarray]]],
              variants: Sequence[Variant], alpha: float = ALPHA,
              ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    """
    Średnie po powtórzeniach: metryki każdego wariantu ({mean, sd_repeats, ci_low, ci_high, values}),
    Δ względem LightGBM ({delta, ci_low, ci_high, p, better, sd_repeats}) oraz uśrednione losowania
    (do porównań par wariantów, np. para TFM vs najlepszy pojedynczy).
    """
    mean_samples: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for v in variants:
        mean_samples[v.id] = {}
        metrics[v.id] = {}
        for key in METRIC_KEYS:
            values = [p[v.id].get(key) for p in points]
            stack = np.stack([s[v.id][key] for s in samples]) if all(key in s[v.id] for s in samples) else None
            avg = stack.mean(axis=0) if stack is not None else np.array([])
            mean_samples[v.id][key] = avg
            finite = avg[np.isfinite(avg)]
            lo, hi = (np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(finite)
                      else (None, None))
            metrics[v.id][key] = {"mean": _mean(values), "sd_repeats": _sd(values),
                                  "ci_low": None if lo is None else float(lo), "ci_high": None if hi is None else float(hi),
                                  "values": [None if x is None or not np.isfinite(x) else float(x) for x in values]}
    deltas: dict[str, dict[str, Any]] = {}
    for v in variants:
        if v.id == BASELINE:
            continue
        deltas[v.id] = {}
        for key in METRIC_KEYS:
            point = metrics[v.id][key]["mean"]
            base = metrics[BASELINE][key]["mean"]
            delta = None if point is None or base is None else point - base
            per_repeat = [None if a is None or b is None else a - b
                          for a, b in zip(metrics[v.id][key]["values"], metrics[BASELINE][key]["values"], strict=True)]
            d = mean_samples[v.id][key] - mean_samples[BASELINE][key] if len(mean_samples[v.id][key]) else np.array([])
            deltas[v.id][key] = {**delta_summary(delta, d, alpha, key in HIGHER_IS_BETTER),
                                 "sd_repeats": _sd(per_repeat)}
    return metrics, deltas, mean_samples


# ─────────────────────────────────────────
# Szum seedów, bramka, oszczędność, rekomendacja
# ─────────────────────────────────────────
def lightgbm_seed_sd(feature_set: str, tag: str, lgbm_run: Any, runs_dir: Path | str | None = None,
                     ) -> dict[str, Any]:
    """
    SD log loss LightGBM na VAL między seedami (analiza C1, `09_seed_refits.py`) — `report_sections.seed_sd`
    z split='val' (domyślny split helpera to test). Tylko wyniki seedów liczone z TEGO runu lightgbm i tych
    samych danych (`load_analyses` z run_names). Zwraca {"sd", "n", "source", "reason"}.
    """
    from src.evaluation import report_sections as rs

    loaded = rs.load_analyses(feature_set, tag, runs_dir, fingerprint=lgbm_run.info.get("data_fingerprint"),
                              run_names={BASE_MODEL: lgbm_run.name})
    seeds = loaded.get("seeds") or {}
    if seeds.get("skipped"):
        return {"sd": None, "n": None, "source": None, "reason": seeds["skipped"]}
    sd = rs.seed_sd(seeds, [BASE_MODEL], "log_loss", "argmax", split="val")
    stat = ((((seeds.get("results") or {}).get("models") or {}).get(BASE_MODEL) or {}).get("val") or {}).get("log_loss") or {}
    if sd is None:
        reason = ("brak SD log loss lightgbm na VAL w analizie seedów (potrzebne ≥ 2 wartości z tego runu"
                  + (", lightgbm usunięty jako nieaktualny" if seeds.get("stale_models") else "") + ")")
        return {"sd": None, "n": stat.get("n"), "source": seeds.get("path"), "reason": reason}
    return {"sd": float(sd), "n": stat.get("n"), "source": seeds.get("path"), "reason": None}


def seeds_command(feature_set: str, tag: str) -> str:
    return f"python scripts/09_seed_refits.py --feature-set {feature_set} --tag {tag} --models lightgbm"


def evaluate_gate(variant: Variant, deltas: Mapping[str, Mapping[str, Any]], seed_sd: float | None,
                  mid: float) -> dict[str, Any]:
    """Pięć kryteriów bramki dla wariantu z TFM → {criteria, status, superiority}."""
    from src.evaluation import report_sections as rs

    ll = deltas["log_loss"]
    d, hi = ll.get("delta"), ll.get("ci_high")

    def status(ok: bool | None) -> str:
        return NOT_CHECKED if ok is None else (PASS if ok else FAIL)

    under = (deltas.get(f"undertriage_{MAIN_RULE}") or {}).get("delta")
    ece1 = (deltas.get("ece_esi1") or {}).get("delta")
    criteria = [
        {"key": "delta_log_loss", "value": d, "threshold": -mid, "status": status(None if d is None else d <= -mid)},
        {"key": "ci_upper", "value": hi, "threshold": 0.0, "status": status(None if hi is None else hi < 0)},
        {"key": "seed_noise", "value": None if d is None else abs(d),
         "threshold": None if seed_sd is None else rs.SEED_SD_FACTOR * seed_sd,
         "status": status(None if d is None or seed_sd is None else abs(d) > rs.SEED_SD_FACTOR * seed_sd)},
        {"key": "undertriage_target", "value": under, "threshold": GATE_MAX_UNDERTRIAGE_WORSE,
         "status": status(None if under is None else under <= GATE_MAX_UNDERTRIAGE_WORSE)},
        {"key": "ece_esi1", "value": ece1, "threshold": GATE_MAX_ECE_ESI1_WORSE,
         "status": status(None if ece1 is None else ece1 <= GATE_MAX_ECE_ESI1_WORSE)},
    ]
    states = {c["status"] for c in criteria}
    overall = FAILED if FAIL in states else (INCONCLUSIVE if NOT_CHECKED in states else PASSED)
    return {"variant": variant.id, "set": variant.set_id, "method": variant.method, "members": list(variant.members),
            "n_tfm": sum(m in config.TFM_MODELS for m in variant.members), "criteria": criteria, "status": overall,
            "superiority": rs.superiority("log_loss", d, ll.get("ci_low"), hi, seed_sd,
                                          name_a=set_label(variant.set_id), name_b="LightGBM")}


def best_variants(variants: Sequence[Variant], metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Variant]:
    """Najlepsza metoda (najniższy średni log loss cross-fitted) każdego zestawu kandydackiego."""
    best: dict[str, Variant] = {}
    for v in variants:
        if v.kind != "candidate":
            continue
        value = metrics[v.id]["log_loss"]["mean"]
        current = best.get(v.set_id)
        if value is None:
            continue
        if current is None or value < metrics[current.id]["log_loss"]["mean"]:
            best[v.set_id] = v
    return best


def choose(gates: Mapping[str, Mapping[str, Any]], deltas: Mapping[str, Mapping[str, Any]],
           mean_samples: Mapping[str, Mapping[str, np.ndarray]], best: Mapping[str, Variant], mid: float,
           alpha: float = ALPHA) -> dict[str, Any]:
    """
    Reguła oszczędności: para TFM wybierana tylko, gdy przeszła bramkę i bije najlepszy przechodzący zestaw
    z jednym TFM o ≥ MID (Δ log loss punktowo; przedział sparowany raportowany). Brak przechodzących →
    LightGBM sam. Zwraca {"chosen_set", "variant", "pair_vs_single", "reason"}.
    """
    passing = {sid: g for sid, g in gates.items() if g["status"] == PASSED}
    singles = sorted((sid for sid, g in passing.items() if g["n_tfm"] == 1),
                     key=lambda s: deltas[best[s].id]["log_loss"]["delta"])
    pairs = [sid for sid, g in passing.items() if g["n_tfm"] == 2]
    best_single = singles[0] if singles else None
    out: dict[str, Any] = {"best_single": best_single, "pair": pairs[0] if pairs else None, "pair_vs_single": None,
                           "pair_chosen": False}
    if pairs and best_single:
        pair, single = best[pairs[0]], best[best_single]
        d_pair = deltas[pair.id]["log_loss"]["delta"]
        d_single = deltas[single.id]["log_loss"]["delta"]
        samples = mean_samples[pair.id]["log_loss"] - mean_samples[single.id]["log_loss"]
        comparison = delta_summary(d_pair - d_single, samples, alpha, higher_is_better=False)
        out["pair_vs_single"] = comparison
        out["pair_chosen"] = bool(d_pair - d_single <= -mid)
        if out["pair_chosen"]:
            chosen = pairs[0]
            reason = (f"para TFM przeszła bramkę i bije najlepszy pojedynczy zestaw ({set_label(best_single)}) "
                      f"o {d_pair - d_single:+.4f} log loss (≤ −MID {mid:g})")
        else:
            chosen = best_single
            reason = (f"para TFM przeszła bramkę, ale różnica względem {set_label(best_single)} "
                      f"({d_pair - d_single:+.4f} log loss) nie osiąga −MID {mid:g} — reguła oszczędności: "
                      f"jeden TFM")
    elif pairs:
        chosen = pairs[0]
        out["pair_chosen"] = True
        reason = "przeszła tylko para TFM (żaden zestaw z jednym TFM nie przeszedł bramki)"
    elif best_single:
        chosen = best_single
        reason = (f"{set_label(best_single)} przeszedł bramkę" + (f" (najniższy Δ log loss spośród: "
                  f"{', '.join(set_label(s) for s in singles)})" if len(singles) > 1 else ""))
    else:
        chosen = BASELINE
        inconclusive = [sid for sid, g in gates.items() if g["status"] == INCONCLUSIVE]
        reason = ("żaden zestaw z TFM nie przeszedł bramki — LightGBM bez modeli fundamentalnych"
                  + (f" (nierozstrzygnięte: {', '.join(set_label(s) for s in inconclusive)} — brak SD seedów)"
                     if inconclusive else ""))
    out.update({"chosen_set": chosen, "variant": best[chosen].id if chosen in best else BASELINE, "reason": reason})
    return out


def _decision_flags(cfg: BlendConfig) -> list[str]:
    """Opcje reguł 08 różne od domyślnych (08 dopasuje regułę tak samo jak ten eksperyment)."""
    flags: list[str] = []
    if abs(float(cfg.target) - float(config.DECISION_TARGET_UNDERTRIAGE)) > 1e-12:
        flags += ["--target-undertriage", f"{float(cfg.target):g}"]
    if cfg.fit_population != config.DECISION_FIT_POPULATION:
        flags += ["--fit-population", cfg.fit_population]
    if not cfg.safety_net:
        flags.append("--no-safety-net")
    return flags


def recommendation(cfg: BlendConfig, selection: Selection, choice: Mapping[str, Any], best: Mapping[str, Variant],
                   gates: Mapping[str, Any], seed_info: Mapping[str, Any], complete: bool) -> dict[str, Any]:
    """recommendation.json: członkowie, runy, metoda i DOKŁADNE polecenie 08 (VAL → TEST raz)."""
    chosen = choice["chosen_set"]
    variant = best.get(chosen)
    members = list(variant.members) if variant is not None else [BASE_MODEL]
    method = variant.method if variant is not None else "mean"
    runs = [selection.members[m] for m in members]
    names = [r.name for r in runs]
    fs = cfg.feature_set
    has_tfm = any(m in config.TFM_MODELS for m in members)
    flags = _decision_flags(cfg)
    command = (["python", "scripts/08_final_report.py", "--feature-set", fs, "--runs", *names,
                "--weights", "optimize", "--method", method] + flags)
    sampled = any(r.info.get("train_sample") or r.info.get("eval_sample") for r in runs)
    command_tag = (["python", "scripts/08_final_report.py", "--feature-set", fs, "--tag", selection.tag,
                    "--models", ",".join(members)] + (["--include-pretrained"] if has_tfm else [])
                   + (["--include-samples"] if sampled else []) + ["--weights", "optimize", "--method", method] + flags)
    return {
        "analysis": ANALYSIS,
        "created": now_iso(),
        "feature_set": fs,
        "tag": selection.tag,
        "complete": bool(complete),
        "decision": chosen,
        "decision_label": "LightGBM sam (bez TFM)" if chosen == BASELINE else set_label(chosen),
        "members": members,
        "runs": names,
        "includes_tfm": has_tfm,
        "method": method,
        "weights": "optimize",
        "include_pretrained": has_tfm,
        "command": " ".join(shlex.quote(c) for c in command),
        "command_args": command,
        "command_tag": " ".join(shlex.quote(c) for c in command_tag),
        "reason": choice["reason"],
        "gate": {sid: {"status": g["status"], "variant": g["variant"]} for sid, g in gates.items()},
        "noise_criterion_checked": seed_info.get("sd") is not None,
        "seed_sd": seed_info.get("sd"),
        "note": ("VAL only — zbiór testowy oceń raz poleceniem `command` (08 ponownie dopasuje wagi, temperaturę "
                 "i reguły na całym VAL)" + ("; 08 zapisze też backend_ensemble_weights_no_tfm.json" if has_tfm else "")),
    }


# ─────────────────────────────────────────
# Podsumowania foldów i różnorodności
# ─────────────────────────────────────────
def fold_summary(records: Sequence[Mapping[str, Any]], variants: Sequence[Variant]) -> dict[str, Any]:
    """Wagi członków i temperatura każdego wariantu w foldach: {mean, sd, min, max, values}."""
    out: dict[str, Any] = {}
    for v in variants:
        rows = [r for r in records if r["variant"] == v.id]
        if not rows:
            continue

        def stats(values: list[float]) -> dict[str, Any]:
            arr = np.asarray(values, dtype=float)
            return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1 else None,
                    "min": float(arr.min()), "max": float(arr.max()), "values": [float(x) for x in arr]}

        out[v.id] = {
            "weights": {m: stats([r["weights"].get(m, 0.0) for r in rows]) for m in v.members},
            "temperature": stats([r["temperature"] for r in rows]),
            "n_folds": len(rows),
        }
    return out


def diversity_rows(data: ValData, models: Sequence[str]) -> list[dict[str, Any]]:
    from src.ensemble import diversity

    table = diversity({m: data.probas[m] for m in models if m in data.probas}, data.y)
    return [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}
            for row in table.to_dict(orient="records")]


# ─────────────────────────────────────────
# Wykresy
# ─────────────────────────────────────────
def _marker(kind: str, n_tfm: int) -> tuple[str, bool]:
    """(znacznik, wypełniony) — kodowanie rodzaju wariantu niezależne od koloru."""
    if kind == "reference":
        return "s", True
    if kind == "tfm_only":
        return "o", False
    return ("D", True) if n_tfm == 2 else ("o", True)


def plot_deltas(results: Mapping[str, Any], path: Path) -> Path | None:
    """
    Δ względem LightGBM (95% CI) zestawów kandydackich (i odniesienia klasycznego): log loss, niedotriażowanie
    (progi T, w pp) i ECE ESI 1 — z progami bramki. TFM same są tylko w tabeli (inna skala).
    """
    from src.evaluation import visualizations  # noqa: F401, I001 — najpierw backend Agg
    import matplotlib.pyplot as plt

    variants = [v for v in results["variants"] if v["kind"] in ("candidate", "reference")]
    if not variants:
        return None
    chosen = (results.get("parsimony") or {}).get("variant")
    panels = [("log_loss", "Δ log loss (↓ lepiej)", -results["gate_spec"]["mid_log_loss"], "−MID", 1.0),
              (f"undertriage_{MAIN_RULE}", "Δ niedotriażowanie, progi T [pp] (↓ lepiej)",
               100 * GATE_MAX_UNDERTRIAGE_WORSE, "+1 pp", 100.0),
              ("ece_esi1", "Δ ECE ESI 1 (↓ lepiej)", GATE_MAX_ECE_ESI1_WORSE, "+0.01", 1.0)]
    height = max(2.6, 0.42 * len(variants) + 1.4)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), height), sharey=True)
    ys = np.arange(len(variants))[::-1]
    for ax, (key, title, threshold, th_label, scale) in zip(np.atleast_1d(axes), panels, strict=True):
        ax.axvline(0.0, color=INK_MUTED, lw=1.0)
        ax.axvline(threshold, color=INK_MUTED, lw=1.0, linestyle="--")
        ax.text(threshold, 0.01, f" {th_label}", color=INK_MUTED, fontsize=7, va="bottom",
                transform=ax.get_xaxis_transform())
        for y_pos, v in zip(ys, variants, strict=True):
            d = (results["deltas"].get(v["id"]) or {}).get(key) or {}
            if d.get("delta") is None:
                continue
            color = SET_COLORS.get(v["set"], INK_MUTED)
            n_tfm = sum(m in config.TFM_MODELS for m in v["members"])
            marker, filled = _marker(v["kind"], n_tfm)
            point = scale * d["delta"]
            lo = scale * (d["ci_low"] if d.get("ci_low") is not None else d["delta"])
            hi = scale * (d["ci_high"] if d.get("ci_high") is not None else d["delta"])
            ax.plot([lo, hi], [y_pos, y_pos], color=color, lw=2.0, solid_capstyle="round")
            ax.plot(point, y_pos, marker=marker, ms=8, color=color, mfc=color if filled else "white",
                    mew=1.6, linestyle="none")
        ax.set_title(title, fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    labels = []
    for v in variants:
        gate = (results["gate"].get(v["set"]) or {}) if v["kind"] == "candidate" else {}
        tag = f" — bramka: {STATUS_PL[gate['status']]}" if gate and gate.get("variant") == v["id"] else ""
        if v["kind"] == "reference":
            tag = " — odniesienie"
        labels.append(("★ " if v["id"] == chosen else "") + variant_label(v) + tag)
    first = np.atleast_1d(axes)[0]
    first.set_yticks(ys)
    first.set_yticklabels(labels, fontsize=8, color=INK)
    fig.suptitle(f"Eksperyment B (VAL, cross-fitting {results['config']['repeats']}×{N_FOLDS}): Δ względem LightGBM, "
                 f"95% CI (bootstrap {results['config']['n_boot']}×); ★ = rekomendacja", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_diversity(rows: Sequence[Mapping[str, Any]], models: Sequence[str], path: Path) -> Path | None:
    """Mapy: korelacja błędów argmax i odsetek niezgodności między modelami (surowe predykcje VAL)."""
    from src.evaluation import visualizations  # noqa: F401, I001 — najpierw backend Agg
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    names = [m for m in models if any(m in (r["model_a"], r["model_b"]) for r in rows)]
    if len(names) < 2:
        return None
    # Korelacja (−1…1): rozbieżna niebieski ↔ czerwony z szarym środkiem; niezgodność (0…1): jeden odcień
    diverging = LinearSegmentedColormap.from_list("korelacja", ["#e34948", "#f0efec", "#2a78d6"])
    sequential = LinearSegmentedColormap.from_list("niebieski", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, axes = plt.subplots(1, 2, figsize=(4.2 * 2 + 1.0, 0.55 * len(names) + 2.2))
    for ax, (key, title) in zip(axes, (("corr_errors", "korelacja błędów argmax (−1…1)"),
                                       ("disagreement", "odsetek niezgodnych predykcji argmax")), strict=True):
        mat = np.full((len(names), len(names)), np.nan)
        for r in rows:
            if r["model_a"] in names and r["model_b"] in names and r.get(key) is not None:
                i, j = names.index(r["model_a"]), names.index(r["model_b"])
                mat[i, j] = mat[j, i] = r[key]
        if key == "corr_errors":
            image = ax.imshow(np.ma.masked_invalid(mat), cmap=diverging, vmin=-1.0, vmax=1.0)
        else:
            image = ax.imshow(np.ma.masked_invalid(mat), cmap=sequential, vmin=0.0, vmax=1.0)
        for i in range(len(names)):
            for j in range(len(names)):
                if np.isfinite(mat[i, j]):
                    dark = (abs(mat[i, j]) > 0.6) if key == "corr_errors" else (mat[i, j] > 0.55)
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="#ffffff" if dark else INK)
        labels = [DISPLAY.get(m, m) for m in names]
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8, color=INK)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(labels, fontsize=8, color=INK)
        ax.set_title(title, fontsize=9, color=INK)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Różnorodność modeli na VAL (im niższa korelacja błędów, tym więcej może dać łączenie)",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_weights(folds: Mapping[str, Any], variants: Sequence[Mapping[str, Any]], path: Path,
                 shown_ids: Sequence[str] | None = None) -> Path | None:
    """Wagi członków w kolejnych foldach (2 × powtórzenia): metoda bramki każdego zestawu + odniesienie."""
    from src.evaluation import visualizations  # noqa: F401, I001 — najpierw backend Agg
    import matplotlib.pyplot as plt

    shown = [v for v in variants if len(v["members"]) > 1 and v["id"] in folds
             and (shown_ids is None or v["id"] in shown_ids)]
    if not shown:
        return None
    member_colors = {"lightgbm": "#2a78d6", "tabpfn": "#eb6834", "tabicl": "#1baf7a"}
    cols = min(3, len(shown))
    rows = math.ceil(len(shown) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.3 * rows + 0.5), squeeze=False)
    rng = np.random.default_rng(0)
    for k, ax in enumerate(axes.ravel()):
        if k >= len(shown):
            ax.set_visible(False)
            continue
        v = shown[k]
        entry = folds[v["id"]]["weights"]
        for x, m in enumerate(v["members"]):
            values = np.asarray(entry[m]["values"], dtype=float)
            jitter = rng.uniform(-0.12, 0.12, size=len(values))
            color = member_colors.get(m, INK_MUTED)
            ax.plot(x + jitter, values, "o", ms=6, color=color, mfc=color, alpha=0.85, linestyle="none")
            ax.plot([x - 0.25, x + 0.25], [values.mean()] * 2, color=INK, lw=2.0, solid_capstyle="round")
        ax.set_xticks(range(len(v["members"])))
        ax.set_xticklabels([DISPLAY.get(m, m) for m in v["members"]], rotation=30, ha="right", fontsize=8,
                           color=INK)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(variant_label(v), fontsize=8, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if k % cols == 0:
            ax.set_ylabel("waga w foldzie (linia = średnia)", fontsize=8, color=INK_MUTED)
    fig.suptitle("Stabilność wag połączenia w foldach cross-fittingu (VAL)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ─────────────────────────────────────────
# Raport (po polsku)
# ─────────────────────────────────────────
def _f(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _signed(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{float(value):+.{digits}f}"


def _pp(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{100 * float(value):+.2f} pp"


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{100 * float(value):.2f}%"


def _delta_ci(d: Mapping[str, Any] | None, digits: int = 4, pp: bool = False) -> str:
    if not d or d.get("delta") is None:
        return "—"
    fmt = _pp if pp else (lambda x: _signed(x, digits))
    if d.get("ci_low") is None:
        return fmt(d["delta"])
    return f"{fmt(d['delta'])} [{fmt(d['ci_low'])}; {fmt(d['ci_high'])}]"


def gate_lines(results: Mapping[str, Any]) -> list[str]:
    """Kryteria bramki — wypisywane PRZED wynikami (pre-rejestracja)."""
    spec = results["gate_spec"]
    seed = results["seed_sd"]
    dec = results["config"]["decision"]
    sd_text = (f"SD = {seed['sd']:.4f} (n = {seed.get('n')}), próg 2 × SD = {2 * seed['sd']:.4f}"
               if seed.get("sd") is not None else f"**SD niedostępne** — {seed.get('reason')}")
    return [
        "## 1. Kryteria ustalone z góry (bramka)",
        "",
        "Zapisane przed obliczeniem wyników (stałe w `src/analysis/tfm_blend.py`; ich zmiana = inny eksperyment). "
        "Zestaw z modelem fundamentalnym (LightGBM + TFM) przechodzi bramkę, gdy JEDNOCZEŚNIE:",
        "",
        f"1. Δ log loss (zestaw − LightGBM) ≤ −{spec['mid_log_loss']:g} (minimalna istotna różnica, MID),",
        "2. górna granica 95% przedziału ufności Δ log loss < 0,",
        f"3. |Δ log loss| > {spec['seed_sd_factor']:g} × SD seedów LightGBM (log loss na VAL, analiza seedów C1): "
        f"{sd_text},",
        f"4. niedotriażowanie ESI 1–2 → 3–5 reguły progi T (cel T = {_pct(dec['target'])}) nie gorsze o więcej niż "
        f"{100 * spec['max_undertriage_worse']:g} pp,",
        f"5. ECE ESI 1 (jeden-przeciw-reszcie) nie gorsze o więcej niż {spec['max_ece_esi1_worse']:g}.",
        "",
        "Kryterium 3 bez wyników seedów jest niesprawdzalne → werdykt „nierozstrzygnięte” (zestaw NIE jest "
        f"rekomendowany; uzupełnij: `{seeds_command(results['feature_set'], results['tag'])}`).",
        f"Oszczędność: para TFM (LightGBM + TabPFN + TabICL) jest wybierana tylko, gdy przeszła bramkę i bije "
        f"najlepszy przechodzący zestaw z jednym TFM o ≥ {spec['mid_log_loss']:g} log loss. Najwyżej "
        f"{MAX_TFMS} modele fundamentalne. Metoda łączenia zestawu (średnia / geometryczna) — niższy log loss "
        f"z cross-fittingu; bramkę ocenia się dla tej metody.",
        "Punkt odniesienia: LightGBM sam, przechodzący tę samą procedurę (własna temperatura i własne reguły "
        "dopasowane na połowie dopasowania).",
        "",
    ]


def report_lines(results: Mapping[str, Any]) -> list[str]:
    cfg = results["config"]
    dec = cfg["decision"]
    variants = results["variants"]
    metrics = results["metrics"]
    deltas = results["deltas"]
    rec = results["recommendation"]
    lines = [
        f"# Eksperyment B — LightGBM + modele fundamentalne (TFM) · {results['feature_set']} · tag {results['tag']}",
        "",
        "> **Tylko VAL.** Wszystkie liczby pochodzą ze zbioru walidacyjnego (cross-fitting). Zbiór testowy NIE "
        "był używany — receptura wskazana tutaj jest oceniana na TEST **jeden raz** w `scripts/08_final_report.py`.",
        "",
    ]
    if not results.get("complete"):
        lines += [f"> ⚠ Wyniki NIEPEŁNE: {results['n_done']}/{results['n_units']} powtórzeń "
                  f"(błędy: {len(results.get('failures') or [])}) — uruchom ponownie to samo polecenie.", ""]
    lines += gate_lines(results)

    # 2. Dane i procedura
    lines += ["## 2. Dane i procedura", "",
              "| model | run | profil | próbka trening / ocena |", "|---|---|---|---|"]
    for model, run in {**results["runs"], **results.get("references", {})}.items():
        info = results["run_info"].get(run, {})
        sample = f"{info.get('train_sample') or '—'} / {info.get('eval_sample') or '—'}"
        lines.append(f"| {DISPLAY.get(model, model)} | `{run}` | {info.get('profile') or '—'} | {sample} |")
    counts = results["data"]["esi_counts"]
    lines += [
        "",
        f"- VAL: {results['data']['n']:,} wizyt; ESI 1–5: " + ", ".join(f"{c:,}" for c in counts)
        + (f"; podgrupa vc (komplet HR/SBP/RR): {results['data']['n_vc']:,}" if results["data"].get("n_vc") else ""),
        f"- cross-fitting: {cfg['repeats']} × {N_FOLDS} foldy warstwowane ESI × oddział (seed {cfg['seed']} + "
        f"1000·k); na połowie dopasowania: wagi (`optimize_weights`, min log loss, Nelder–Mead), temperatura "
        f"(`fit_temperature`), reguły argmax / koszt / progi T (T = {_pct(dec['target'])}, siatka bezpieczeństwa "
        f"{'tak' if dec['safety_net'] else 'nie'}, populacja dopasowania {dec['fit_population']}); ocena na drugiej "
        f"połowie; role połówek zamieniane — każdy wiersz VAL oceniony przez recepturę, która go nie widziała,",
        f"- Δ = zestaw − LightGBM: średnia z {cfg['repeats']} powtórzeń; log loss i Brier — średnia różnic per "
        f"wiersz (ci sami pacjenci); 95% CI — sparowany bootstrap warstwowany ESI ({cfg['n_boot']} losowań, te same "
        f"dla wszystkich zestawów i powtórzeń; percentyle, p z tych samych ogonów),",
        "- metody łączenia: " + ", ".join(METHOD_PL[m] for m in cfg["methods"]) + ".",
    ]
    for note in results.get("notes") or []:
        lines.append(f"- uwaga: {note}")
    if any((results["run_info"].get(r) or {}).get("train_sample") or (results["run_info"].get(r) or {}).get("eval_sample")
           for r in results["runs"].values()):
        lines.append("- ⚠ runy na PRÓBKACH (profil smoke / quick) — wyniki tylko do testu, nie do pracy.")
    lines.append("")

    # 3. Wyniki
    lines += [f"## 3. Wyniki: Δ względem LightGBM (średnia z {cfg['repeats']} powtórzeń, 95% CI)", "",
              "| zestaw | log loss | Δ log loss [95% CI] | SD Δ (powt.) | Δ Brier | Δ ECE | Δ ECE ESI 1 | Δ ECE ESI 5 "
              "| Δ niedotriaż. (progi T) | Δ QWK (progi T) | bramka |",
              "|---|---:|---|---:|---|---|---|---|---|---|---|"]
    base_ll = metrics[BASELINE]["log_loss"]["mean"]
    lines.append(f"| {variant_label(variants[0])} (odniesienie) | {_f(base_ll)} | — | — | — | — | — | — | — | — | — |")
    for v in variants[1:]:
        d = deltas[v["id"]]
        gate = results["gate"].get(v["set"]) if v["kind"] == "candidate" else None
        verdict = (STATUS_PL[gate["status"]] if gate and gate["variant"] == v["id"]
                   else ("(druga metoda)" if gate else "odniesienie"))
        lines.append(
            f"| {variant_label(v)} | {_f(metrics[v['id']]['log_loss']['mean'])} | {_delta_ci(d['log_loss'])} | "
            f"{_f(d['log_loss'].get('sd_repeats'))} | {_delta_ci(d['brier'])} | {_delta_ci(d['ece'])} | "
            f"{_delta_ci(d['ece_esi1'])} | {_delta_ci(d['ece_esi5'])} | {_delta_ci(d[f'undertriage_{MAIN_RULE}'], pp=True)} "
            f"| {_delta_ci(d[f'qwk_{MAIN_RULE}'])} | {verdict} |")
    lines += ["", "Ujemna Δ log loss / Brier / ECE / niedotriażowania = zestaw lepszy niż LightGBM; dodatnia Δ QWK = "
              "zestaw lepszy. „SD Δ (powt.)” — rozrzut Δ między powtórzeniami podziału VAL (szum procedury).", ""]

    # 4. Bramka
    lines += ["## 4. Bramka — szczegóły (metoda o niższym log loss w każdym zestawie)", "",
              "| zestaw | metoda | 1. Δ ≤ −MID | 2. CI górna < 0 | 3. \\|Δ\\| > 2 × SD seedów | 4. niedotriaż. ≤ +1 pp "
              "| 5. ECE ESI 1 ≤ +0.01 | reguła C1 (superiority) | wynik |",
              "|---|---|---|---|---|---|---|---|---|"]
    for sid, g in results["gate"].items():
        c = {x["key"]: x for x in g["criteria"]}

        def cell(key: str, pp: bool = False, c: Mapping[str, Any] = c) -> str:
            x = c[key]
            value = _pp(x["value"]) if pp else _signed(x["value"])
            if key == "seed_noise":
                value = f"{_f(x['value'])} vs {_f(x['threshold'])}"
            return f"{CRITERION_PL[x['status']]} {value}"

        lines.append(f"| {set_label(sid)} | {METHOD_PL[g['method']]} | {cell('delta_log_loss')} | {cell('ci_upper')} | "
                     f"{cell('seed_noise')} | {cell('undertriage_target', pp=True)} | {cell('ece_esi1')} | "
                     f"{g['superiority']} | **{STATUS_PL[g['status']]}** |")
    lines += ["", "✓ spełnione · ✗ niespełnione · ? niesprawdzalne (brak danych).", ""]

    # 5. Decyzja
    pars = results["parsimony"]
    lines += ["## 5. Decyzja i rekomendacja", "",
              f"**{rec['decision_label']}** — {pars['reason']}.", ""]
    if pars.get("pair_vs_single"):
        p = pars["pair_vs_single"]
        lines += [f"Para TFM vs {set_label(pars['best_single'])}: Δ log loss {_delta_ci(p)} (sparowany bootstrap).", ""]
    if not rec["noise_criterion_checked"]:
        lines += [f"⚠ Kryterium szumu seedów niesprawdzone — uruchom `{seeds_command(results['feature_set'], results['tag'])}` "
                  "i powtórz ten krok (gotowe powtórzenia zostaną użyte ponownie).", ""]
    lines += ["Raport końcowy (TEST — jeden raz) dla tej receptury:", "", "```", rec["command"], "```", "",
              "To samo przez tag (najnowsze runy):", "", "```", rec["command_tag"], "```", ""]
    if rec["includes_tfm"]:
        lines += ["Receptura ma model fundamentalny — 08 zapisze także `backend_ensemble_weights_no_tfm.json` "
                  "(ta sama receptura dopasowana na VAL bez TFM; backend użyje jej, gdy TFM nie jest serwowany).", ""]

    # 6. Reguły — wartości bezwzględne
    lines += ["## 6. Reguły decyzyjne — wartości bezwzględne (cross-fitting, średnia z powtórzeń)", "",
              "| zestaw | QWK argmax | QWK koszt | QWK progi T | niedotriaż. argmax | niedotriaż. koszt | niedotriaż. "
              "progi T | kryt. pominięcie progi T | czułość ESI 1 progi T | AUC makro | ECE ESI 1 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in variants:
        m = metrics[v["id"]]
        lines.append(
            f"| {variant_label(v)} | {_f(m['qwk_argmax']['mean'])} | {_f(m['qwk_cost']['mean'])} | "
            f"{_f(m['qwk_target']['mean'])} | {_pct(m['undertriage_argmax']['mean'])} | "
            f"{_pct(m['undertriage_cost']['mean'])} | {_pct(m['undertriage_target']['mean'])} | "
            f"{_pct(m['critical_miss_target']['mean'])} | {_pct(m['recall_esi1_target']['mean'])} | "
            f"{_f(m['auc_macro']['mean'])} | {_f(m['ece_esi1']['mean'])} |")
    if results.get("vc"):
        lines += ["", "Podgrupa vc (komplet HR/SBP/RR — populacja dopasowania progów, jak TOPSOR):", "",
                  "| zestaw | QWK progi T | niedotriaż. progi T | kryt. pominięcie progi T | log loss |",
                  "|---|---:|---:|---:|---:|"]
        for v in variants:
            m = results["vc"].get(v["id"]) or {}
            lines.append(f"| {variant_label(v)} | {_f(m.get('qwk_target'))} | {_pct(m.get('undertriage_target'))} | "
                         f"{_pct(m.get('critical_miss_target'))} | {_f(m.get('log_loss'))} |")
    lines.append("")

    # 7. Odniesienie klasyczne
    classical = [v for v in variants if v["set"] == CLASSICAL_SET]
    lines += ["## 7. Odniesienie: LightGBM sam vs zestaw klasyczny", ""]
    if classical:
        v = classical[0]
        d = deltas[v["id"]]["log_loss"]
        lines += [f"{variant_label(v)} ({', '.join(DISPLAY.get(m, m) for m in v['members'])}): Δ log loss względem "
                  f"LightGBM {_delta_ci(d)}; reguła C1: {results['classical_superiority']}. (Tylko kontekst — zestaw "
                  "klasyczny nie jest kandydatem tego eksperymentu.)", ""]
    else:
        lines += ["Pominięte — brak runów innych modeli klasycznych z tym tagiem (albo inne wiersze VAL).", ""]

    # 8. Różnorodność
    lines += ["## 8. Różnorodność modeli (surowe predykcje VAL)", "",
              "| model A | model B | niezgodność argmax | korelacja E[ESI] | korelacja błędów |", "|---|---|---:|---:|---:|"]
    for r in results.get("diversity") or []:
        lines.append(f"| {DISPLAY.get(r['model_a'], r['model_a'])} | {DISPLAY.get(r['model_b'], r['model_b'])} | "
                     f"{_pct(r['disagreement'])} | {_f(r['corr_expected'], 3)} | {_f(r['corr_errors'], 3)} |")
    lines += ["", "Łączenie pomaga najbardziej, gdy modele o podobnej jakości mylą się na RÓŻNYCH pacjentach "
              "(niska korelacja błędów).", ""]

    # 9. Wagi
    lines += ["## 9. Wagi i temperatury w foldach (średnia ± SD z 2 × powtórzeń)", "",
              "| wariant | wagi członków | temperatura T |", "|---|---|---:|"]
    for v in variants:
        f = results["folds"].get(v["id"])
        if not f:
            continue
        weights = ", ".join(f"{DISPLAY.get(m, m)} {s['mean']:.3f}" + (f" ± {s['sd']:.3f}" if s.get("sd") is not None else "")
                            for m, s in f["weights"].items())
        t = f["temperature"]
        lines.append(f"| {variant_label(v)} | {weights} | {t['mean']:.3f}"
                     + (f" ± {t['sd']:.3f}" if t.get("sd") is not None else "") + " |")
    lines.append("")

    # 10. Metoda
    lines += ["## 10. Opis metody do pracy", "", methods_paragraph(results), ""]

    # 11. Zastrzeżenia
    lines += ["## 11. Zastrzeżenia", "",
              "- Wyłącznie VAL; wybór metody łączenia (2 metody) na tych samych oszacowaniach daje niewielki "
              "optymizm — dlatego bramka wymaga przewagi ≥ MID i przedziału bez zera.",
              "- W cross-fittingu wagi, temperatura i progi dopasowane są na połowie VAL — oszacowanie jest raczej "
              "zachowawcze względem receptury 08 dopasowanej na całym VAL.",
              "- Kryterium szumu używa SD seedów LightGBM; zmienność samych TFM (losowanie worków kontekstu) nie "
              "jest w nim ujęta (refity seedów TFM tylko na jawne żądanie: `09_seed_refits.py --models tfm`).",
              "- Predykcje TFM na GPU (TF32 / autocast) mogą minimalnie różnić się między kartami i wersjami bibliotek; "
              "wagi TabPFN-3.5 mają licencję niekomercyjną (praca dyplomowa — dozwolone; wdrożenie komercyjne — nie).",
              "- Bootstrap opisuje zmienność pacjentów VAL, nie zmienność treningu; dlatego dodatkowo kryterium 3.",
              ""]
    lines += ["## Pliki", "",
              "- `results.json` — wszystkie liczby (metryki wariantów z wartościami z powtórzeń, Δ z CI, bramka, wagi foldów),",
              "- `recommendation.json` — decyzja, runy i dokładne polecenie 08,",
              "- `units/repeat<k>.json`, `preds/repeat<k>.npz` — powtórzenia cross-fittingu (wznawianie),"]
    if results.get("figures"):
        lines.append("- `figures/`: " + ", ".join(f"`{Path(f).name}`" for f in results["figures"]) + ".")
    lines += ["", f"Wygenerowano {results['created']}.", ""]
    return lines


def methods_paragraph(results: Mapping[str, Any]) -> str:
    cfg = results["config"]
    dec = cfg["decision"]
    spec = results["gate_spec"]
    members = ", ".join(DISPLAY.get(m, m) for m in results["runs"] if m != BASE_MODEL)
    return (
        f"Wpływ dołączenia wstępnie wytrenowanych tabularnych modeli fundamentalnych ({members}) do LightGBM "
        f"oceniono wyłącznie na zbiorze walidacyjnym (n = {results['data']['n']:,}) metodą {cfg['repeats']}-krotnie "
        f"powtarzanego dwukrotnego cross-fittingu: VAL dzielono losowo na dwie połowy warstwowane względem poziomu "
        f"ESI i oddziału; na jednej połowie dobierano wagi połączenia (minimalizacja log loss), temperaturę "
        f"(temperature scaling) oraz regułę decyzyjną z docelowym niedotriażowaniem T = {100 * dec['target']:g}% "
        f"(progi oczekiwanego poziomu ESI z siatką bezpieczeństwa, dopasowane w podgrupie z kompletem parametrów "
        f"życiowych), a oceniano je na drugiej połowie, po czym role połówek zamieniano. LightGBM stanowiący punkt "
        f"odniesienia przechodził identyczną procedurę. Rozważano zestawy LightGBM z jednym albo dwoma modelami "
        f"fundamentalnymi, łączone średnią arytmetyczną lub geometryczną prawdopodobieństw. Różnice względem "
        f"LightGBM liczono na tych samych pacjentach i uśredniano po powtórzeniach; 95% przedziały ufności "
        f"wyznaczono sparowanym bootstrapem warstwowanym względem ESI ({cfg['n_boot']} losowań). Kryteria "
        f"włączenia modelu fundamentalnego ustalono przed analizą: spadek log loss o co najmniej "
        f"{spec['mid_log_loss']:g} z przedziałem ufności poniżej zera, większy niż dwukrotne odchylenie "
        f"standardowe log loss LightGBM między seedami, bez pogorszenia niedotriażowania o ponad "
        f"{100 * spec['max_undertriage_worse']:g} punkt procentowy i kalibracji ESI 1 (ECE) o ponad "
        f"{spec['max_ece_esi1_worse']:g}; zestaw z dwoma modelami fundamentalnymi wybierano tylko, gdy poprawiał "
        f"log loss najlepszego zestawu z jednym modelem o co najmniej {spec['mid_log_loss']:g}. Zbiór testowy "
        f"wykorzystano jednokrotnie, do oceny wybranej receptury."
    )


def summary_lines(results: Mapping[str, Any]) -> list[str]:
    rec = results["recommendation"]
    lines = [f"# Eksperyment B (TFM + LightGBM, VAL) — {results['feature_set']} · tag {results['tag']}", "",
             f"- powtórzenia: {results['n_done']}/{results['n_units']}"
             + (f", błędy: {len(results['failures'])}" if results.get("failures") else ""),
             f"- decyzja: **{rec['decision_label']}** — {results['parsimony']['reason']}",
             f"- 08: `{rec['command']}`", "", "| zestaw | Δ log loss [95% CI] | bramka |", "|---|---|---|"]
    for sid, g in results["gate"].items():
        lines.append(f"| {set_label(sid)} ({METHOD_PL[g['method']]}) | {_delta_ci(results['deltas'][g['variant']]['log_loss'])} "
                     f"| {STATUS_PL[g['status']]} |")
    lines += ["", "Szczegóły: report.md (VAL only — TEST raz w 08)."]
    return lines


def decision_lines(results: Mapping[str, Any]) -> list[str]:
    rec = results["recommendation"]
    out = [f"Decyzja (VAL): {rec['decision_label']} — {results['parsimony']['reason']}"]
    for sid, g in results["gate"].items():
        d = results["deltas"][g["variant"]]["log_loss"]
        out.append(f"  {set_label(sid)} ({METHOD_PL[g['method']]}): Δ log loss {_delta_ci(d)} → {STATUS_PL[g['status']]}")
    if not rec["noise_criterion_checked"]:
        out.append(f"  ⚠ brak SD seedów lightgbm — {seeds_command(results['feature_set'], results['tag'])}")
    out.append(f"Raport końcowy (TEST raz): {rec['command']}")
    return out


# ─────────────────────────────────────────
# Przebieg
# ─────────────────────────────────────────
def config_hash(variants: Sequence[Variant], cfg: BlendConfig, n_rows: int) -> str:
    payload = {"version": TFM_BLEND_VERSION, "variants": [v.as_dict() for v in variants], "n_folds": N_FOLDS,
               "seed": int(cfg.seed), "rules": list(RULES), "target": float(cfg.target),
               "safety_net": bool(cfg.safety_net), "population": cfg.fit_population, "n_rows": int(n_rows)}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def estimate_repeat_s(variants: Sequence[Variant], n: int) -> float:
    """Zgrubny szacunek jednego powtórzenia (Nelder–Mead ~150 ewaluacji / członek + reguły) [s]."""
    total = 0.0
    for v in variants:
        m = len(v.members)
        evals = 1 if m == 1 else 150 * m
        total += N_FOLDS * (0.3 + (n / 2) * 1.5e-7 * evals * m + (n / 2) * 2e-5)
    return total


def plan_lines(selection: Selection, variants: Sequence[Variant], units: Sequence[Mapping[str, Any]],
               todo: int, n: int, cfg: BlendConfig) -> list[str]:
    est = estimate_repeat_s(variants, n) * todo
    lines = [f"Eksperyment B — plan (tag {selection.tag}, VAL {n:,} wierszy)",
             "  runy: " + ", ".join(f"{m}={r.name}" for m, r in selection.members.items()),
             "  odniesienie: " + (", ".join(f"{m}={r.name}" for m, r in selection.references.items()) or "—"),
             f"  warianty ({len(variants)}): " + ", ".join(v.id for v in variants),
             f"  powtórzenia: {len(units)} × {N_FOLDS} foldy; do policzenia: {todo} (≈ {format_duration(est)} + "
             f"bootstrap {cfg.n_boot}×)"]
    for note in selection.notes:
        lines.append(f"  uwaga: {note}")
    return lines


def run_tfm_blend(cfg: BlendConfig, plan_only: bool = False) -> dict[str, Any]:
    """
    Eksperyment B → results.json + recommendation.json + report.md (albo plan przy `plan_only`).
    `BlendSkipped` — warunki niespełnione (skrypt kończy się kodem 0 z linią `skipped: …`).
    """
    from src.evaluation import report_sections as rs

    cfg = validate_config(cfg)
    selection = select_runs(cfg)
    data, selection = load_val(selection)
    variants = build_variants(selection, cfg.methods)
    dopts = decision_options(cfg)
    root = analysis_path(cfg.feature_set, selection.tag, cfg.runs_dir)
    store = UnitStore(root)
    source = selection.source_key()
    digest = config_hash(variants, cfg, data.n)
    units = [{"unit": f"repeat{k}", "model": "blend", "source_run": source, "repeat": k,
              "seed": repeat_seed(cfg.seed, k)} for k in range(1, int(cfg.repeats) + 1)]

    def expect(u: Mapping[str, Any]) -> dict[str, Any]:
        return {"source_run": source, "config_hash": digest, "version": TFM_BLEND_VERSION, "seed": u["seed"]}

    def preds_path(u: Mapping[str, Any]) -> Path:
        return root / PREDS_SUBDIR / f"{u['unit']}.npz"

    def ready(u: Mapping[str, Any]) -> bool:
        return (store.valid(str(u["unit"]), expect(u)) is not None
                and load_preds(preds_path(u), variants, data.row_ids) is not None)

    todo = [u for u in units if cfg.force or not ready(u)]
    if plan_only:
        return {"plan_lines": plan_lines(selection, variants, units, len(todo), data.n, cfg),
                "feature_set": cfg.feature_set, "tag": selection.tag, "variants": [v.as_dict() for v in variants]}

    store.write_plan({"analysis": ANALYSIS, "feature_set": cfg.feature_set, "tag": selection.tag,
                      "config_hash": digest, "variants": [v.as_dict() for v in variants], "units": units})
    for line in plan_lines(selection, variants, units, len(todo), data.n, cfg):
        log.info(line)

    def compute(u: Mapping[str, Any]) -> dict[str, Any]:
        preds, records = crossfit_repeat(data, variants, int(u["seed"]), dopts)
        save_preds(preds_path(u), preds, variants, data.row_ids)
        ll = {v.id: float(proba_metrics(data.y, preds.proba[v.id])["log_loss"]) for v in variants}
        log.info(f"[{u['unit']}] log loss (cross-fitted): "
                 + ", ".join(f"{k} {val:.4f}" for k, val in ll.items()))
        return {"model": "blend", "repeat": u["repeat"], "folds": records, "log_loss": ll}

    # Plik jednostki bez pliku predykcji (np. usunięty) → jednostka liczona ponownie
    force_units = cfg.force
    missing = [u for u in units if not cfg.force and store.valid(str(u["unit"]), expect(u)) is not None
               and load_preds(preds_path(u), variants, data.row_ids) is None]
    for u in missing:
        store.path(str(u["unit"])).unlink(missing_ok=True)
    failures = execute_units(store, units, expect, compute, ANALYSIS, force=force_units,
                             estimate_s=lambda u: estimate_repeat_s(variants, data.n))

    done: list[tuple[Mapping[str, Any], dict, RepeatPreds]] = []
    for u in units:
        payload = store.valid(str(u["unit"]), expect(u))
        preds = load_preds(preds_path(u), variants, data.row_ids) if payload else None
        if payload and preds is not None:
            done.append((u, payload, preds))
    if not done:
        raise RuntimeError("Żadne powtórzenie cross-fittingu nie zakończyło się — szczegóły w logu")

    log.info(f"Bootstrap {cfg.n_boot}× (warstwy ESI, sparowany) dla {len(variants)} wariantów × {len(done)} powtórzeń…")
    boot_seed = int(cfg.seed)
    points, samples, vc_points = [], [], []
    for _, _, preds in done:
        point, sample = repeat_bootstrap(data.y, preds, variants, int(cfg.n_boot), boot_seed)
        points.append(point)
        samples.append(sample)
        if data.vc is not None and data.vc.any():
            vc_points.append({v.id: subgroup_point(data.y, preds, v.id, data.vc) for v in variants})
    metrics, deltas, mean_samples = aggregate(points, samples, variants)
    vc_summary = ({v.id: {k: _mean([p[v.id].get(k) for p in vc_points]) for k in vc_points[0][v.id]}
                   for v in variants} if vc_points else {})

    seed_info = lightgbm_seed_sd(cfg.feature_set, selection.tag, selection.lightgbm, cfg.runs_dir)
    mid = float(rs.MINIMAL_IMPORTANT_DIFFERENCE["log_loss"])
    best = best_variants(variants, metrics)
    gates = {sid: evaluate_gate(v, deltas[v.id], seed_info.get("sd"), mid) for sid, v in best.items()}
    choice = choose(gates, deltas, mean_samples, best, mid)
    complete = not failures and len(done) == len(units)
    rec = recommendation(cfg, selection, choice, best, gates, seed_info, complete)
    records = [r for _, payload, _ in done for r in payload.get("folds") or []]
    classical = next((v for v in variants if v.set_id == CLASSICAL_SET), None)
    classical_superiority = None
    if classical is not None:
        d = deltas[classical.id]["log_loss"]
        classical_superiority = rs.superiority("log_loss", d.get("delta"), d.get("ci_low"), d.get("ci_high"),
                                               seed_info.get("sd"), name_a="zestaw klasyczny", name_b="LightGBM")
    diversity_models = [BASE_MODEL, *selection.tfms, *selection.references]
    all_runs = {**selection.members, **selection.references}
    results: dict[str, Any] = {
        "analysis": ANALYSIS,
        "feature_set": cfg.feature_set,
        "tag": selection.tag,
        "created": now_iso(),
        "version": TFM_BLEND_VERSION,
        "metrics_version": METRICS_VERSION,
        "data_fingerprint": selection.lightgbm.info.get("data_fingerprint"),
        "complete": complete,
        "n_units": len(units),
        "n_done": len(done),
        "failures": failures,
        "runs": {m: r.name for m, r in selection.members.items()},
        "references": {m: r.name for m, r in selection.references.items()},
        "source_runs": {m: r.name for m, r in all_runs.items()},
        "run_info": {r.name: {k: r.info.get(k) for k in ("profile", "tag", "train_sample", "eval_sample", "pretrained",
                                                          "data_fingerprint")} for r in all_runs.values()},
        "notes": list(selection.notes),
        "data": {"n": data.n, "esi_counts": np.bincount(data.y, minlength=config.N_CLASSES).tolist(),
                 "n_vc": int(data.vc.sum()) if data.vc is not None else None},
        "config": {"repeats": int(cfg.repeats), "n_folds": N_FOLDS, "n_boot": int(cfg.n_boot),
                   "methods": list(cfg.methods), "seed": int(cfg.seed), "alpha": ALPHA, "config_hash": digest,
                   "decision": {"main_rule": MAIN_RULE, "rules": list(RULES), "target": float(cfg.target),
                                "safety_net": bool(cfg.safety_net), "fit_population": cfg.fit_population}},
        "gate_spec": {"mid_log_loss": mid, "seed_sd_factor": float(rs.SEED_SD_FACTOR),
                      "max_undertriage_worse": GATE_MAX_UNDERTRIAGE_WORSE,
                      "max_ece_esi1_worse": GATE_MAX_ECE_ESI1_WORSE, "max_tfms": MAX_TFMS,
                      "main_rule": MAIN_RULE},
        "seed_sd": seed_info,
        "variants": [v.as_dict() for v in variants],
        "metrics": metrics,
        "deltas": deltas,
        "vc": vc_summary,
        "gate": gates,
        "parsimony": choice,
        "classical_superiority": classical_superiority,
        "recommendation": rec,
        "folds": fold_summary(records, variants),
        "diversity": diversity_rows(data, diversity_models),
        "figures": [],
    }
    if cfg.plots:
        fig_dir = root / FIGURES_SUBDIR
        made: list[Path | None] = []
        for name, fn in (("delta_forest.png", lambda p: plot_deltas(results, p)),
                         ("diversity.png", lambda p: plot_diversity(results["diversity"], diversity_models, p)),
                         ("weights_folds.png", lambda p: plot_weights(
                             results["folds"], results["variants"], p,
                             [g["variant"] for g in gates.values()] + ([classical.id] if classical else [])))):
            try:
                made.append(fn(fig_dir / name))
            except Exception as exc:  # noqa: BLE001 — wykres nie może zatrzymać analizy
                log.warning(f"Wykres {name} pominięty: {type(exc).__name__}: {exc}")
        results["figures"] = [str(p) for p in made if p is not None]
    store.write_results(results)
    write_json(root / "recommendation.json", rec)
    write_text_atomic(root / "report.md", "\n".join(report_lines(results)) + "\n")
    store.write_summary(summary_lines(results))
    for line in decision_lines(results):
        log.info(line)
    return results


__all__ = [
    "ANALYSIS",
    "BASELINE",
    "BlendConfig",
    "BlendSkipped",
    "Variant",
    "aggregate",
    "analysis_path",
    "best_variants",
    "build_variants",
    "choose",
    "crossfit_repeat",
    "delta_summary",
    "evaluate_gate",
    "fold_strata",
    "latest_source_runs",
    "lightgbm_seed_sd",
    "load_val",
    "parse_methods",
    "parse_tfm_models",
    "recommendation",
    "repeat_bootstrap",
    "run_tfm_blend",
    "select_runs",
]
