"""
Krok 8: RAPORT KOŃCOWY do pracy — jedna ocena na TEST dla receptury wybranej wcześniej na VAL.

Nic nie jest trenowane: wszystko liczone z zapisanych predykcji runów
(runs/<run>/predictions_{val,test}.parquet). Obliczenia nowych sekcji: src/evaluation/report_sections.py.

Receptura (wybrana na VAL, np. po `07_compare_runs.py`):
    runy         --runs RUN …  albo  --feature-set triage --tag standard (najnowszy ukończony run
                 każdego modelu; opcjonalnie --models xgboost,catboost,ebm)
    wagi         --weights '{"xgboost": 0.4, "catboost": 0.3, "ebm": 0.3}' | optimize | equal
                 (albo ścieżka do pliku JSON z wagami, także backend_ensemble_weights.json —
                 jego "method", "rule" i "temperature" są przejmowane; sprzeczna opcja = odmowa)
    łączenie     --method mean | geometric (domyślnie mean)
    temperatura  --temperature fit (dopasowana na VAL, domyślnie) | none
    reguła       --main-rule target | bayes_target | cost | argmax (domyślnie target) — reguła
                 w pliku dla backendu; --target-undertriage T (domyślnie z config), --guarantee
                 (B6), --safety-net / --no-safety-net (B2, domyślnie włączona), --fit-population
                 vc | all (wiersze VAL do dopasowania progów, domyślnie vc); --rule all (domyślnie:
                 wszystkie reguły w tabelach) | both (argmax + koszt) | jedna reguła

Kroki:
    1. Kontrola runów: ten sam zestaw cech, ta sama wersja i to samo przygotowanie danych, jeden
       run na model, te same wiersze i etykiety w val i test — inaczej ODMOWA (kod wyjścia 2).
    2. VAL: połączenie (wagi, metoda) i temperatura T dopasowana na VAL; reguły decyzyjne
       (progi z celem T + siatka bezpieczeństwa, decyzja Bayesa z λ) dopasowane na VAL po
       temperaturze. Każdy pojedynczy model przechodzi tę samą obróbkę (własna T, własne reguły).
    3. TEST — raz:
       (a) połączenie i każdy model z 95% przedziałami ufności (bootstrap percentylowy,
           stratyfikowany po ESI; odsetki pacjentów — dokładny przedział Clopper–Pearsona z k/n)
           dla WSZYSTKICH reguł — cała populacja, podgrupa vc (komplet HR/SBP/RR), oddziały;
       (b) sparowane Δ: połączenie − najlepszy pojedynczy model (najniższy log loss na VAL)
           dla QWK, niedotriażowania i log loss, z regułą wyższości C1 (CI bez 0, |Δ| > MID,
           |Δ| > 2 × SD seedów z runs/analyses);
       (c) kalibracja przed i po temperaturze (ECE, wykresy niezawodności) i per poziom / oddział (C8);
       (d) macierze pomyłek; (e) przyjęcia do szpitala: AUC, podgrupy, niezgodność model–pielęgniarka (C4);
       (f) punkt pracy na VAL (C6), podgrupy i równość (C2), analiza błędów (C10), krzywe
           decyzyjne (C7), odsyłanie niepewnych i zbiory conformal (C13), podsumowanie analiz
           runs/analyses/ (C1 seedy, C3 leave-site-out, C9 krzywa uczenia).
       Sekcja bez danych (brak `admitted`, atrybutów pacjentów, wyników analiz) jest pomijana z notą.
    4. Zapis: runs/final_report__<zestaw>__<czas>/ (albo --out): report.md (po polsku, tabele do
       pracy), metrics.json, recipe.json, tables/*.csv, figures/*.png oraz
       backend_ensemble_weights.json (format backend/app/models/ensemble.py, z parametrami reguły).

UWAGA: zbiór TESTOWY oceniasz RAZ, dla ostatecznej receptury. Kolejne uruchomienia z innymi
recepturami i wybór „najlepszej” na teście zawyżają wyniki (test przestaje być niezależny).
Każde uruchomienie jest dopisywane do rejestru runs/final_reports.jsonl (niezależnie od --out);
kolejny raport dla tych samych runów wypisuje wcześniejsze (z informacją, czy receptura była ta sama).

Użycie:
    python scripts/08_final_report.py --feature-set triage --tag standard      # domyślnie: optimize, target T=15%
    python scripts/08_final_report.py --runs RUN_A RUN_B RUN_C --weights optimize --method geometric
    python scripts/08_final_report.py --feature-set triage --tag smoke --include-samples --bootstrap 300
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.ensemble import (  # noqa: E402
    Run,
    aligned_meta,
    aligned_probas,
    apply_temperature,
    blend,
    fit_temperature,
    latest_runs,
    load_run,
    optimize_weights,
    resolve_feature_set,
)
from src.evaluation import report_sections as rs  # noqa: E402
from src.evaluation.decision import expected_level, normalize_spec  # noqa: E402
from src.evaluation.report_sections import (  # noqa: E402
    COUNT_METRICS,  # noqa: F401 — eksport dla zgodności (formatowanie w report_sections)
    NURSE,
    PERCENT_METRICS,  # noqa: F401
    RULE_LABELS,
    RULE_ORDER,
    DecisionOptions,
    FittedRules,
    Fmt,
    md_table,
)
from src.evaluation.report_sections import fnum as _f  # noqa: E402
from src.evaluation.report_sections import int_text as _int  # noqa: E402
from src.evaluation.report_sections import jsonable as _jsonable  # noqa: E402
from src.evaluation.statistics import (  # noqa: E402
    DEFAULT_CI_METRICS,
    DEFAULT_N_BINS,
    BootstrapSamples,
    calibration_summary,
    ci_to_dict,
    crossfit_temperature,
    point_metrics,
)
from src.utils import config  # noqa: E402
from src.utils.config import N_CLASSES  # noqa: E402
from src.utils.fileio import write_json_atomic, write_text_atomic  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

# ─────────────────────────────────────────
# Stałe
# ─────────────────────────────────────────
BLEND = "połączenie"
BLEND_RAW = "połączenie (bez T)"
RULES = RULE_ORDER                                   # argmax, cost, target, bayes_target
RULE_SETS: dict[str, list[str]] = {"all": list(RULE_ORDER), "both": ["argmax", "cost"]}
EXIT_REFUSED = 2
ECE_NULL_SIMS = 200          # losowania idealnie skalibrowanego modelu (wartość odniesienia ECE)
DEFAULT_METHOD = "mean"
DEFAULT_TEMPERATURE = "fit"
DEFAULT_RULE = "all"
DEFAULT_MAIN_RULE = "target"
RULE_PARAM_KEYS = frozenset({"cutpoints", "safety_q", "lam"})
BACKEND_KEYS = frozenset({"weights", "method", "temperature", "rule"}) | RULE_PARAM_KEYS   # backend ensemble.py
TEMPERATURE_RTOL = 1e-3      # T dopasowana ponownie vs T z pliku wag: większa różnica = ostrzeżenie
REGISTRY_FILE = "final_reports.jsonl"   # rejestr wszystkich raportów końcowych (w RUNS_DIR)

METRIC_LABELS: dict[str, str] = {
    "quadratic_weighted_kappa": "QWK ↑",
    "accuracy": "trafność (accuracy) ↑",
    "within_one_accuracy": "trafność ±1 poziom ↑",
    "undertriage_rate": "niedotriażowanie ESI 1–2 → 3–5 ↓",
    "critical_miss_rate": "krytyczne pominięcie ESI 1 → 4–5 ↓",
    "overtriage_rate": "przetriażowanie ESI 4–5 → 1–2 ↓",
    "mistriage_under": "zaniżenie pilności o ≥ 1 poziom ↓",
    "mistriage_over": "zawyżenie pilności o ≥ 1 poziom ↓",
    "recall_esi1": "czułość ESI 1 ↑",
    "recall_esi12": "czułość ESI 1–2 (jako ESI 1–2) ↑",
    "auc_macro": "AUC makro (jeden-przeciw-reszcie) ↑",
    "log_loss": "log loss ↓",
    "brier": "Brier ↓",
    "expected_cost": "oczekiwany koszt decyzji (macierz k = 3) ↓",
    "ece": "ECE top-label ↓",
}
SHORT_LABELS: dict[str, str] = {
    "quadratic_weighted_kappa": "QWK",
    "undertriage_rate": "niedotriażowanie",
    "critical_miss_rate": "kryt. pominięcie",
    "overtriage_rate": "przetriażowanie",
    "recall_esi1": "czułość ESI 1",
    "expected_cost": "koszt",
    "auc_macro": "AUC",
    "log_loss": "log loss",
    "brier": "Brier",
    "ece": "ECE",
}
# Tabela 2 (połączenie vs modele): metryki z decyzji i z prawdopodobieństw
COMPARE_DECISION_METRICS = ["quadratic_weighted_kappa", "undertriage_rate", "critical_miss_rate",
                            "overtriage_rate", "recall_esi1", "expected_cost"]
COMPARE_PROBA_METRICS = ["auc_macro", "log_loss", "brier", "ece"]
PAIRED_METRICS = ["quadratic_weighted_kappa", "undertriage_rate", "log_loss"]
SITE_METRICS = ["quadratic_weighted_kappa", "undertriage_rate", "critical_miss_rate", "overtriage_rate",
                "recall_esi1"]
CALIBRATION_PAIRED_METRICS = ["ece", "log_loss", "brier"]


class RecipeError(ValueError):
    """Runy albo receptura nie nadają się do raportu końcowego (odmowa, kod 2)."""


# ─────────────────────────────────────────
# Argumenty
# ─────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raport końcowy do pracy: jedna ocena na TEST dla receptury wybranej na VAL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs", nargs="+", default=None, help="Nazwy katalogów w runs/ albo ścieżki")
    parser.add_argument("--feature-set", default=None,
                        help="Zestaw cech przy wyborze najnowszych runów (domyślnie triage)")
    parser.add_argument("--tag", default=None, help="Tylko runy z tym tagiem (np. standard)")
    parser.add_argument("--models", default=None, help="Lista modeli przy wyborze najnowszych runów, np. xgboost,ebm")
    parser.add_argument("--include-samples", action="store_true", help="Uwzględnij runy na próbkach (tylko test!)")
    parser.add_argument("--include-untuned", action="store_true", help="Uwzględnij runy bez tuningu")
    parser.add_argument("--include-weighted", action="store_true", help="Uwzględnij runy z wagami klas")
    parser.add_argument("--weights", default="optimize",
                        help="JSON {model: waga}, ścieżka do pliku JSON (także backend_ensemble_weights.json — "
                             "wtedy jego method / rule / temperature są przejmowane), 'optimize' (min log loss "
                             "na VAL, domyślnie) albo 'equal'")
    parser.add_argument("--method", choices=["mean", "geometric"], default=None,
                        help=f"Sposób łączenia (domyślnie z pliku wag, inaczej {DEFAULT_METHOD})")
    parser.add_argument("--temperature", choices=["fit", "none"], default=None,
                        help=f"Temperatura połączonych prawdopodobieństw dopasowana na VAL albo bez "
                             f"(domyślnie z pliku wag, inaczej {DEFAULT_TEMPERATURE})")
    rules = parser.add_argument_group("reguły decyzyjne (dopasowanie na VAL, ocena na TEST)")
    rules.add_argument("--main-rule", choices=list(RULES), default=None,
                       help=f"Reguła główna: tabele szczegółowe, podgrupy, analiza błędów i plik dla backendu "
                            f"(domyślnie z pliku wag, inaczej {DEFAULT_MAIN_RULE})")
    rules.add_argument("--backend-rule", choices=list(RULES), default=None,
                       help="Dawna nazwa --main-rule (zgodność wstecz)")
    rules.add_argument("--rule", choices=["all", "both", *RULES], default=None,
                       help=f"Reguły oceniane w raporcie: all = argmax, koszt, progi T, Bayes T (domyślnie "
                            f"{DEFAULT_RULE}); both = argmax + koszt; jedna nazwa = tylko ta reguła "
                            f"(i ona trafia do backendu)")
    rules.add_argument("--target-undertriage", type=float, default=config.DECISION_TARGET_UNDERTRIAGE,
                       metavar="T", help="Cel niedotriażowania ESI 1–2 → 3–5 reguł progi T / Bayes T (na VAL)")
    rules.add_argument("--guarantee", action="store_true",
                       help="B6: próg tak, by górna granica Clopper–Pearsona niedotriażowania na VAL ≤ T")
    rules.add_argument("--safety-net", action=argparse.BooleanOptionalAction, default=True,
                       help="B2: P(ESI 1)+P(ESI 2) ≥ q → najwyżej ESI 3 (q dopasowane na VAL)")
    rules.add_argument("--fit-population", choices=["vc", "all"], default=config.DECISION_FIT_POPULATION,
                       help="Wiersze VAL do dopasowania progów: vc (komplet HR/SBP/RR, jak TOPSOR) albo all")
    parser.add_argument("--bootstrap", type=int, default=2000, metavar="N", help="Liczba losowań bootstrap")
    parser.add_argument("--alpha", type=float, default=0.05, help="1 − poziom ufności przedziałów")
    parser.add_argument("--seed", type=int, default=42, help="Ziarno losowań bootstrap")
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS, help="Biny ECE i wykresów niezawodności")
    parser.add_argument("--decimal-comma", action="store_true", help="Przecinek dziesiętny w report.md (typografia PL)")
    parser.add_argument("--no-plots", action="store_true", help="Bez wykresów")
    parser.add_argument("--out", type=Path, default=None, help="Katalog raportu")
    args = parser.parse_args(argv)
    if args.bootstrap < 1:
        parser.error("--bootstrap musi być ≥ 1 (raport końcowy zawsze podaje przedziały ufności)")
    if not 0 < args.alpha < 1:
        parser.error("--alpha musi być w (0, 1)")
    if not 0 < args.target_undertriage < 1:
        parser.error("--target-undertriage musi być w (0, 1)")
    return args


# ─────────────────────────────────────────
# Runy i ich kontrola
# ─────────────────────────────────────────
def resolve_runs(args: argparse.Namespace) -> list[Run]:
    """Jawna lista --runs albo najnowszy ukończony run każdego modelu (filtry jak w 07)."""
    if args.runs:
        try:
            return [load_run(r) for r in args.runs]
        except FileNotFoundError as exc:
            raise RecipeError(str(exc)) from exc
    feature_set = resolve_feature_set(args.feature_set or "triage")
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    runs = latest_runs(feature_set, models=models, include_samples=args.include_samples, tag=args.tag,
                       include_weighted=args.include_weighted, require_tuned=not args.include_untuned)
    # Refit analizy z modelem (09_seed_refits.py --save-models) bywa „najnowszym” runem — raport
    # dotyczy runu głównego, z którego refit powstał (refit jawnie: --runs)
    for i, run in enumerate(runs):
        source = run.info.get("refit_of")
        if source:
            try:
                runs[i] = load_run(source)
            except FileNotFoundError as exc:
                raise RecipeError(f"{run.name} to refit analizy, a jego run główny {source} nie istnieje — "
                                  f"podaj runy jawnie (--runs)") from exc
            log.warning(f"{run.name} to refit analizy — w raporcie biorę jego run główny {source}")
    if not runs:
        raise RecipeError(
            f"Brak ukończonych runów dla zestawu '{feature_set}'"
            + (f" z tagiem '{args.tag}'" if args.tag else "")
            + " (filtry: próbki, wagi klas, bez tuningu — zob. --include-samples / --include-weighted / "
              "--include-untuned albo jawne --runs)")
    return runs


def check_runs(runs: list[Run], feature_set_arg: str | None) -> dict[str, Any]:
    """
    Odmowa (RecipeError), gdy runy nie pochodzą z jednego przygotowania danych i zestawu cech
    albo gdy jeden model występuje dwukrotnie. Zwraca wspólne informacje o danych.
    """
    feature_sets = {resolve_feature_set(str(r.info.get("feature_set"))) for r in runs}
    if len(feature_sets) > 1:
        raise RecipeError(f"Runy mają różne zestawy cech: {sorted(feature_sets)} — łącz runy jednego zestawu")
    feature_set = feature_sets.pop()
    if feature_set_arg and resolve_feature_set(feature_set_arg) != feature_set:
        raise RecipeError(f"--feature-set {feature_set_arg} nie zgadza się z zestawem runów ('{feature_set}')")
    versions = {r.info.get("data_version") for r in runs}
    if None in versions:
        missing = [r.name for r in runs if not r.info.get("data_version")]
        raise RecipeError(f"Runy bez data_version w run_info.json: {missing}")
    if len(versions) > 1:
        raise RecipeError(f"Runy z różnych wersji danych: {sorted(versions)}")
    fingerprints = {r.info.get("data_fingerprint") for r in runs if r.info.get("data_fingerprint")}
    if len(fingerprints) > 1:
        raise RecipeError(f"Runy z różnych przygotowań danych (odciski {sorted(fingerprints)}) — łącz runy "
                          f"z jednego przygotowania")
    models = [r.model for r in runs]
    duplicates = sorted({m for m in models if models.count(m) > 1})
    if duplicates:
        raise RecipeError(f"Model występuje w kilku runach: {duplicates} — podaj jeden run na model")
    version = versions.pop()
    if version != config.DATA_VERSION:
        log.warning(f"Runy z wersji danych '{version}', kod jest w wersji '{config.DATA_VERSION}'")
    samples = [r.name for r in runs if r.info.get("train_sample") or r.info.get("eval_sample")]
    if samples:
        log.warning("Runy trenowane / oceniane na PRÓBKACH — wyniki NIE do raportowania: " + ", ".join(samples))
    unfinished = [r.name for r in runs if not r.done]
    if unfinished:
        log.warning("Runy bez DONE.json (nieukończone): " + ", ".join(unfinished))
    strategies = {r.info.get("split_strategy") for r in runs if r.info.get("split_strategy")}
    tags = {r.info.get("tag") for r in runs}
    return {
        "feature_set": feature_set,
        "data_version": version,
        "data_fingerprint": fingerprints.pop() if fingerprints else None,
        "split_strategy": strategies.pop() if len(strategies) == 1 else None,
        "samples": samples,
        "tag": tags.pop() if len(tags) == 1 else None,
    }


@dataclass
class SplitData:
    """Wyrównane predykcje jednego splitu (kolejność wierszy = rosnące row_id)."""

    y: np.ndarray
    probas: dict[str, np.ndarray]
    vc: np.ndarray | None
    meta: pd.DataFrame


def load_split(runs: list[Run], split: str) -> SplitData:
    try:
        y, probas, vc = aligned_probas(runs, split, return_vc=True)
        meta = aligned_meta(runs, split)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise RecipeError(f"Predykcje '{split}' nie są wyrównane między runami: {exc}") from exc
    if len(meta) != len(y) or not np.array_equal(meta["target"].to_numpy(), y):
        raise RecipeError(f"Kolumny meta '{split}' nie zgadzają się z etykietami predykcji")
    return SplitData(y=y, probas=probas, vc=vc, meta=meta)


# ─────────────────────────────────────────
# Receptura
# ─────────────────────────────────────────
def _read_weights_spec(spec: str) -> tuple[Any, str]:
    """--weights jako JSON wprost albo ścieżka do pliku JSON → (obiekt, opis pochodzenia)."""
    path = Path(spec)
    try:
        if not spec.startswith("{") and path.exists():
            return json.loads(path.read_text(encoding="utf-8")), f"z pliku {path.name}"
        return json.loads(spec), "ręczne"
    except (OSError, ValueError) as exc:
        raise RecipeError(f"--weights: niepoprawny JSON / plik ({exc})") from exc


def _is_backend_config(payload: Any) -> bool:
    """Format backend_ensemble_weights.json: {"weights": {...}, "method", "temperature", "rule", parametry reguły}."""
    return isinstance(payload, dict) and bool(set(payload) & BACKEND_KEYS)


def weights_file_recipe(spec: str) -> dict[str, Any] | None:
    """
    Ustawienia receptury zapisane w pliku wag w formacie backendu (tylko klucze obecne w pliku:
    method, rule (+ jej parametry), temperature). Zwykłe {model: waga}, 'equal', 'optimize' → None.
    """
    spec = spec.strip()
    if spec in ("equal", "optimize"):
        return None
    payload, _ = _read_weights_spec(spec)
    if not _is_backend_config(payload):
        return None
    unknown = sorted(set(payload) - BACKEND_KEYS)
    if unknown:
        raise RecipeError(f"--weights: nieznane klucze {unknown} w pliku w formacie backendu "
                          f"(dozwolone: {sorted(BACKEND_KEYS)})")
    recipe = {k: payload[k] for k in ("method", "rule", "temperature") if k in payload}
    if "method" in recipe and recipe["method"] not in ("mean", "geometric"):
        raise RecipeError(f"--weights: \"method\" w pliku musi być mean | geometric (jest {recipe['method']!r})")
    if "rule" in recipe or set(payload) & RULE_PARAM_KEYS:
        if recipe.get("rule") not in RULES:
            raise RecipeError(f"--weights: \"rule\" w pliku musi być jedną z {list(RULES)} "
                              f"(jest {recipe.get('rule')!r})")
        try:
            recipe["rule_spec"] = normalize_spec({k: payload[k] for k in ("rule", *RULE_PARAM_KEYS) if k in payload})
        except ValueError as exc:
            raise RecipeError(f"--weights: reguła w pliku jest niepoprawna ({exc})") from exc
    t = recipe.get("temperature")
    if t is not None and (isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0):
        raise RecipeError(f"--weights: \"temperature\" w pliku musi być liczbą > 0 albo null (jest {t!r})")
    return recipe


@dataclass
class RecipeOptions:
    """Ustawienia receptury po uzgodnieniu linii poleceń z plikiem wag."""

    method: str
    temperature_mode: str
    rules: list[str]
    main_rule: str
    file_temperature: float | None = None
    file_rule_spec: dict[str, Any] | None = None
    adopted: tuple[str, ...] = ()

    @property
    def backend_rule(self) -> str:
        """Dawna nazwa reguły głównej (reguła zapisywana w pliku dla backendu)."""
        return self.main_rule


def resolve_options(args: argparse.Namespace, file_recipe: dict[str, Any] | None) -> RecipeOptions:
    """
    Metoda, temperatura i reguły: jawna opcja > ustawienie z pliku wag (format backendu) > domyślne.
    Jawna opcja sprzeczna z plikiem → odmowa (test oceniono by inną recepturą niż ta w pliku,
    a nowy plik dla backendu nie odpowiadałby recepturze wybranej na VAL).
    """
    fr = file_recipe or {}
    adopted: list[str] = []

    def pick(option: str, cli: str | None, from_file: str | None, default: str) -> str:
        if from_file is None:
            return cli or default
        if cli is not None and cli != from_file:
            raise RecipeError(f"--{option} {cli} jest sprzeczne z plikiem wag ({option} = {from_file}); usuń "
                              f"opcję, żeby przejąć ustawienie z pliku, albo podaj wagi bez pliku backendu")
        if cli is None:
            adopted.append(option)
        return from_file

    method = pick("method", args.method, fr.get("method"), DEFAULT_METHOD)
    file_mode = None if "temperature" not in fr else ("none" if fr["temperature"] is None else "fit")
    temperature_mode = pick("temperature", args.temperature, file_mode, DEFAULT_TEMPERATURE)
    rule_arg = args.rule or DEFAULT_RULE
    rules = list(RULE_SETS.get(rule_arg, [rule_arg]))
    if args.main_rule and args.backend_rule and args.main_rule != args.backend_rule:
        raise RecipeError(f"--main-rule {args.main_rule} i --backend-rule {args.backend_rule} są sprzeczne "
                          f"(to ta sama opcja — podaj jedną)")
    cli_main = args.main_rule or args.backend_rule or (rule_arg if rule_arg in RULES else None)
    file_rule = fr.get("rule")
    if file_rule is not None and cli_main is not None and cli_main != file_rule:
        raise RecipeError(f"Reguła {cli_main} (z --main-rule / --backend-rule / --rule) jest sprzeczna z plikiem "
                          f"wag (rule = {file_rule}); usuń opcję albo podaj wagi bez pliku backendu")
    if cli_main is None and file_rule is not None:
        adopted.append("rule")
    if cli_main or file_rule:
        main_rule = cli_main or file_rule
    elif DEFAULT_MAIN_RULE in rules:
        main_rule = DEFAULT_MAIN_RULE
    else:
        main_rule = "cost" if "cost" in rules else rules[0]
    if main_rule not in rules:
        raise RecipeError(f"Reguła główna '{main_rule}' nie jest oceniana w raporcie (--rule {rule_arg}) "
                          f"— dodaj ją (--rule {main_rule} albo all)")
    t = fr.get("temperature")
    return RecipeOptions(method=method, temperature_mode=temperature_mode, rules=rules, main_rule=main_rule,
                         file_temperature=float(t) if t is not None else None,
                         file_rule_spec=fr.get("rule_spec"), adopted=tuple(adopted))


def resolve_weights(
    spec: str, names: list[str], P_val: dict[str, np.ndarray], y_val: np.ndarray, method: str
) -> tuple[dict[str, float], str]:
    """Wagi (znormalizowane do sumy 1, każdy model z listy) i opis ich pochodzenia."""
    spec = spec.strip()
    if spec == "equal":
        raw: dict[str, Any] = dict.fromkeys(names, 1.0)
        mode = "równe"
    elif spec == "optimize":
        log.info("Dobór wag: minimalny log loss na VAL (optimize_weights)…")
        raw = optimize_weights(P_val, y_val, method=method)
        mode = "optimize (minimalny log loss na VAL)"
    else:
        payload, mode = _read_weights_spec(spec)
        if _is_backend_config(payload):
            if "weights" not in payload:                   # jak backend: brak wag = równe
                payload = dict.fromkeys(names, 1.0)
                mode += " (bez \"weights\" → równe)"
            else:
                payload = payload["weights"]
        if not isinstance(payload, dict) or not payload:
            raise RecipeError("--weights: oczekiwano obiektu {model: waga}")
        raw = payload
    unknown = sorted(set(raw) - set(names))
    if unknown:
        raise RecipeError(f"--weights: modele spoza wybranych runów: {unknown} (dostępne: {names})")
    weights: dict[str, float] = {}
    for name in names:
        try:
            w = float(raw.get(name, 0.0))
        except (TypeError, ValueError) as exc:
            raise RecipeError(f"--weights: waga '{name}' nie jest liczbą") from exc
        if not math.isfinite(w) or w < 0:
            raise RecipeError(f"--weights: waga '{name}' musi być skończona i ≥ 0")
        weights[name] = w
    total = sum(weights.values())
    if total <= 0:
        raise RecipeError("--weights: co najmniej jedna waga musi być > 0")
    skipped = [n for n, w in weights.items() if w == 0]
    if skipped:
        log.info("Modele z wagą 0 (poza połączeniem, ale oceniane osobno): " + ", ".join(skipped))
    return {n: w / total for n, w in weights.items()}, mode


def recipe_hash(run_names: list[str], weights: dict[str, float], opts: RecipeOptions,
                dopts: DecisionOptions | None = None) -> str:
    """Krótki skrót receptury (runy, wagi, metoda, temperatura, reguły, cel T…) — do rozpoznania powtórek."""
    payload = {"runs": sorted(run_names), "weights": {k: round(v, 6) for k, v in sorted(weights.items())},
               "method": opts.method, "temperature": opts.temperature_mode, "rules": sorted(opts.rules),
               "backend_rule": opts.main_rule}
    if dopts is not None:
        payload["decision"] = {"target": round(dopts.target, 6), "guarantee": dopts.guarantee,
                               "safety_net": dopts.safety_net, "population": dopts.population}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _registry_path() -> Path:
    return Path(config.RUNS_DIR) / REGISTRY_FILE


def _read_registry() -> list[dict[str, Any]]:
    path = _registry_path()
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def register_report(entry: dict[str, Any]) -> None:
    """Dopisz raport do rejestru runs/final_reports.jsonl (zapis atomowy całego pliku)."""
    path = _registry_path()
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if old and not old.endswith("\n"):
        old += "\n"
    write_text_atomic(path, old + json.dumps(_jsonable(entry), ensure_ascii=False, sort_keys=True) + "\n")


def previous_reports(run_names: list[str], current_hash: str, out_dir: Path,
                     dirs: list[Path]) -> list[dict[str, Any]]:
    """
    Wcześniejsze raporty końcowe dla dokładnie tego samego zestawu runów: z rejestru
    (każde --out) i — dla starszych raportów sprzed rejestru — z katalogów */recipe.json.
    Zwraca [{"path", "created_at", "recipe_hash", "same_recipe"}] (bez bieżącego katalogu).
    """
    wanted = sorted(run_names)
    here = Path(out_dir).resolve()
    found: dict[str, dict[str, Any]] = {}

    def add(path: str, created_at: Any, digest: Any) -> None:
        if Path(path).resolve() == here:
            return
        found.setdefault(str(Path(path).resolve()), {
            "path": path, "created_at": created_at, "recipe_hash": digest,
            "same_recipe": None if digest is None else digest == current_hash})

    for entry in _read_registry():
        if sorted(entry.get("runs") or []) == wanted and entry.get("out_dir"):
            add(str(entry["out_dir"]), entry.get("created_at"), entry.get("recipe_hash"))
    for root in dirs:
        for recipe in sorted(Path(root).glob("*/recipe.json")):
            try:
                payload = json.loads(recipe.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(payload, dict) or "best_single_by" not in payload:   # nie raport końcowy
                continue
            if sorted(r.get("name") for r in payload.get("runs", [])) == wanted:
                add(str(recipe.parent), payload.get("created_at"), payload.get("recipe_hash"))
    return sorted(found.values(), key=lambda e: (str(e.get("created_at") or ""), e["path"]))


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def warn_test_once(prior: list[dict[str, Any]]) -> None:
    bar = "!" * 78
    log.warning(bar)
    log.warning("UWAGA: ten raport ocenia zbiór TESTOWY. Generuj go RAZ, dla ostatecznej receptury")
    log.warning("wybranej na VAL. Porównywanie receptur na teście i wybór najlepszej zawyża wyniki.")
    if prior:
        log.warning(f"Dla tych samych runów TEST oceniono już {len(prior)} raz(y):")
        for entry in prior:
            same = {True: "ta sama receptura", False: "INNA receptura", None: "receptura nieznana"}[
                entry.get("same_recipe")]
            log.warning(f"  {entry['path']} ({entry.get('created_at') or '?'}; {same})")
        if any(e.get("same_recipe") is not True for e in prior):
            log.warning("Receptura się zmieniła — opisz to w pracy (test przestaje być niezależny).")
    log.warning(bar)


def _ci_lookup(res: BootstrapSamples | None, name: str, rule: str) -> dict[str, dict[str, Any]]:
    if res is None or name not in res.names or rule not in res.rules:
        return {}
    return ci_to_dict(res.ci(name, rule))


def shown_rules(rules: list[str], main_rule: str) -> list[str]:
    """Reguły w tabelach szczegółowych: główna + argmax (punkt odniesienia bez warstwy decyzyjnej)."""
    return [main_rule] + (["argmax"] if "argmax" in rules and main_rule != "argmax" else [])


def run_section(name: str, timings: dict[str, float], fn: Callable[[], Any]) -> Any:
    """
    Sekcja dodatkowa: błąd nie przerywa raportu — sekcja dostaje {"skipped": opis błędu}
    (ostrzeżenie w logu), a pozostałe liczby i plik dla backendu powstają normalnie.
    """
    t = time.perf_counter()
    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001 — raport końcowy ma powstać mimo błędu jednej sekcji
        log.warning(f"Sekcja '{name}' pominięta z powodu błędu: {type(exc).__name__}: {exc}")
        out = {"skipped": f"błąd obliczeń ({type(exc).__name__}: {exc})"}
    timings[f"{name}_s"] = round(time.perf_counter() - t, 2)
    return out


# ─────────────────────────────────────────
# Wykresy
# ─────────────────────────────────────────
def make_figures(
    fig_dir: Path, y: np.ndarray, vc: np.ndarray | None, p_raw: np.ndarray, p_final: np.ndarray,
    temperature: float | None, rules: list[str], decisions: dict[str, np.ndarray], res_all: BootstrapSamples,
    res_vc: BootstrapSamples | None, order: list[str], n_bins: int,
) -> list[str]:
    from src.evaluation.visualizations import (
        plot_confusion_matrix,
        plot_metric_ci,
        plot_reliability,
    )

    paths: list[Path | None] = []
    sets = {"przed T": p_raw, f"po T (T = {temperature:.3f})": p_final} if temperature is not None else \
        {BLEND: p_final}
    paths.append(plot_reliability(y, sets, path=fig_dir / "reliability_test_blend.png", n_bins=n_bins,
                                  title="Kalibracja połączenia na TEST"))
    if vc is not None and vc.any():
        sets_vc = {k: v[vc] for k, v in sets.items()}
        paths.append(plot_reliability(y[vc], sets_vc, path=fig_dir / "reliability_test_blend_vc.png",
                                      n_bins=n_bins, title="Kalibracja połączenia na TEST — podgrupa vc"))
    for rule in rules:
        pred = decisions[rule]
        label = RULE_LABELS[rule]
        paths.append(plot_confusion_matrix(y, pred, normalize=None,
                                           save_path=fig_dir / f"confusion_test_{rule}_counts.png",
                                           title=f"Macierz pomyłek — {BLEND}, TEST, reguła {label} (liczby)"))
        paths.append(plot_confusion_matrix(y, pred, normalize="true",
                                           save_path=fig_dir / f"confusion_test_{rule}_rowpct.png",
                                           title=f"Macierz pomyłek — {BLEND}, TEST, reguła {label}"))
        if vc is not None and vc.any():
            paths.append(plot_confusion_matrix(y[vc], pred[vc], normalize="true",
                                               save_path=fig_dir / f"confusion_test_vc_{rule}_rowpct.png",
                                               title=f"Macierz pomyłek — {BLEND}, TEST vc, reguła {label}"))
        rows = []
        for group, res in (("całość", res_all), ("vc", res_vc)):
            if res is None:
                continue
            for name in order:
                ci = res.ci(name, rule).set_index("metric")
                for metric in ("quadratic_weighted_kappa", "undertriage_rate"):
                    r = ci.loc[metric]
                    rows.append({"label": name, "metric": metric, "group": group, "value": r["value"],
                                 "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
        paths.append(plot_metric_ci(pd.DataFrame(rows), fig_dir / f"forest_test_{rule}.png",
                                    title=f"TEST, reguła {label}: wartość i 95% przedział ufności",
                                    highlight=[BLEND],
                                    metric_labels={"quadratic_weighted_kappa": "QWK ↑",
                                                   "undertriage_rate": "niedotriażowanie ESI 1–2 ↓"}))
    rows = []
    for group, res in (("całość", res_all), ("vc", res_vc)):
        if res is None:
            continue
        for name in order:
            ci = res.ci(name, rules[0]).set_index("metric")
            for metric in ("log_loss", "ece"):
                r = ci.loc[metric]
                rows.append({"label": name, "metric": metric, "group": group, "value": r["value"],
                             "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    paths.append(plot_metric_ci(pd.DataFrame(rows), fig_dir / "forest_test_probabilities.png",
                                title="TEST: jakość prawdopodobieństw (95% przedziały ufności)", highlight=[BLEND],
                                metric_labels={"log_loss": "log loss ↓", "ece": "ECE top-label ↓"}))
    return [Path(p).name for p in paths if p is not None]


# ─────────────────────────────────────────
# Raport
# ─────────────────────────────────────────
def _rule_params_text(spec: dict[str, Any], fmt: Fmt) -> str:
    rule = spec.get("rule")
    if rule == "target":
        cuts = ", ".join(fmt.num(c, 3) for c in spec["cutpoints"])
        q = spec.get("safety_q")
        return f"progi EL = [{cuts}]" + (f", siatka q = {fmt.num(q, 3)}" if q is not None else ", bez siatki")
    if rule == "bayes_target":
        return f"λ = {fmt.num(spec['lam'], 3)}"
    if rule == "cost":
        return "macierz kosztów repo (niedotriażowanie 3×, kwadrat odległości)"
    return "bez parametrów"


def render_recipe_decision(ctx: dict[str, Any], fmt: Fmt) -> list[str]:
    """Reguły decyzyjne w recepturze: ustawienia, parametry dopasowane na VAL i wynik na VAL."""
    dec = ctx["decision"]
    dopts: DecisionOptions = ctx["dopts"]
    fitted: FittedRules = ctx["fitted_blend"]
    pop = {"vc": "podgrupa vc VAL (komplet HR/SBP/RR, jak TOPSOR)", "all": "cały VAL"}[fitted.population]
    lines = [
        "### Reguły decyzyjne (dopasowane na VAL, na prawdopodobieństwach po łączeniu i temperaturze)",
        "",
        f"- reguła główna (backend, tabele szczegółowe): **{RULE_LABELS[dopts.main_rule]}** "
        f"(`{dopts.main_rule}`); cel: {dopts.describe()}; dopasowanie: {pop}"
        + (f" — {fitted.note}" if fitted.note else ""),
    ]
    for rule in dopts.rules:
        lines.append(f"- {RULE_LABELS[rule]} (`{rule}`): {_rule_params_text(fitted.specs[rule], fmt)}")
    lines.append("")
    rows = []
    for rule in dopts.rules:
        v = dec["val"][rule]
        a = v["all"]
        row = [RULE_LABELS[rule], v["kind"], fmt.num(a["qwk"]), fmt.rate(a["k_under"], a["n_high"]),
               fmt.rate(a["k_crit"], a["n_esi1"]), fmt.pct(a["overtriage"])]
        f = v.get("fit_population")
        if f is not None:
            row += [fmt.num(f["qwk"]), fmt.rate(f["k_under"], f["n_high"])]
        rows.append(row)
    header = ["reguła", "ocena na VAL", "QWK", "niedotriażowanie", "kryt. pominięcie", "przetriażowanie"]
    if fitted.mask is not None:
        header += ["QWK (vc)", "niedotriażowanie (vc)"]
    hits = []
    for rule in dopts.rules:
        v = dec["val"][rule]
        if rule not in rs.FITTED_RULES or not v["kind"].startswith("cross-fit"):
            continue
        u = (v.get("fit_population") or v["all"]).get("undertriage")
        if u is not None:
            hits.append(f"{RULE_LABELS[rule]} {fmt.pct(u)} ({fmt.pp(u - dopts.target)} względem T)")
    where = "w populacji dopasowania" if fitted.mask is not None else "na całym VAL"
    hit_text = ""
    if hits:
        hit_text = f"Niedotriażowanie z cross-fittingu {where}: " + "; ".join(hits) + ". "
    elif any(r in rs.FITTED_RULES for r in dopts.rules):
        hit_text = "Cross-fitting niemożliwy — trafienia w cel T na VAL nie oceniono. "
    lines += [md_table(header, rows), "",
              "Reguły dopasowywane (progi T, Bayes T) oceniono na VAL cross-fittingiem: parametry z jednej połowy "
              "VAL, wynik na drugiej — to przybliża wynik na nowych danych. Na TEST trafiają parametry dopasowane "
              f"na CAŁYM VAL. {hit_text}`--guarantee` przesuwa próg tak, by górna granica Clopper–Pearsona na "
              "VAL była ≤ T.", ""]
    return lines


def build_report(ctx: dict[str, Any], fmt: Fmt) -> str:
    """report.md: tabele gotowe do wklejenia do pracy (po polsku)."""
    rules: list[str] = ctx["rules"]
    shown: list[str] = ctx["shown_rules"]
    main_rule: str = ctx["main_rule"]
    order: list[str] = ctx["order"]
    res_all: BootstrapSamples = ctx["res_all"]
    res_vc: BootstrapSamples | None = ctx["res_vc"]
    recipe = ctx["recipe"]
    data = ctx["data"]
    n_boot = res_all.n_boot
    conf = int(round(100 * (1 - res_all.alpha)))
    lines: list[str] = [
        f"# Raport końcowy — zestaw `{recipe['feature_set']}` (TEST)",
        "",
        "> **Zbiór TESTOWY — ocena jednorazowa.** Recepturę (modele, wagi, łączenie, temperaturę, regułę "
        "decyzyjną i jej parametry) wybrano wcześniej na VAL. Raport generuje się raz, dla ostatecznej receptury; "
        "kolejne uruchomienia z innymi recepturami i wybór najlepszej na teście zawyżają wyniki.",
        "",
    ]
    if recipe.get("samples"):
        lines += ["> **Runy na próbkach** (`--sample` / `--eval-sample`) — liczby nie nadają się do pracy: "
                  + ", ".join(f"`{s}`" for s in recipe["samples"]), ""]
    if recipe.get("previous_reports"):
        lines += [f"> Dla tych samych runów powstało wcześniej {len(recipe['previous_reports'])} raport(ów) "
                  f"końcowych — zob. `recipe.json` → `previous_reports`.", ""]

    # --- Receptura ---
    weights_rows = [[f"`{r['name']}`", r["model"], fmt.num(recipe["weights"][r["model"]], 4),
                     fmt.num(recipe["per_model_temperature"].get(r["model"]), 3)] for r in recipe["runs"]]
    t_text = fmt.num(recipe["temperature"], 4) if recipe["temperature"] is not None else "brak (T = 1)"
    lines += [
        "## Receptura (wybrana na VAL)",
        "",
        md_table(["run", "model", "waga w połączeniu", "T własna (na VAL)"], weights_rows),
        "",
        f"- wagi: {recipe['weights_mode']}; łączenie: `{recipe['method']}`"
        f" ({'średnia arytmetyczna' if recipe['method'] == 'mean' else 'średnia geometryczna'} prawdopodobieństw)",
        f"- temperatura połączenia (dopasowana na VAL): {t_text}",
        f"- reguły w raporcie: {', '.join(RULE_LABELS[r] for r in rules)}; reguła w pliku dla backendu: "
        f"`{main_rule}`",
        f"- najlepszy pojedynczy model (najniższy log loss na VAL): **{recipe['best_single_model']}**",
        f"- dane: wersja `{recipe['data_version']}`, odcisk `{recipe['data_fingerprint']}`, "
        f"podział `{recipe.get('split_strategy') or '?'}`",
        "",
        "Jakość prawdopodobieństw połączenia na VAL (punkt odniesienia — na tych danych dobrano recepturę): "
        f"log loss {fmt.num(ctx['val_summary']['log_loss'])}, ECE {fmt.num(ctx['val_summary']['ece'])}.",
        "",
    ]
    val = ctx["val_summary"]
    if val.get("crossfit") is not None:
        cf = val["crossfit"]
        lines += [f"Temperatura oceniona uczciwie na VAL (2 foldy: T dopasowana na jednym, oceniana na drugim; "
                  f"T foldów {', '.join(fmt.num(t, 3) for t in cf['temperatures'])}): log loss "
                  f"{fmt.num(cf['raw']['log_loss'])} → {fmt.num(cf['calibrated']['log_loss'])}, ECE "
                  f"{fmt.num(cf['raw']['ece_top_label'])} → {fmt.num(cf['calibrated']['ece_top_label'])}.", ""]
    lines += render_recipe_decision(ctx, fmt)
    lines += rs.render_tradeoff(ctx["tradeoff"], fmt, ctx["dopts"])

    # --- Dane ---
    counts = data["esi_counts"]
    lines += [
        "## Zbiór testowy",
        "",
        md_table(["", "n", *[f"ESI {k}" for k in range(1, N_CLASSES + 1)]],
                 [["cała populacja", _int(data["n"]), *[_int(c) for c in counts["all"]]]]
                 + ([["podgrupa vc", _int(data["n_vc"]), *[_int(c) for c in counts["vc"]]]]
                    if counts.get("vc") else [])),
        "",
        "*vc* = pacjenci z kompletem tętna, ciśnienia skurczowego i częstości oddechów przy triażu "
        "(`vitals_complete == 1`) — populacja jak w TOPSOR, gdzie pomiar jest obowiązkowy.",
        "",
        "## Jak czytać",
        "",
        f"- **`0.712 [0.701; 0.723]`** — wartość na całym zbiorze testowym i {conf}% przedział ufności "
        f"(bootstrap percentylowy: {n_boot} losowań pacjentów ze zwracaniem, osobno w każdej klasie ESI, "
        f"więc rzadkie ESI 1 jest w każdym losowaniu). Przedział pokazuje, o ile wynik mógłby się różnić na "
        f"innej próbce pacjentów z tej samej populacji; nie obejmuje zmienności samego treningu (tę mierzy "
        f"analiza seedów, sekcja 13).",
        "- Odsetki pacjentów (trafność, niedotriażowanie, czułość…) w procentach, z **dokładnym przedziałem "
        "Clopper–Pearsona** z liczebności, a nie z bootstrapu. Przy rzadkich zdarzeniach bootstrap daje "
        "przedział za wąski (0 zdarzeń → [0; 0]); Clopper–Pearson przy 0/38 daje [0; 9,3%]. Przy metrykach "
        "ESI podano też liczebność **k/n** (np. `(0/38)` = 0 krytycznych pominięć na 38 pacjentów ESI 1). "
        "QWK, AUC, log loss, Brier, ECE i koszt jako liczby (bootstrap). ↑ — im więcej, tym lepiej; "
        "↓ — im mniej, tym lepiej.",
        "- **Porównanie modeli**: nakładające się przedziały w tabeli 2 NIE oznaczają braku różnicy. "
        "O różnicy rozstrzyga tabela 3 (sparowany bootstrap: oba modele na tych samych losowaniach), "
        "a o „wyższości” — reguła C1 (przedział bez zera, |Δ| większa od minimalnej istotnej różnicy "
        "i od 2 × SD seedów).",
        "- **ECE** (expected calibration error) — średnia różnica między deklarowaną pewnością modelu "
        "a jego faktyczną trafnością (0 = idealnie; 0,05 = model myli się średnio o 5 punktów procentowych "
        "w ocenie własnej pewności). **T** (temperatura) — jeden parametr dopasowany na VAL: T > 1 "
        "„uspokaja” zbyt pewny model, T < 1 go wyostrza; nie zmienia najbardziej prawdopodobnego poziomu, "
        "zmienia decyzje pozostałych reguł. ECE z binów jest zawyżone przez szum: nawet idealnie skalibrowany "
        "model ma ECE > 0 (tym większe, im mniejsza grupa), więc jego przedział nie zawiera 0. Punktem "
        "odniesienia jest wiersz „ECE idealnie skalibrowanego modelu” w sekcji 4; ECE modeli porównuj "
        "parami na tych samych pacjentach.",
        "- **Reguły decyzyjne** (te same prawdopodobieństwa, różne decyzje): *argmax* — najbardziej "
        "prawdopodobny poziom; *koszt* — najmniejszy oczekiwany koszt (niedotriażowanie 3× droższe, koszt "
        "rośnie z kwadratem odległości); *progi T* — oczekiwany poziom E[ESI] dzielony 4 progami: próg "
        "ESI 2/3 ustawiony tak, by niedotriażowanie na VAL wynosiło T, pozostałe maksymalizują QWK, "
        "opcjonalnie z siatką bezpieczeństwa (P(ESI 1)+P(ESI 2) ≥ q → najwyżej ESI 3); *Bayes T* — decyzja "
        "minimalizująca stratę (p − t)² + λ·[ESI 1–2 → ESI 3–5], λ dobrane do celu T. Metrykę główną "
        "czytaj jako **QWK przy zadanym niedotriażowaniu**; oczekiwany koszt liczony macierzą k = 3 faworyzuje "
        "regułę kosztową z definicji.",
        "- ESI 1 jest rzadkie — przedziały czułości ESI 1 i krytycznych pominięć są szerokie (nawet przy "
        "0 zdarzeń górna granica jest > 0); „—” oznacza metrykę nieokreśloną (np. brak ESI 1 w podgrupie).",
        "",
    ]

    # --- 1. Tabela główna ---
    lines += [f"## 1. Wynik końcowy: {BLEND} na TEST ({conf}% CI)", ""]
    for res, title in ((res_all, "cała populacja"), (res_vc, "podgrupa vc")):
        if res is None:
            continue
        header = ["metryka", *[RULE_LABELS[r] + (" (główna)" if r == main_rule else "") for r in rules]]
        rows = []
        for metric in DEFAULT_CI_METRICS:
            if metric in rs.PROBA_METRICS and len(rules) > 1:
                continue
            rows.append([METRIC_LABELS.get(metric, metric),
                         *[fmt.ci(metric, _ci_lookup(res, BLEND, rule).get(metric)) for rule in rules]])
        lines += [f"### 1.{1 if res is res_all else 2}. {title.capitalize()}", "", md_table(header, rows), ""]
        if len(rules) > 1:
            ci = _ci_lookup(res, BLEND, rules[0])
            proba_text = "; ".join(f"{SHORT_LABELS.get(m, m)} {fmt.ci(m, ci.get(m))}" for m in COMPARE_PROBA_METRICS)
            lines += [f"Metryki prawdopodobieństw (nie zależą od reguły): {proba_text}.", ""]

    # --- 2. Połączenie vs modele ---
    lines += [f"## 2. Połączenie i pojedyncze modele na TEST ({conf}% CI)", "",
              "Każdy pojedynczy model z własną temperaturą i własnymi regułami dopasowanymi na VAL (ta sama "
              "obróbka co połączenie)." if recipe["temperature"] is not None else
              "Prawdopodobieństwa bez temperatury; reguły każdego modelu dopasowane na jego VAL.",
              "Tabele dla reguły głównej i argmax; pozostałe reguły: `tables/ci_all_models.csv`.", ""]
    sub = 0
    for res, title in ((res_all, "cała populacja"), (res_vc, "podgrupa vc")):
        if res is None:
            continue
        for rule in shown:
            rows = []
            for name in order:
                ci = _ci_lookup(res, name, rule)
                label = f"**{name}**" if name == BLEND else name
                rows.append([label, *[fmt.ci(m, ci.get(m)) for m in COMPARE_DECISION_METRICS]])
            sub += 1
            lines += [f"### 2.{sub}. {title}, reguła {RULE_LABELS[rule]}", "",
                      md_table(["model", *[SHORT_LABELS[m] for m in COMPARE_DECISION_METRICS]], rows), ""]
        rows = []
        for name in order:
            ci = _ci_lookup(res, name, rules[0])
            label = f"**{name}**" if name == BLEND else name
            rows.append([label, *[fmt.ci(m, ci.get(m)) for m in COMPARE_PROBA_METRICS]])
        sub += 1
        lines += [f"### 2.{sub}. {title}, jakość prawdopodobieństw", "",
                  md_table(["model", *[SHORT_LABELS[m] for m in COMPARE_PROBA_METRICS]], rows), ""]

    # --- 3. Sparowane różnice ---
    best = recipe["best_single_model"]
    sup = ctx["superiority"]
    lines += [
        f"## 3. Czy połączenie jest lepsze od najlepszego pojedynczego modelu ({best})?",
        "",
        f"Sparowany bootstrap ({n_boot} losowań; w każdym oba modele na tych samych pacjentach). "
        f"Δ = {BLEND} − {best}. Dla QWK dodatnia Δ = połączenie lepsze; dla niedotriażowania i log loss "
        f"UJEMNA Δ = połączenie lepsze. Przedział bez zera = różnica istotna (p < {res_all.alpha:g}). "
        f"Kolumna **wyższość (C1)**: „TAK — <zwycięzca>” tylko, gdy przedział wyklucza 0, |Δ| > minimalna "
        f"istotna różnica (QWK 0,01; niedotriażowanie 1 pp; log loss 0,005) i |Δ| > 2 × SD seedów "
        f"(SD: {sup['source']}); zwycięzca wynika z kierunku metryki (może nim być {best}, nie połączenie).",
        "",
    ]
    rows = []
    for group, table in ctx["paired"].items():
        if table is None:
            continue
        for rec in table.to_dict(orient="records"):
            rule = "—" if rec["rule"] == "*" else RULE_LABELS[rec["rule"]]
            verdict = {"A": f"lepsze: {BLEND}", "B": f"lepszy: {best}"}.get(rec["better"], rec["better"])
            key = f"{group}|{rec['rule']}|{rec['metric']}"
            rows.append([group, SHORT_LABELS.get(rec["metric"], rec["metric"]), rule,
                         fmt.ci(rec["metric"], {"value": rec["value_a"], "ci_low": None, "ci_high": None}),
                         fmt.ci(rec["metric"], {"value": rec["value_b"], "ci_low": None, "ci_high": None}),
                         fmt.delta(rec["metric"], rec["delta"], rec["ci_low"], rec["ci_high"]),
                         fmt.p(rec["p_two_sided"]), verdict, fmt._dc(sup["verdicts"].get(key, "—"))])
    lines += [md_table(["populacja", "metryka", "reguła", BLEND, best, f"Δ [{conf}% CI]", "p", "wniosek",
                        "wyższość (C1)"], rows),
              "", "pp = punkty procentowe. Pełne porównania z każdym modelem: `metrics.json` → "
              "`test.paired_all`.", ""]

    # --- 4. Kalibracja ---
    cal = ctx["calibration"]
    lines += ["## 4. Kalibracja prawdopodobieństw", ""]
    if recipe["temperature"] is not None:
        rows = []
        for label, key in (("ECE top-label ↓", "ece_top_label"), ("ECE top-label, biny kwantylowe ↓",
                                                                 "ece_top_label_quantile"),
                           ("MCE (najgorszy bin) ↓", "mce_top_label"), ("pewność − trafność", "overconfidence"),
                           ("log loss ↓", "log_loss"), ("Brier ↓", "brier")):
            rows.append([label, fmt.num(cal["test_raw"][key], 4), fmt.num(cal["test_calibrated"][key], 4)])
        rows.append(["ECE per poziom ESI (średnia) ↓", fmt.num(cal["test_raw"]["ece_classwise"]["mean"], 4),
                     fmt.num(cal["test_calibrated"]["ece_classwise"]["mean"], 4)])
        ref_raw, ref_cal = cal["test_raw"]["ece_reference"], cal["test_calibrated"]["ece_reference"]
        rows.append(["ECE idealnie skalibrowanego modelu (szum; średnia / 95. percentyl)",
                     f"{fmt.num(ref_raw['null_mean'], 4)} / {fmt.num(ref_raw['null_q95'], 4)}",
                     f"{fmt.num(ref_cal['null_mean'], 4)} / {fmt.num(ref_cal['null_q95'], 4)}"])
        rows.append(["p testu kalibracji (małe = gorzej niż szum)", fmt.p(ref_raw["p_value"]),
                     fmt.p(ref_cal["p_value"])])
        lines += [f"Połączenie na TEST przed i po temperaturze T = {fmt.num(recipe['temperature'], 4)} "
                  f"(dopasowanej na VAL):", "", md_table(["miara", "przed T", "po T"], rows), ""]
        cal_paired = ctx.get("calibration_paired")
        if cal_paired is not None:
            rows = [[SHORT_LABELS.get(r["metric"], r["metric"]),
                     fmt.delta(r["metric"], r["delta"], r["ci_low"], r["ci_high"]), fmt.p(r["p_two_sided"])]
                    for r in cal_paired.to_dict(orient="records")]
            lines += ["Zmiana po temperaturze (sparowany bootstrap, Δ = po − przed; ujemna = lepiej):", "",
                      md_table(["miara", f"Δ [{conf}% CI]", "p"], rows), ""]
    else:
        c = cal["test_calibrated"]
        ref = c["ece_reference"]
        lines += [f"Bez temperatury. ECE top-label {fmt.num(c['ece_top_label'], 4)} (idealnie skalibrowany model "
                  f"przy tej liczności: ~{fmt.num(ref['null_mean'], 4)}, p testu kalibracji {fmt.p(ref['p_value'])}), "
                  f"ECE per poziom {fmt.num(c['ece_classwise']['mean'], 4)}, log loss {fmt.num(c['log_loss'], 4)}.", ""]
    rows = [[name, fmt.num(v.get("temperature"), 3), fmt.num(v["ece_raw"], 4), fmt.num(v["ece_final"], 4)]
            for name, v in cal["per_model"].items()]
    lines += ["Modele (TEST): temperatura z VAL i ECE top-label przed / po:", "",
              md_table(["model", "T", "ECE przed", "ECE po"], rows), "",
              "Wykres: `figures/reliability_test_blend.png` (punkty poniżej przekątnej = model zbyt pewny siebie). "
              "Kalibracja per poziom ESI i oddział: sekcja 11.", ""]

    # --- 5. Macierze pomyłek ---
    lines += ["## 5. Macierze pomyłek połączenia (TEST, cała populacja)", "",
              "Wiersze — prawdziwy poziom (ESI z triażu), kolumny — decyzja modelu. Pozostałe reguły: "
              "`metrics.json` → `test.confusion`.", ""]
    for rule in shown:
        cm = np.asarray(ctx["confusion"]["all"][rule])
        names = [f"ESI {k}" for k in range(1, N_CLASSES + 1)]
        rows = [[names[i], *[_int(v) for v in cm[i]], _int(cm[i].sum())] for i in range(len(cm))]
        lines += [f"Reguła {RULE_LABELS[rule]} — liczby:", "",
                  md_table(["prawda \\ decyzja", *names, "razem"], rows), ""]
        row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        rows = [[names[i], *[fmt.pct(v) for v in row_pct[i]]] for i in range(len(cm))]
        lines += [f"Reguła {RULE_LABELS[rule]} — % wiersza (przekątna = czułość poziomu):", "",
                  md_table(["prawda \\ decyzja", *names], rows), ""]

    # --- 6. Oddziały ---
    sites = ctx["sites"]
    if sites:
        lines += ["## 6. Połączenie na oddziałach (TEST)", ""]
        for rule in shown:
            rows = []
            for site, info in sites.items():
                ci = info["ci"][rule]
                rows.append([site, _int(info["n"]), str(info["n_esi1"]),
                             *[fmt.ci(m, ci.get(m)) for m in SITE_METRICS]])
            lines += [f"Reguła {RULE_LABELS[rule]}:", "",
                      md_table(["oddział", "n", "ESI 1", *[SHORT_LABELS[m] for m in SITE_METRICS]], rows), ""]
        lines += ["Oddział z kilkoma wizytami ESI 1 ma bardzo szerokie przedziały metryk ESI 1. Reguła dopasowana "
                  "na VAL wszystkich oddziałów — różnice między oddziałami to w dużej mierze inny rozkład ESI "
                  "(sekcje 11 i 13.2).", ""]

    # --- 7. Przyjęcia ---
    adm = ctx["admission"]
    lines += ["## 7. Trafność względem przyjęcia do szpitala: model a pielęgniarka (TEST)", ""]
    if adm.get("auc") is None:
        lines += [f"Pominięte: {adm.get('reason', 'brak danych o przyjęciach')}.", ""]
    else:
        lines += [f"Źródło: {adm['source']}; n = {_int(adm['n'])}"
                  + f", przyjętych {fmt.pct(adm['admission_rate'])}. Przyjęcie to niezależny od etykiety "
                  f"wskaźnik ciężkości stanu: dobry poziom pilności powinien je przewidywać. AUC: "
                  f"wyższy wynik pilności = większe prawdopodobieństwo przyjęcia.", ""]
        rows = []
        for rec in adm["auc"].to_dict(orient="records"):
            if str(rec["score"]).startswith("Δ"):
                rows.append([rec["score"], fmt.delta("auc", rec["value"], rec["ci_low"], rec["ci_high"]),
                             fmt.p(rec.get("p_two_sided"))])
            else:
                rows.append([rec["score"], fmt.ci("auc", rec), ""])
        lines += [md_table(["wynik", f"AUC [{conf}% CI]", "p"], rows), "",
                  "Odsetek przyjętych według poziomu (ESI z triażu vs decyzja połączenia):", ""]
        by_level = adm["by_level"]
        header = list(by_level[0])
        rows = [[(fmt.pct(v) if k.startswith("przyjęci") else (_int(v) if isinstance(v, int) else str(v)))
                 for k, v in r.items()] for r in by_level]
        lines += [md_table(header, rows), ""]
        lines += rs.render_admission_extra(ctx.get("admission_subgroups"), ctx.get("discordance"), fmt,
                                           main_rule, conf)

    # --- 8–13. Sekcje dodatkowe ---
    lines += rs.render_subgroups(ctx["subgroups"], fmt, main_rule, conf, ctx.get("attributes_source"),
                                 population=ctx["fitted_blend"].population)
    errors = ctx["errors"]
    if errors.get("skipped"):
        lines += ["## 9. Analiza błędów", "", f"Pominięte: {errors['skipped']}.", ""]
    else:
        lines += rs.render_errors(errors, fmt, main_rule)
    dca_label, dca_adm = ctx["dca_label"], ctx["dca_admission"]
    if dca_label.get("skipped"):
        lines += ["## 10. Krzywe decyzyjne", "", f"Pominięte: {dca_label['skipped']}.", ""]
    else:
        lines += rs.render_dca(dca_label, dca_adm, fmt, main_rule, conf)
    cal_groups = ctx["calibration_groups"]
    if cal_groups.get("skipped"):
        lines += ["## 11. Kalibracja według poziomu ESI i oddziału", "", f"Pominięte: {cal_groups['skipped']}.", ""]
    else:
        lines += rs.render_calibration_groups(cal_groups, fmt)
    deferral, conformal = ctx["deferral"], ctx["conformal"]
    if deferral.get("skipped") or conformal.get("skipped"):
        reason = deferral.get("skipped") or conformal.get("skipped")
        lines += ["## 12. Odsyłanie niepewnych przypadków i zbiory conformal", "", f"Pominięte: {reason}.", ""]
    else:
        lines += rs.render_deferral(deferral, conformal, fmt, main_rule)
    lines += rs.render_analyses(ctx["analyses"], fmt, ctx["dopts"], ctx.get("analysis_figures"))

    # --- Pliki ---
    lines += [
        "## Pliki",
        "",
        "- `metrics.json` — wszystkie liczby (także przedziały dla każdego modelu, reguły i podgrupy, porównania "
        "parami z każdym modelem, kalibracja na VAL i TEST, metryki per oddział, sekcje 8–13),",
        "- `recipe.json` — receptura i runy (do odtworzenia), w tym dopasowane reguły decyzyjne (`decision`),",
        "- `backend_ensemble_weights.json` — wagi, metoda, temperatura i reguła główna z parametrami w formacie "
        "backendu (skopiuj jako `backend/models/ensemble_weights.json`),",
        "- `tables/*.csv` — tabele do arkusza / LaTeX-a (m.in. `ci_all_models.csv`, `subgroups.csv`, "
        "`tradeoff_val.csv`, `dca_*.csv`, `calibration_*.csv`, `deferral.csv`, `conformal.csv`),",
    ]
    if ctx["figures"]:
        lines += ["- `figures/`: " + ", ".join(f"`{f}`" for f in ctx["figures"]) + "."]
    lines += ["", "Lista kontrolna TRIPOD+AI i porównanie z literaturą: `docs/TRIPOD_AI.md`, `docs/LITERATURA.md`.",
              "", f"Wygenerowano {recipe['created_at']} (bootstrap {n_boot}×, ziarno {res_all.seed}).", ""]
    return "\n".join(lines)


def _frame(records: Any) -> pd.DataFrame | None:
    if isinstance(records, pd.DataFrame):
        return records
    if isinstance(records, list) and records:
        return pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in records])
    return None


def write_tables(table_dir: Path, ctx: dict[str, Any]) -> list[str]:
    """Tabele CSV (UTF-8) z surowymi liczbami — do arkusza albo LaTeX-a."""
    written = []
    res_all: BootstrapSamples = ctx["res_all"]
    res_vc: BootstrapSamples | None = ctx["res_vc"]
    frames = []
    for group, res in (("all", res_all), ("vc", res_vc)):
        if res is None:
            continue
        for name in ctx["order"]:
            for rule in ctx["rules"]:
                df = res.ci(name, rule)
                df.insert(0, "rule", rule)
                df.insert(0, "population", group)
                df.insert(0, "model", name)
                frames.append(df)
    tables: dict[str, pd.DataFrame | None] = {"ci_all_models.csv": pd.concat(frames, ignore_index=True)}
    paired = [t.assign(population=g) for g, t in ctx["paired"].items() if t is not None]
    if paired:
        tables["paired_blend_vs_best.csv"] = pd.concat(paired, ignore_index=True)
    if ctx["sites"]:
        rows = []
        for site, info in ctx["sites"].items():
            for rule, ci in info["ci"].items():
                for metric, r in ci.items():
                    rows.append({"site": site, "n": info["n"], "n_esi1": info["n_esi1"], "rule": rule,
                                 "metric": metric, **r})
        tables["sites_blend.csv"] = pd.DataFrame(rows)
    if ctx["admission"].get("auc") is not None:
        tables["admission_auc.csv"] = ctx["admission"]["auc"]
    tables["tradeoff_val.csv"] = ctx["tradeoff"].get("table")
    tables["subgroups.csv"] = _frame(ctx["subgroups"].get("rows"))
    tables["admission_subgroups.csv"] = _frame((ctx.get("admission_subgroups") or {}).get("rows"))
    tables["admission_discordance.csv"] = _frame(ctx.get("discordance"))
    errors = ctx["errors"]
    tables["errors_pairs.csv"] = _frame(errors.get("pairs"))
    tables["errors_complaints.csv"] = _frame(errors.get("complaints"))
    dca = ctx["dca_label"]
    if "all" in dca:
        frame = pd.DataFrame({"pt": dca["thresholds"]})
        for pop in ("all", "vc"):
            if pop in dca:
                frame[f"model_{pop}"] = dca[pop]["model"]
                frame[f"treat_all_{pop}"] = dca[pop]["treat_all"]
                for rule, v in dca[pop]["rules"].items():
                    frame[f"rule_{rule}_{pop}"] = v
        tables["dca_label.csv"] = frame
    adm = ctx["dca_admission"]
    if "model" in adm:
        frame = pd.DataFrame({"pt": adm["thresholds"], "model": adm["model"], "nurse_risk": adm["nurse_risk"],
                              "nurse_rule_esi12": adm["nurse_rule"], "treat_all": adm["treat_all"]})
        for rule, v in adm["rules"].items():
            frame[f"rule_{rule}"] = v
        tables["dca_admission.csv"] = frame
        tables["dca_admission_delta.csv"] = _frame(adm.get("delta_vs_nurse_rule"))
    cal = ctx["calibration_groups"]
    tables["calibration_classes.csv"] = _frame(cal.get("classes"))
    tables["calibration_esi12.csv"] = _frame(cal.get("esi12"))
    tables["deferral.csv"] = _frame(ctx["deferral"].get("rows"))
    tables["conformal.csv"] = _frame(ctx["conformal"].get("rows"))
    for name, df in tables.items():
        if df is None or df.empty:
            continue
        write_text_atomic(table_dir / name, df.to_csv(index=False))
        written.append(name)
    return written


# ─────────────────────────────────────────
# Główny przebieg
# ─────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    # 1. Receptura z linii poleceń / pliku wag, runy i ich kontrola
    try:
        opts = resolve_options(args, weights_file_recipe(args.weights))
        runs = resolve_runs(args)
        info = check_runs(runs, args.feature_set)
        val = load_split(runs, "val")
        test = load_split(runs, "test")
    except RecipeError as exc:
        log.error(f"ODMOWA: {exc}")
        return EXIT_REFUSED
    names = list(val.probas)
    if names != list(test.probas):
        log.error("ODMOWA: różne zestawy modeli w val i test")
        return EXIT_REFUSED
    log.info("Runy: " + ", ".join(r.name for r in runs))
    rules, main_rule, method = opts.rules, opts.main_rule, opts.method
    shown = shown_rules(rules, main_rule)
    dopts = DecisionOptions(main_rule=main_rule, rules=rules, target=float(args.target_undertriage),
                            guarantee=bool(args.guarantee), safety_net=bool(args.safety_net),
                            population=args.fit_population)
    if opts.adopted:
        log.info(f"Z pliku wag ({args.weights}) przejęto: " + ", ".join(opts.adopted)
                 + f" → metoda {method}, temperatura {opts.temperature_mode}, reguła główna {main_rule}")

    out_dir = Path(args.out or (Path(config.RUNS_DIR) / f"final_report__{info['feature_set']}__"
                                                        f"{datetime.now():%Y%m%d_%H%M%S}"))

    # 2. Receptura na VAL
    try:
        weights, weights_mode = resolve_weights(args.weights, names, val.probas, val.y, method)
    except RecipeError as exc:
        log.error(f"ODMOWA: {exc}")
        return EXIT_REFUSED
    digest = recipe_hash([r.name for r in runs], weights, opts, dopts)
    prior = previous_reports([r.name for r in runs], digest, out_dir, [Path(config.RUNS_DIR), out_dir.parent])
    warn_test_once(prior)
    p_val_raw = blend(val.probas, weights, method)
    p_test_raw = blend(test.probas, weights, method)
    temperature: float | None = None
    per_model_t: dict[str, float] = {}
    crossfit = None
    if opts.temperature_mode == "fit":
        temperature = fit_temperature(p_val_raw, val.y)
        if opts.file_temperature is not None:
            # Plik wag nie jest źródłem T: T zawsze dopasowujemy na VAL tych runów i tych wag
            log.info(f"Temperatura dopasowana ponownie na VAL: T = {temperature:.4f} "
                     f"(w pliku wag: {opts.file_temperature:.4f}; używam dopasowanej)")
            if abs(temperature - opts.file_temperature) > TEMPERATURE_RTOL * max(1.0, opts.file_temperature):
                log.warning(f"T dopasowana na VAL ({temperature:.4f}) różni się od T w pliku wag "
                            f"({opts.file_temperature:.4f}) — plik pochodzi z innych runów / wag / metody?")
        per_model_t = {n: fit_temperature(val.probas[n], val.y) for n in names}
        p_cf, temps = crossfit_temperature(p_val_raw, val.y, n_folds=2, seed=args.seed)
        crossfit = {"temperatures": temps, "raw": calibration_summary(val.y, p_val_raw, args.n_bins),
                    "calibrated": calibration_summary(val.y, p_cf, args.n_bins)}
    p_val = apply_temperature(p_val_raw, temperature) if temperature is not None else p_val_raw
    p_test = apply_temperature(p_test_raw, temperature) if temperature is not None else p_test_raw

    def final(split: SplitData, name: str) -> np.ndarray:
        p = split.probas[name]
        return apply_temperature(p, per_model_t[name]) if name in per_model_t else p

    # Reguły decyzyjne: dopasowanie na VAL (połączenie i — ta sama obróbka — każdy model)
    t_rules = time.perf_counter()
    val_sets: dict[str, np.ndarray] = {BLEND: p_val}
    if temperature is not None:
        val_sets[BLEND_RAW] = p_val_raw
    for n in names:
        val_sets[n] = final(val, n)
    try:
        fitted: dict[str, FittedRules] = {k: rs.fit_rules(val.y, p, val.vc, dopts) for k, p in val_sets.items()}
    except ValueError as exc:
        log.error(f"ODMOWA: nie da się dopasować reguł decyzyjnych na VAL ({exc})")
        return EXIT_REFUSED
    fitted_blend = fitted[BLEND]
    main_spec = fitted_blend.specs[main_rule]
    if opts.file_rule_spec is not None and opts.file_rule_spec.get("rule") == main_rule and main_rule in rs.FITTED_RULES:
        log.info(f"Parametry reguły dopasowane ponownie na VAL: {main_spec} (w pliku wag: {opts.file_rule_spec})")
        if opts.file_rule_spec != main_spec:
            log.warning("Parametry reguły z pliku wag różnią się od dopasowanych na VAL — plik pochodzi z innych "
                        "runów / wag / celu T? Do backendu trafią parametry dopasowane teraz.")
    val_ll = {n: point_metrics(val.y, val.probas[n], ["log_loss"])["log_loss"] for n in names}
    best = min(val_ll, key=val_ll.get)
    val_proba = point_metrics(val.y, p_val, ["log_loss", "ece"], n_bins=args.n_bins)
    val_summary = {
        "log_loss": val_proba["log_loss"], "ece": val_proba["ece"],
        "rules": rs.val_rule_summary(val.y, p_val, fitted_blend, dopts, seed=args.seed),
        "single_log_loss": val_ll,
        "crossfit": crossfit,
    }
    log.info(f"Wagi ({weights_mode}): " + ", ".join(f"{n}={w:.3f}" for n, w in weights.items())
             + f"; metoda {method}; T = {temperature if temperature is None else round(temperature, 4)}; "
               f"najlepszy pojedynczy (log loss VAL): {best}; reguła główna {main_rule}: {main_spec}")
    timings["reguly_val_s"] = round(time.perf_counter() - t_rules, 2)
    timings["receptura_val_s"] = round(time.perf_counter() - t0, 2)

    # 3. TEST — raz
    t1 = time.perf_counter()
    candidates: dict[str, np.ndarray] = {BLEND: p_test}
    if temperature is not None:
        candidates[BLEND_RAW] = p_test_raw
    singles = sorted(names, key=lambda n: val_ll[n])
    for n in singles:
        candidates[n] = final(test, n)
    order = [BLEND, *singles]
    decisions = {name: rs.predict_rules(p, fitted[name]) for name, p in candidates.items()}
    decisions = {name: {r: d[r] for r in rules} for name, d in decisions.items()}
    blend_dec = decisions[BLEND]
    pred_main = blend_dec[main_rule]
    log.info(f"TEST: bootstrap {args.bootstrap}× dla {len(candidates)} wariantów × {len(rules)} reguł "
             f"(cała populacja)…")
    res_all = rs.bootstrap_decisions(test.y, candidates, decisions, DEFAULT_CI_METRICS, n_boot=args.bootstrap,
                                     alpha=args.alpha, seed=args.seed, n_bins=args.n_bins)
    res_vc = None
    if test.vc is not None and test.vc.any():
        log.info(f"TEST: bootstrap {args.bootstrap}× — podgrupa vc ({int(test.vc.sum()):,} wierszy)…")
        res_vc = rs.bootstrap_decisions(
            test.y[test.vc], {k: v[test.vc] for k, v in candidates.items()},
            {k: {r: p[test.vc] for r, p in d.items()} for k, d in decisions.items()},
            DEFAULT_CI_METRICS, n_boot=args.bootstrap, alpha=args.alpha, seed=args.seed, n_bins=args.n_bins)
    timings["bootstrap_test_s"] = round(time.perf_counter() - t1, 2)

    paired = {"cała populacja": res_all.paired(BLEND, best, PAIRED_METRICS, shown),
              "vc": res_vc.paired(BLEND, best, PAIRED_METRICS, shown) if res_vc is not None else None}
    paired_all = {
        group: {n: res.paired(BLEND, n, None, rules) for n in singles}
        for group, res in (("all", res_all), ("vc", res_vc)) if res is not None
    }
    calibration_paired = (res_all.paired(BLEND, BLEND_RAW, CALIBRATION_PAIRED_METRICS, rules[:1])
                          if temperature is not None else None)

    # Analizy runs/analyses (C1, C3, C9) i reguła wyższości C1 dla sekcji 3
    # tylko analizy z tych samych danych i tych samych runów co raport (inne → pominięte z notą)
    analyses = rs.load_analyses(info["feature_set"], args.tag or info.get("tag"),
                                fingerprint=info["data_fingerprint"], run_names={r.model: r.name for r in runs})
    seeds = analyses.get("seeds") if not (analyses.get("seeds") or {}).get("skipped") else None
    members = [best]                       # SD seedów porównywanego modelu (połączenie nie ma refitów)
    verdicts: dict[str, str] = {}
    sds: dict[str, float | None] = {}
    for group, table in paired.items():
        if table is None:
            continue
        population = "vc" if group == "vc" else "all"
        for rec in table.to_dict(orient="records"):
            rule = main_rule if rec["rule"] == "*" else rec["rule"]
            sd = rs.seed_sd(seeds, members, rec["metric"], rule, population=population)
            key = f"{group}|{rec['rule']}|{rec['metric']}"
            sds[key] = sd
            verdicts[key] = rs.superiority(rec["metric"], rec["delta"], rec["ci_low"], rec["ci_high"], sd,
                                           name_a=BLEND, name_b=best)
    superiority = {"verdicts": verdicts, "seed_sd": sds, "mid": rs.MINIMAL_IMPORTANT_DIFFERENCE,
                   "sd_factor": rs.SEED_SD_FACTOR, "models": members,
                   "source": (f"refity modelu {best} z innymi seedami, TEST" if seeds is not None
                              else "brak wyników seedów — `run_all.py --stage analyses`")}

    # Oddziały (połączenie)
    t2 = time.perf_counter()
    sites: dict[str, Any] = {}
    if "site" in test.meta.columns:
        site_values = test.meta["site"].astype("string").to_numpy()
        for site in sorted(pd.unique(site_values[~pd.isna(site_values)])):
            mask = site_values == site
            y_s = test.y[mask]
            res_s = rs.bootstrap_decisions(y_s, {BLEND: p_test[mask]}, {BLEND: {r: blend_dec[r][mask] for r in rules}},
                                           SITE_METRICS, n_boot=args.bootstrap, alpha=args.alpha, seed=args.seed)
            sites[str(site)] = {
                "n": int(mask.sum()), "n_esi1": int((y_s == 0).sum()),
                "ci": {rule: ci_to_dict(res_s.ci(BLEND, rule)) for rule in rules},
                "single_models": {n: {rule: rs.decision_point_metrics(y_s, candidates[n][mask],
                                                                      decisions[n][rule][mask], SITE_METRICS)
                                      for rule in rules} for n in singles},
            }
    timings["oddzialy_s"] = round(time.perf_counter() - t2, 2)

    # Kalibracja
    calibration = {
        "temperature": temperature,
        "val_raw": calibration_summary(val.y, p_val_raw, args.n_bins),
        "val_calibrated_in_sample": calibration_summary(val.y, p_val, args.n_bins),
        "val_crossfit": crossfit,
        "test_raw": calibration_summary(test.y, p_test_raw, args.n_bins, n_null=ECE_NULL_SIMS, seed=args.seed),
        "test_calibrated": calibration_summary(test.y, p_test, args.n_bins, n_null=ECE_NULL_SIMS, seed=args.seed),
        "test_vc_raw": calibration_summary(test.y[test.vc], p_test_raw[test.vc], args.n_bins,
                                           n_null=ECE_NULL_SIMS, seed=args.seed) if res_vc is not None else None,
        "test_vc_calibrated": calibration_summary(test.y[test.vc], p_test[test.vc], args.n_bins,
                                                  n_null=ECE_NULL_SIMS, seed=args.seed) if res_vc is not None else None,
        "per_model": {n: {"temperature": per_model_t.get(n),
                          "ece_raw": calibration_summary(test.y, test.probas[n], args.n_bins)["ece_top_label"],
                          "ece_final": calibration_summary(test.y, candidates[n], args.n_bins)["ece_top_label"]}
                      for n in singles},
    }

    # Macierze pomyłek
    from sklearn.metrics import confusion_matrix

    labels = list(range(N_CLASSES))
    confusion = {"all": {rule: confusion_matrix(test.y, blend_dec[rule], labels=labels).tolist() for rule in rules}}
    if res_vc is not None:
        confusion["vc"] = {rule: confusion_matrix(test.y[test.vc], blend_dec[rule][test.vc], labels=labels).tolist()
                           for rule in rules}

    # Przyjęcia (C4)
    t3 = time.perf_counter()
    esi_test = rs.nurse_esi(test.meta)
    admitted, source = rs.admission_labels(test.meta, "test", info["data_fingerprint"])
    admission: dict[str, Any] = {"auc": None, "source": source}
    has_admissions = rs.usable_admissions(admitted)
    if not has_admissions:
        admission["reason"] = source if admitted is None else "kolumna `admitted` bez obu wartości"
        log.info(f"Trafność względem przyjęć pominięta: {admission['reason']}")
    else:
        admission.update(rs.admission_section(admitted, esi_test, p_test, {r: blend_dec[r] for r in shown},
                                              args.bootstrap, args.alpha, args.seed, blend_label=BLEND))
    timings["przyjecia_s"] = round(time.perf_counter() - t3, 2)

    # Sekcje dodatkowe (każda pomijana z notą, gdy brak danych)
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = not args.no_plots

    def fig(name: str) -> Path | None:
        return fig_dir / name if plots else None

    tradeoff = run_section("kompromis_val", timings, lambda: rs.tradeoff_section(
        val.y, p_val, fitted_blend.mask, fitted_blend.population, dopts, fig("tradeoff_val.png"), args.seed))
    try:
        pred_frame = runs[0].predictions("test").set_index("row_id")
    except (OSError, KeyError, ValueError):
        pred_frame = None
    attrs, attrs_source = rs.load_attributes(test.meta, "test", info["data_fingerprint"], pred_frame)
    if attrs is None:
        log.info(f"Atrybuty pacjentów (płeć, wiek, przybycie, dolegliwości): {attrs_source}")
    groups = rs.subgroup_masks(test.meta, attrs)
    adm_or_none = admitted if has_admissions else None
    subgroups = run_section("podgrupy", timings, lambda: rs.subgroup_section(
        test.y, p_test, pred_main, groups, esi=esi_test, admitted=adm_or_none,
        pred_argmax=blend_dec.get("argmax"), n_boot=args.bootstrap, alpha=args.alpha, seed=args.seed))
    admission_subgroups = discordance = None
    if has_admissions:
        admission_subgroups = run_section("przyjecia_podgrupy", timings, lambda: rs.admission_subgroups(
            admitted, esi_test, p_test, groups, args.bootstrap, args.alpha, args.seed))
        discordance = rs.discordance_table(admitted, esi_test, pred_main)
    errors = run_section("bledy", timings, lambda: rs.error_analysis(test.y, pred_main, attrs, adm_or_none))
    dca_label = run_section("dca_etykieta", timings, lambda: rs.dca_label_section(
        test.y, p_test, test.vc, {r: blend_dec[r] for r in shown}, fig("dca_esi12_test.png")))
    if has_admissions:
        adm_val, _ = rs.admission_labels(val.meta, "val", info["data_fingerprint"])
        dca_admission = run_section("dca_przyjecia", timings, lambda: rs.dca_admission_section(
            adm_val, expected_level(p_val), rs.nurse_esi(val.meta), admitted, expected_level(p_test), esi_test,
            {r: blend_dec[r] for r in shown}, args.bootstrap, args.alpha, args.seed, fig("dca_admission_test.png")))
    else:
        dca_admission = {"skipped": admission.get("reason", "brak danych o przyjęciach")}
    cal_sets = [("cała populacja", np.ones(len(test.y), dtype=bool))]
    if test.vc is not None and test.vc.any():
        cal_sets.append(("podgrupa vc", test.vc))
    if "site" in test.meta.columns:
        site_values = test.meta["site"].astype("string").to_numpy()
        cal_sets += [(f"oddział {s}", site_values == s) for s in sorted(sites)]
    calibration_groups = run_section("kalibracja_grupy", timings,
                                     lambda: rs.calibration_groups(test.y, p_test, cal_sets, args.alpha))
    deferral = run_section("odsylanie", timings, lambda: rs.deferral_section(
        p_val, fitted_blend.mask, test.y, p_test, test.vc, pred_main, main_spec))
    conformal = run_section("conformal", timings, lambda: rs.conformal_section(
        val.y, p_val, test.y, p_test, test.vc))
    analysis_figures: dict[str, str] = {}
    lc = analyses.get("learning_curve") or {}
    lc_plot = (lc.get("results") or {}).get("plot")
    if plots and lc_plot and Path(lc_plot).exists():
        fig_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(lc_plot, fig_dir / "learning_curve.png")
        analysis_figures["learning_curve"] = "learning_curve.png"

    # 4. Zapis
    figures: list[str] = []
    if plots:
        t4 = time.perf_counter()
        figures = make_figures(fig_dir, test.y, test.vc, p_test_raw, p_test, temperature, shown, blend_dec, res_all,
                               res_vc, order, args.n_bins)
        for sec in (tradeoff, dca_label, dca_admission):
            if isinstance(sec, dict) and sec.get("figure"):
                figures.append(sec["figure"])
        figures += list(analysis_figures.values())
        timings["wykresy_s"] = round(time.perf_counter() - t4, 2)

    decision_record = {
        "main_rule": main_rule, "rules": rules, "target_undertriage": dopts.target, "guarantee": dopts.guarantee,
        "guarantee_alpha": config.DECISION_GUARANTEE_ALPHA, "safety_net": dopts.safety_net,
        "fit_population_requested": dopts.population, "fit_population": fitted_blend.population,
        "fit_population_note": fitted_blend.note, "main_spec": main_spec,
        "blend": fitted_blend.to_dict(), "per_model": {k: v.specs for k, v in fitted.items() if k != BLEND},
        "val": val_summary["rules"],
    }
    recipe = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_set": info["feature_set"],
        "data_version": info["data_version"],
        "data_fingerprint": info["data_fingerprint"],
        "split_strategy": info["split_strategy"],
        "samples": info["samples"],
        "runs": [{"name": r.name, "model": r.model, "tag": r.info.get("tag"), "path": str(r.path),
                  "train_sample": r.info.get("train_sample"), "eval_sample": r.info.get("eval_sample")}
                 for r in runs],
        "weights": weights,
        "weights_mode": weights_mode,
        "weights_arg": args.weights,
        "method": method,
        "temperature_mode": opts.temperature_mode,
        "options_from_weights_file": list(opts.adopted),
        "weights_file_temperature": opts.file_temperature,
        "weights_file_rule": opts.file_rule_spec,
        "temperature": temperature,
        "per_model_temperature": per_model_t,
        "rules": rules,
        "main_rule": main_rule,
        "backend_rule": main_rule,
        "decision": decision_record,
        "best_single_model": best,
        "best_single_by": "najniższy log loss na VAL",
        "bootstrap": {"n_boot": args.bootstrap, "alpha": args.alpha, "seed": args.seed, "stratified": True,
                      "method": "percentile", "proportions": "clopper-pearson (k/n)",
                      "paired": "percentile (statystyki pozycyjne), p = 2·min(P(Δ≤0), P(Δ≥0))"},
        "n_bins": args.n_bins,
        "recipe_hash": digest,
        "previous_reports": [e["path"] for e in prior],
        "previous_reports_detail": prior,
        "git_commit": _git_commit(),
        "argv": sys.argv[1:] if argv is None else list(argv),
    }
    counts = {"all": np.bincount(test.y, minlength=N_CLASSES).tolist()}
    if test.vc is not None:
        counts["vc"] = np.bincount(test.y[test.vc], minlength=N_CLASSES).tolist()
    data = {"n": int(len(test.y)), "n_vc": int(test.vc.sum()) if test.vc is not None else None,
            "esi_counts": counts}

    ctx = {"rules": rules, "shown_rules": shown, "main_rule": main_rule, "order": order, "res_all": res_all,
           "res_vc": res_vc, "recipe": recipe, "data": data, "val_summary": val_summary, "paired": paired,
           "calibration": calibration, "calibration_paired": calibration_paired, "confusion": confusion,
           "sites": sites, "admission": admission, "figures": figures, "dopts": dopts, "fitted_blend": fitted_blend,
           "decision": {"val": val_summary["rules"]}, "superiority": superiority, "tradeoff": tradeoff,
           "subgroups": subgroups, "attributes_source": attrs_source, "admission_subgroups": admission_subgroups,
           "discordance": discordance, "errors": errors, "dca_label": dca_label, "dca_admission": dca_admission,
           "calibration_groups": calibration_groups, "deferral": deferral, "conformal": conformal,
           "analyses": analyses, "analysis_figures": analysis_figures}
    tables = write_tables(out_dir / "tables", ctx)
    write_text_atomic(out_dir / "report.md", build_report(ctx, Fmt(args.decimal_comma, args.alpha)))
    write_json_atomic(out_dir / "recipe.json", _jsonable(recipe))
    backend = {
        "weights": {n: round(w, 6) for n, w in weights.items() if w > 0},
        "method": method,
        "temperature": None if temperature is None else round(temperature, 6),
        **main_spec,
    }
    write_json_atomic(out_dir / "backend_ensemble_weights.json", _jsonable(backend))
    timings["razem_s"] = round(time.perf_counter() - t0, 2)
    analyses_summary = {k: ({"skipped": v["skipped"]} if v.get("skipped") else {"path": v.get("path")})
                        for k, v in analyses.items()}
    metrics = {
        "recipe": recipe,
        "data": data,
        "val": val_summary,
        "test": {
            "ci": {group: {name: {rule: ci_to_dict(res.ci(name, rule)) for rule in rules} for name in res.names}
                   for group, res in (("all", res_all), ("vc", res_vc)) if res is not None},
            "paired_blend_vs_best": {g: t for g, t in paired.items() if t is not None},
            "paired_all": paired_all,
            "superiority": superiority,
            "calibration_paired": calibration_paired,
            "calibration": calibration,
            "calibration_groups": calibration_groups,
            "confusion": confusion,
            "sites": sites,
            "admission": admission,
            "admission_subgroups": admission_subgroups,
            "admission_discordance": discordance,
            "subgroups": {k: v for k, v in subgroups.items() if k != "rows"} | {
                "rows": [{k: v for k, v in r.items() if k != "mask"} for r in subgroups.get("rows", [])]},
            "errors": errors,
            "dca_label": dca_label,
            "dca_admission": dca_admission,
            "deferral": deferral,
            "conformal": conformal,
        },
        "val_tradeoff": tradeoff,
        "analyses": analyses_summary,
        "files": {"figures": figures, "tables": tables},
        "timings": timings,
    }
    write_json_atomic(out_dir / "metrics.json", _jsonable(metrics))
    register_report({"created_at": recipe["created_at"], "out_dir": str(out_dir.absolute()),
                     "runs": sorted(r.name for r in runs), "feature_set": info["feature_set"],
                     "recipe_hash": digest, "weights": weights, "method": method,
                     "temperature": temperature, "rules": rules, "backend_rule": main_rule,
                     "main_rule": main_rule, "main_spec": main_spec})

    # Podsumowanie na konsoli
    fmt = Fmt()
    print(f"\n=== Raport końcowy (TEST, n={len(test.y):,}) → {out_dir} ===")
    for rule in rules:
        ci = _ci_lookup(res_all, BLEND, rule)
        star = " *" if rule == main_rule else ""
        print(f"{BLEND} [{RULE_LABELS[rule]}{star}]: QWK {fmt.ci('quadratic_weighted_kappa', ci['quadratic_weighted_kappa'])}"
              f" | niedotriażowanie {fmt.ci('undertriage_rate', ci['undertriage_rate'])}"
              f" | log loss {fmt.ci('log_loss', ci['log_loss'])}")
    table = paired["cała populacja"]
    for rec in table.to_dict(orient="records"):
        rule = "" if rec["rule"] == "*" else f" [{RULE_LABELS[rec['rule']]}]"
        print(f"Δ {SHORT_LABELS[rec['metric']]}{rule} ({BLEND} − {best}): "
              f"{fmt.delta(rec['metric'], rec['delta'], rec['ci_low'], rec['ci_high'])}, p {fmt.p(rec['p_two_sided'])}")
    skipped = [name for name, sec in (("kompromis", tradeoff), ("podgrupy", subgroups), ("błędy", errors),
                                      ("DCA etykieta", dca_label), ("DCA przyjęcia", dca_admission),
                                      ("kalibracja grup", calibration_groups), ("odsyłanie", deferral),
                                      ("conformal", conformal))
               if isinstance(sec, dict) and sec.get("skipped")]
    skipped += [f"analiza {k}" for k, v in analyses.items() if v.get("skipped")]
    if skipped:
        print("Sekcje pominięte (powód w report.md): " + ", ".join(skipped))
    print("Czasy [s]: " + ", ".join(f"{k} {v}" for k, v in timings.items()))
    warn_test_once(prior)
    log.info(f"Raport: {out_dir / 'report.md'} (zapisany w rejestrze {_registry_path()})")
    log.info(f"Wagi i reguła dla backendu: {out_dir / 'backend_ensemble_weights.json'} "
             f"(skopiuj jako backend/models/ensemble_weights.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
