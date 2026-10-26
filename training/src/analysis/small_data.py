"""
Eksperyment A (TFM) — krzywa uczenia przy MAŁEJ liczbie lokalnych etykiet: czy wstępnie wytrenowane
tabularne modele fundamentalne (TabPFN-3.5, TabICLv2) „z pudełka” wygrywają ze STROJONYM LightGBM,
gdy szpital ma niewiele oznaczonych wizyt — i do jakiej liczby wizyt n*.

Scenariusze
    pooled      — podpróby n wierszy TRAIN wszystkich oddziałów (warstwy ESI × oddział); ocena na stałej
                  warstwowej próbie `eval_rows` wierszy VAL (0 = cały VAL),
    site_local  — podpróby tylko z TRAIN oddziału `local_site` (domyślnie config.HOLDOUT_SITE; warstwy ESI);
                  ocena na WSZYSTKICH wierszach VAL tego oddziału (`local_eval_rows` > 0 — warstwowa próba).
Rozmiary `sizes` (domyślnie 500 … 50 000) × powtórzenia `repeats`; każda podpróba (scenariusz, n, r)
losowana RAZ (`samples/<scenariusz>_n<n>_r<r>.npy` — row_id) i wspólna dla wszystkich modeli (układ
sparowany); w każdej ≥ 2 wiersze każdej klasy obecnej w źródle (zamiana z najliczniejszą klasą).

Modele (jednostka = scenariusz × n × r × model; jednostka = granulat wznawiania)
    lightgbm_tuned      — Optuna (w pamięci, na jednostkę) wyłącznie na podpróbie: `lgbm_trials` prób,
                          wewnętrzna warstwowa K-krotna CV, K = max(2, min(5, liczność najrzadszej klasy)),
                          przestrzeń dla małych n (num_leaves 4–64, min_child_samples 5–100, learning_rate,
                          feature/bagging fraction, lambda L1/L2), cel log loss, liczba drzew = średnia
                          najlepsza iteracja z wczesnego zatrzymania w foldach; model finalny na całej podpróbie,
    lightgbm_default    — domyślne ustawienia BIBLIOTEKI LightGBM (100 drzew, bez strojenia),
    tabpfn / tabicl     — „z pudełka” przez wrappery (kontekst = cała podpróba, 1 worek, domyślne ustawienia
                          modelu biblioteki; tylko pokrętła pamięci, które nie zmieniają predykcji),
    <tfm>+foreign       — (site_local) kontekst = n lokalnych + `foreign_rows` wierszy innych oddziałów
                          (warstwy ESI × oddział) — test transferu negatywnego,
    lightgbm_transported — (site_local, jedna jednostka odniesienia) ≤ `transport_rows` wierszy INNYCH
                          oddziałów, parametry najnowszego strojonego runa lightgbm z tagiem (brak → domyślne
                          repo, odnotowane), ocena na lokalnych wierszach VAL: surowo i z korektą znanego
                          lokalnego rozkładu ESI (src/analysis/priors.py, Saerens i in. 2002).

Statystyka (każdy model vs lightgbm_tuned w danym scenariuszu i n; metryka pierwszorzędowa: log loss)
    - sparowane Δ w każdym powtórzeniu → średnia, SD, 95% przedział t (df = R − 1), p testu t,
    - dwupoziomowy bootstrap (`n_boot`): losowanie powtórzeń ze zwracaniem, potem wierszy oceny
      warstwowo względem ESI (jedno losowanie wierszy na replikację, wspólne dla powtórzeń — ten sam zbiór
      oceny) → percentylowy 95% CI i p (te same ogony),
    - p do werdyktu = max(p bootstrapu, p testu t) — co najmniej tak ostrożne jak sparowany test t po
      powtórzeniach: przy R = 2–5 percentylowy poziom 1 bootstrapu jest zbyt wąski (zgodne znaki Δ we
      wszystkich powtórzeniach → p bootstrapu = 1/B niezależnie od rozrzutu), a sam test t pomija wspólny
      szum wierszy oceny; R < 2 → werdykt 'na' (brak testu),
    - korekta Holma między rozmiarami n dla metryki pierwszorzędowej; „wygrana” tylko gdy p_Holm < 0,05
      (przedział t I przedział bootstrapu bez zera) I |Δ| > MID (report_sections.MINIMAL_IMPORTANT_DIFFERENCE),
    - krzywe a + b·n^(−c) (learning_curve.fit_power_law) i punkt przecięcia n* z bootstrapem po powtórzeniach
      (przedział n* „orientacyjny”, gdy któryś rozmiar ma < MIN_REPEATS_NSTAR_CI powtórzeń).

Wyniki: runs/analyses/small_data__<cechy>__<tag>/ — plan.json, units/*.json (UnitStore; klucz source_run),
preds/*.npy (prawdopodobieństwa wierszy oceny), samples/*.npy, results.json, units.csv, figures/*.png,
report.md (po polsku: tabele, decyzja, akapit metod do pracy, zastrzeżenia), summary.md.

Brak torch / biblioteki / wag → dany TFM pominięty z powodem (results.json `skipped_models`, report.md),
części LightGBM liczone normalnie. CPU: jednostki TFM z kontekstem > config.TFM_CPU_MAX_CONTEXT pomijane
(z powodem) — na CPU predykcja trwałaby godziny.

Szacunek czasu (`plan`) czyta runs/analyses/tfm_setup/benchmark.json, jeśli istnieje (13_tfm_setup.py):
lista rekordów {"model", "n_ctx", "rows_per_s", "fit_s"?} pod kluczem "results" (albo {model: [...]},
albo {"models": {model: {"results": [...]}}}); bez pomiaru — czasy z planu pamięci (GPU, orientacyjnie).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.common import (
    AnalysisArgumentError,
    UnitStore,
    analyses_root,
    clean_json,
    default_tag,
    evaluate,
    execute_units,
    format_duration,
    now_iso,
    read_json,
    vc_mask,
    write_json,
)
from src.utils import config
from src.utils.fileio import replace_with_retry, write_text_atomic
from src.utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────
# Stałe (ustalone z góry — zmiana = inny eksperyment)
# ─────────────────────────────────────────
ANALYSIS = "small_data"
# Wersja semantyki jednostek — PODBIJ przy zmianie sposobu liczenia jednostki (stare zostaną przeliczone)
SMALL_DATA_VERSION = 1
SCENARIOS: tuple[str, ...] = ("pooled", "site_local")
DEFAULT_SIZES: tuple[int, ...] = (500, 1_000, 2_000, 5_000, 10_000, 50_000)
DEFAULT_REPEATS: tuple[int, ...] = (10, 10, 10, 5, 5, 3)
DEFAULT_EVAL_ROWS = 20_000
DEFAULT_FOREIGN_ROWS = 10_000
DEFAULT_TRANSPORT_ROWS = 100_000
DEFAULT_LGBM_TRIALS = 40
DEFAULT_BOOTSTRAP = 1_000
MIN_SIZE = 20                          # ≥ 2 wiersze × 5 klas + zapas
MIN_PER_CLASS = 2
ALPHA = 0.05
PRIMARY_METRIC = "log_loss"
BASELINE = "lightgbm_tuned"
LGBM_ARMS: tuple[str, ...] = ("lightgbm_tuned", "lightgbm_default")
TRANSPORTED = "lightgbm_transported"
FOREIGN_SUFFIX = "+foreign"
RECALL_MIN_N = 5_000                   # czułość ESI 1 raportowana tylko od tylu wierszy uczących
SMALL_COUNT_ESI1 = 100                 # mniej wizyt ESI 1 w zbiorze oceny → ostrzeżenie (Clopper–Pearson)
RULES: tuple[str, ...] = ("argmax", "cost")
CROSSOVER_FACTOR = 10.0                # n* szukane do 10 × największego n (dalej — za daleko poza danymi)
MIN_REPEATS_TEST = 2                   # werdykt (test) wymaga ≥ 2 powtórzeń; R = 1 → 'na'
MIN_REPEATS_NSTAR_CI = 5               # CI n* „pełny” tylko, gdy każdy rozmiar ma ≥ 5 powtórzeń; inaczej orientacyjny

# Strojenie LightGBM na małych n (wersja przestrzeni w konfiguracji jednostki)
LGBM_SPACE_VERSION = 1
LGBM_MAX_ROUNDS = 5_000                # limit drzew w foldzie (wczesne zatrzymanie wybiera liczbę)
LGBM_ES_ROUNDS = 50
LGBM_MAX_FOLDS = 5
LGBM_SPACE: dict[str, tuple[float, float, bool]] = {   # nazwa → (min, max, skala log)
    "num_leaves": (4, 64, True),
    "min_child_samples": (5, 100, True),
    "learning_rate": (0.01, 0.3, True),
    "colsample_bytree": (0.3, 1.0, False),   # feature_fraction
    "subsample": (0.5, 1.0, False),          # bagging_fraction (subsample_freq = 1)
    "reg_alpha": (1e-8, 10.0, True),         # lambda_l1
    "reg_lambda": (1e-8, 10.0, True),        # lambda_l2
}
LGBM_INT_PARAMS = frozenset({"num_leaves", "min_child_samples"})
# Domyślne ustawienia BIBLIOTEKI LightGBM (nie domyślne repo — te mają 20 000 drzew pod early stopping)
LGBM_LIBRARY_DEFAULTS: dict[str, Any] = {
    "objective": "multiclass", "num_class": config.N_CLASSES, "n_estimators": 100, "learning_rate": 0.1,
    "num_leaves": 31, "max_depth": -1, "min_child_samples": 20, "subsample": 1.0, "subsample_freq": 0,
    "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 0.0, "verbosity": -1,
}
# TFM „z pudełka”: domyślne ustawienia MODELU biblioteki (TabPFN n_estimators "auto", TabICL 8 estymatorów,
# temperatura softmax domyślna); ustawione tylko pokrętła pamięci/czasu, które nie zmieniają predykcji:
#   TabPFN fit_mode "fit_preprocessors" (domyślny biblioteki; jedno wywołanie na jednostkę — bufor KV nic
#   nie daje), predict_batch 32 768 ≥ wierszy oceny (jedno wywołanie);
#   TabICL kv_cache False (domyślny biblioteki), batch_size "auto" (1 przy n ≥ 10 000, inaczej 4 — RAM),
#   predict_batch "auto" (budżet RAM hosta).
TFM_OUT_OF_BOX: dict[str, dict[str, Any]] = {
    "tabpfn": {"n_estimators": "auto", "fit_mode": "fit_preprocessors", "predict_batch": 32_768},
    "tabicl": {"n_estimators": 8, "kv_cache": False, "batch_size": "auto", "predict_batch": "auto"},
}
TFM_DISPLAY = {"tabpfn": "TabPFN-3.5", "tabicl": "TabICLv2"}
# Metryki porównań: klucz raportu → (metryka src.evaluation.statistics, reguła; '*' = z prawdopodobieństw)
COMPARE_METRICS: dict[str, tuple[str, str]] = {
    "log_loss": ("log_loss", "*"),
    "brier": ("brier", "*"),
    "ece": ("ece", "*"),
    "qwk_argmax": ("quadratic_weighted_kappa", "argmax"),
    "qwk_cost": ("quadratic_weighted_kappa", "cost"),
    "undertriage_argmax": ("undertriage_rate", "argmax"),
    "undertriage_cost": ("undertriage_rate", "cost"),
}
HIGHER_IS_BETTER: dict[str, bool] = {"log_loss": False, "brier": False, "ece": False, "qwk_argmax": True,
                                     "qwk_cost": True, "undertriage_argmax": False, "undertriage_cost": False,
                                     "auc_macro": True, "ece_esi1": False, "ece_esi5": False}
# Kolumny tabel (klucz metryki jednostki → nagłówek)
TABLE_METRICS: dict[str, str] = {
    "log_loss": "log loss", "brier": "Brier", "ece": "ECE", "ece_esi1": "ECE ESI1", "ece_esi5": "ECE ESI5",
    "auc_macro": "AUC", "qwk_argmax": "QWK argmax", "qwk_cost": "QWK koszt",
    "undertriage_argmax": "undertriage argmax", "undertriage_cost": "undertriage koszt",
    "recall_esi1_cost": "czułość ESI1 koszt",
}
CURVE_METRICS: tuple[tuple[str, str], ...] = (("log_loss", "log loss (↓)"), ("qwk_argmax", "QWK argmax (↑)"),
                                              ("undertriage_cost", "niedotriażowanie, reguła koszt (↓)"))
# Szacunki czasu bez pomiaru: plan pamięci §7 (GPU RTX 4080 Super, 20 000 wierszy oceny; czas FLOP × 3)
_MEMPLAN_UNIT_S: dict[str, dict[int, float]] = {
    "tabpfn": {500: 12, 1_000: 12, 2_000: 15, 5_000: 24, 10_000: 39, 50_000: 270},
    "tabicl": {500: 6, 1_000: 6, 2_000: 6, 5_000: 9, 10_000: 15, 50_000: 75},
}
# Estymatory jednostki „z pudełka” (TabPFN "auto" = 8 przy 271 cechach) — przeliczanie benchmarku 13_tfm_setup
_UNIT_ESTIMATORS: dict[str, float] = {"tabpfn": 8.0, "tabicl": 8.0}
# Kolory modeli — stała kolejność (kolor należy do modelu, nie do pozycji); paleta zweryfikowana (CVD)
ARM_COLORS: dict[str, str] = {
    "lightgbm_tuned": "#2a78d6", "tabpfn": "#eb6834", "tabicl": "#1baf7a", "lightgbm_default": "#eda100",
    "tabpfn+foreign": "#e87ba4", "tabicl+foreign": "#008300", "lightgbm_transported": "#6250d6",
}
ARM_STYLE: dict[str, tuple[str, str]] = {    # (styl linii, znacznik) — kodowanie niezależne od koloru
    "lightgbm_tuned": ("-", "o"), "tabpfn": ("-", "s"), "tabicl": ("-", "D"), "lightgbm_default": (":", "v"),
    "tabpfn+foreign": ("--", "s"), "tabicl+foreign": ("--", "D"),
}


# ─────────────────────────────────────────
# Konfiguracja
# ─────────────────────────────────────────
@dataclass
class SmallDataConfig:
    """Parametry eksperymentu A (CLI: scripts/14_small_data_curve.py)."""

    feature_set: str = "triage"
    tag: str | None = None
    models: list[str] = field(default_factory=lambda: list(config.TFM_MODELS))   # porównywane TFM
    scenarios: list[str] = field(default_factory=lambda: list(SCENARIOS))
    sizes: list[int] = field(default_factory=lambda: list(DEFAULT_SIZES))
    repeats: list[int] = field(default_factory=lambda: list(DEFAULT_REPEATS))
    eval_rows: int = DEFAULT_EVAL_ROWS              # pooled; 0 = cały VAL
    local_eval_rows: int = 0                        # site_local; 0 = wszystkie wiersze VAL oddziału
    local_site: str = config.HOLDOUT_SITE
    foreign_rows: int = DEFAULT_FOREIGN_ROWS        # 0 = bez jednostek +foreign
    transport_rows: int = DEFAULT_TRANSPORT_ROWS    # 0 = bez jednostki lightgbm_transported
    lgbm_trials: int = DEFAULT_LGBM_TRIALS
    n_boot: int = DEFAULT_BOOTSTRAP
    device: str = "auto"                            # auto | cpu | gpu (TFM)
    seed: int = config.RANDOM_SEED
    force: bool = False

    def resolved_tag(self) -> str:
        return self.tag or default_tag()


def parse_int_list(spec: str | Sequence[int], name: str) -> list[int]:
    """'500,1000' albo lista → liczby całkowite (separatory tysięcy '_' dozwolone)."""
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    try:
        values = [int(str(v).strip().replace("_", "")) for v in items if str(v).strip()]
    except ValueError as exc:
        raise AnalysisArgumentError(f"{name}: oczekiwano liczb całkowitych po przecinku, podano {spec!r}") from exc
    if not values:
        raise AnalysisArgumentError(f"{name}: pusta lista")
    return values


def parse_sizes_repeats(sizes: str | Sequence[int], repeats: str | Sequence[int]) -> tuple[list[int], list[int]]:
    """Rozmiary (rosnąco, unikalne, ≥ MIN_SIZE) i powtórzenia (jedna wartość = dla wszystkich rozmiarów)."""
    raw_sizes = parse_int_list(sizes, "--sizes")
    raw_repeats = parse_int_list(repeats, "--repeats")
    if len(set(raw_sizes)) != len(raw_sizes):
        raise AnalysisArgumentError(f"--sizes: powtórzone rozmiary {raw_sizes}")
    if any(n < MIN_SIZE for n in raw_sizes):
        raise AnalysisArgumentError(f"--sizes: każdy rozmiar musi być ≥ {MIN_SIZE}, podano {raw_sizes}")
    if len(raw_repeats) == 1:
        raw_repeats = raw_repeats * len(raw_sizes)
    if len(raw_repeats) != len(raw_sizes):
        raise AnalysisArgumentError(f"--repeats: {len(raw_repeats)} wartości, a rozmiarów {len(raw_sizes)} "
                                    f"(podaj jedną wartość albo po jednej na rozmiar)")
    if any(r < 1 for r in raw_repeats):
        raise AnalysisArgumentError(f"--repeats: każda liczba powtórzeń musi być ≥ 1, podano {raw_repeats}")
    pairs = sorted(zip(raw_sizes, raw_repeats, strict=True))
    return [n for n, _ in pairs], [r for _, r in pairs]


def parse_scenarios(spec: str | Sequence[str]) -> list[str]:
    items = [s.strip().lower() for s in (spec.split(",") if isinstance(spec, str) else spec) if str(s).strip()]
    unknown = [s for s in items if s not in SCENARIOS]
    if unknown or not items:
        raise AnalysisArgumentError(f"--scenarios: dostępne {', '.join(SCENARIOS)}, podano {spec!r}")
    return [s for s in SCENARIOS if s in items]


def parse_tfm_models(spec: str | Sequence[str] | None) -> list[str]:
    """
    TFM do porównania: nazwy / aliasy (pfn, icl) / 'tfm'; 'none' albo pusty tekst → tylko LightGBM.
    'lightgbm' (alias 'lgbm') jest akceptowane i pomijane — ramiona LightGBM liczone są zawsze (orkiestrator
    przekazuje listę modeli kroku razem z lightgbm).
    """
    from src.models import resolve_model_name

    if spec is None:
        return list(config.TFM_MODELS)
    items = [s.strip() for s in (spec.split(",") if isinstance(spec, str) else spec) if str(s).strip()]
    out: list[str] = []
    for item in items:
        if item.lower() == "none":
            continue
        if item.lower() == "tfm":
            names: tuple[str, ...] = tuple(config.TFM_MODELS)
        else:
            try:
                names = (resolve_model_name(item),)
            except ValueError as exc:
                raise AnalysisArgumentError(str(exc)) from exc
            if names == ("lightgbm",):
                continue
        bad = [n for n in names if n not in config.TFM_MODELS]
        if bad:
            raise AnalysisArgumentError(f"--models przyjmuje tylko modele TFM ({', '.join(config.TFM_MODELS)}, 'tfm', "
                                        f"'none'; 'lightgbm' — bez znaczenia, LightGBM liczony zawsze), podano {item!r}")
        out.extend(n for n in names if n not in out)
    return [m for m in config.TFM_MODELS if m in out]


def validate_config(cfg: SmallDataConfig) -> SmallDataConfig:
    """Normalizacja i kontrola argumentów (AnalysisArgumentError → kod wyjścia 2)."""
    from src.data.dataset import resolve_feature_set

    try:
        cfg.feature_set = resolve_feature_set(cfg.feature_set)
    except ValueError as exc:
        raise AnalysisArgumentError(str(exc)) from exc
    cfg.sizes, cfg.repeats = parse_sizes_repeats(cfg.sizes, cfg.repeats)
    cfg.scenarios = parse_scenarios(cfg.scenarios)
    cfg.models = parse_tfm_models(cfg.models)
    cfg.local_site = str(cfg.local_site).strip().upper()
    cfg.device = str(cfg.device).strip().lower()
    if cfg.device not in ("auto", "cpu", "gpu"):
        raise AnalysisArgumentError(f"--device: auto | cpu | gpu, podano {cfg.device!r}")
    for name in ("eval_rows", "local_eval_rows", "foreign_rows", "transport_rows", "n_boot"):
        if int(getattr(cfg, name)) < 0:
            raise AnalysisArgumentError(f"{name} musi być ≥ 0, podano {getattr(cfg, name)}")
    if int(cfg.lgbm_trials) < 1:
        raise AnalysisArgumentError(f"--lgbm-trials musi być ≥ 1, podano {cfg.lgbm_trials}")
    return cfg


def analysis_path(feature_set: str, tag: str, runs_dir: Path | str | None = None) -> Path:
    """runs/analyses/small_data__<cechy>__<tag>/ (common.analysis_dir zna tylko 3 starsze analizy)."""
    return analyses_root(runs_dir) / f"{ANALYSIS}__{feature_set}__{tag}"


# ─────────────────────────────────────────
# Losowanie podprób
# ─────────────────────────────────────────
def derive_seed(base: int, *parts: Any) -> int:
    """Deterministyczny seed z bazowego i etykiet (scenariusz, n, r, …) — niezależny od kolejności liczenia."""
    text = "|".join([str(int(base)), *(str(p) for p in parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little") % (2**31 - 1)


def _allocate(counts: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """m wierszy proporcjonalnie do liczności warstw (największe reszty, remisy losowo; ≤ liczności)."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if m >= total:
        return counts.copy()
    exact = counts * (m / total)
    alloc = np.floor(exact).astype(np.int64)
    rest = int(m - alloc.sum())
    if rest > 0:
        frac = exact - alloc + rng.random(len(counts)) * 1e-9
        frac[alloc >= counts] = -1.0
        alloc[np.argsort(-frac, kind="stable")[:rest]] += 1
    return alloc


def stratified_sample(pool: np.ndarray, labels: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """
    m pozycji z `pool` (pozycje bezwzględne) proporcjonalnie do warstw `labels` (etykieta każdej pozycji
    puli, ta sama kolejność); m ≥ len(pool) → cała pula. Wynik posortowany.
    """
    pool = np.asarray(pool, dtype=np.int64)
    labels = np.asarray(labels)
    if len(labels) != len(pool):
        raise ValueError(f"stratified_sample: {len(labels)} etykiet ≠ {len(pool)} pozycji")
    if m <= 0 or m >= len(pool):
        return np.sort(pool) if m > 0 else np.empty(0, dtype=np.int64)
    uniq, inv = np.unique(labels.astype(str), return_inverse=True)
    alloc = _allocate(np.bincount(inv, minlength=len(uniq)), int(m), rng)
    parts = [rng.choice(pool[inv == s], int(alloc[s]), replace=False) for s in range(len(uniq)) if alloc[s] > 0]
    return np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)


def ensure_min_per_class(chosen: np.ndarray, pool: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                         min_per_class: int = MIN_PER_CLASS) -> tuple[np.ndarray, list[dict[str, int]]]:
    """
    Każda klasa obecna w puli: ≥ min(min_per_class, liczność w puli) wierszy w wyborze. Brakujące wiersze
    dobierane z puli W MIEJSCE losowych wierszy najliczniejszej klasy wyboru (rozmiar bez zmian, o ile ta
    klasa ma zapas). `y` — etykiety indeksowane pozycją bezwzględną. Zwraca (pozycje, dobrane klasy).
    """
    chosen = np.sort(np.asarray(chosen, dtype=np.int64))
    pool = np.asarray(pool, dtype=np.int64)
    y = np.asarray(y).astype(np.int64)
    pool_counts = np.bincount(y[pool], minlength=config.N_CLASSES)
    events: list[dict[str, int]] = []
    for k in np.flatnonzero(pool_counts):
        need = min(int(min_per_class), int(pool_counts[k]))
        have = int((y[chosen] == k).sum())
        if have >= need:
            continue
        missing = need - have
        candidates = np.setdiff1d(pool[y[pool] == k], chosen, assume_unique=True)
        add = rng.choice(candidates, missing, replace=False)
        chosen_y = y[chosen]
        counts = np.bincount(chosen_y, minlength=config.N_CLASSES)
        counts[k] = -1
        major = int(np.argmax(counts))
        major_need = min(int(min_per_class), int(pool_counts[major]))
        if counts[major] - missing >= major_need:
            drop = rng.choice(chosen[chosen_y == major], missing, replace=False)
            chosen = np.setdiff1d(chosen, drop, assume_unique=True)
        chosen = np.sort(np.concatenate([chosen, add]))
        events.append({"class": int(k), "esi": int(k) + 1, "added": int(missing)})
    return chosen, events


def draw_sample(pool: np.ndarray, labels: np.ndarray, y: np.ndarray, m: int, seed: int,
                min_per_class: int = MIN_PER_CLASS) -> np.ndarray:
    """Warstwowa próba m pozycji z puli + gwarancja ≥ min_per_class wierszy każdej klasy (deterministycznie)."""
    rng = np.random.default_rng(int(seed))
    chosen = stratified_sample(pool, labels, m, rng)
    if min_per_class > 0 and 0 < len(chosen) < len(pool):
        chosen, _ = ensure_min_per_class(chosen, pool, y, rng, min_per_class)
    return chosen


def ids_hash(ids: np.ndarray) -> str:
    """Skrót zbioru row_id (kolejność bez znaczenia) — część oczekiwań jednostki."""
    values = np.sort(np.asarray(ids).astype(str))
    return hashlib.sha1(",".join(values.tolist()).encode("utf-8")).hexdigest()[:16]


def _save_npy(path: Path, array: np.ndarray) -> Path:
    """np.save przez plik tymczasowy + replace_with_retry (czytelnik nie widzi połowy pliku)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            np.save(fh, np.asarray(array), allow_pickle=False)
        replace_with_retry(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    return path


class SampleBank:
    """
    Próby losowane RAZ: samples/<nazwa>.npy (row_id) + samples/manifest.json (seed, odcisk danych, oddział).
    Istniejący plik zgodny z manifestem i danymi (wszystkie row_id obecne, oczekiwana liczność) jest
    używany ponownie; inaczej próba losowana od nowa i zapisywana.
    """

    def __init__(self, root: Path, spec: Mapping[str, Any], save: bool = True) -> None:
        self.dir = Path(root) / "samples"
        self.spec = clean_json(dict(spec))
        self.save = save
        manifest = read_json(self.dir / "manifest.json") or {}
        self.stale = manifest.get("spec") != self.spec
        self.files: dict[str, dict[str, Any]] = {} if self.stale else dict(manifest.get("files") or {})
        self.reused = 0
        self.drawn = 0

    def get(self, name: str, row_ids: np.ndarray, draw: Callable[[], np.ndarray],
            expected: int | None = None) -> np.ndarray:
        """Pozycje próby `name` w ramce o kolumnie `row_ids` (posortowane)."""
        path = self.dir / f"{name}.npy"
        if not self.stale and path.exists():
            try:
                ids = np.load(path, allow_pickle=False)
                import pandas as pd

                pos = pd.Index(np.asarray(row_ids)).get_indexer(ids)
                if (pos >= 0).all() and (expected is None or len(pos) == int(expected)):
                    self.reused += 1
                    return np.sort(pos.astype(np.int64))
            except (OSError, ValueError):
                pass
        pos = np.sort(np.asarray(draw(), dtype=np.int64))
        ids = np.asarray(row_ids)[pos]
        if self.save:
            _save_npy(path, ids)
        self.files[name] = {"n": int(len(ids)), "hash": ids_hash(ids)}
        self.drawn += 1
        return pos

    def write_manifest(self) -> None:
        if self.save:
            write_json(self.dir / "manifest.json", {"spec": self.spec, "files": self.files, "written_at": now_iso()})


# ─────────────────────────────────────────
# Dane
# ─────────────────────────────────────────
class SplitData:
    """TRAIN i VAL (cechy zestawu + meta) wczytane raz: X (float32), y, oddział, row_id, vc, przyjęcie."""

    def __init__(self, feature_set: str) -> None:
        from src.data.dataset import feature_names, load_splits

        self.feature_set = feature_set
        self.features = feature_names(feature_set)
        splits = load_splits(("train", "val"), feature_set=feature_set)
        self.X: dict[str, Any] = {}
        self.y: dict[str, np.ndarray] = {}
        self.site: dict[str, np.ndarray] = {}
        self.row_id: dict[str, np.ndarray] = {}
        self.vc: dict[str, np.ndarray | None] = {}
        self.admitted: dict[str, np.ndarray | None] = {}
        for name in ("train", "val"):
            df = splits.pop(name).reset_index(drop=True)
            self.X[name] = df[self.features].astype(np.float32)
            self.y[name] = df[config.TARGET_COL].to_numpy().astype(np.int64)
            self.site[name] = df["site"].astype(str).to_numpy() if "site" in df.columns else np.full(len(df), "?")
            self.row_id[name] = df["row_id"].to_numpy() if "row_id" in df.columns else np.arange(len(df))
            self.vc[name] = vc_mask(df)
            self.admitted[name] = (df["admitted"].to_numpy(dtype=float) if "admitted" in df.columns else None)
            del df

    def strata(self, split: str) -> np.ndarray:
        """Etykiety warstw ESI × oddział."""
        return np.char.add(np.char.add(self.y[split].astype(str), "_"), self.site[split].astype(str))


@dataclass
class SourceCounts:
    """Liczności źródeł prób (do planu bez wczytywania cech)."""

    n_train: int
    n_val: int
    sites_train: dict[str, int]
    sites_val: dict[str, int]
    local_train: int
    local_val: int
    foreign_train: int
    local_esi1_val: int

    @staticmethod
    def from_arrays(site_train: np.ndarray, site_val: np.ndarray, y_val: np.ndarray, local_site: str) -> SourceCounts:
        st, sv = np.asarray(site_train).astype(str), np.asarray(site_val).astype(str)
        return SourceCounts(
            n_train=len(st), n_val=len(sv),
            sites_train={s: int((st == s).sum()) for s in sorted(set(st))},
            sites_val={s: int((sv == s).sum()) for s in sorted(set(sv))},
            local_train=int((st == local_site).sum()), local_val=int((sv == local_site).sum()),
            foreign_train=int((st != local_site).sum()),
            local_esi1_val=int(((sv == local_site) & (np.asarray(y_val) == 0)).sum()),
        )


def light_counts(local_site: str) -> SourceCounts:
    """Liczności z samych kolumn site / target (szybko — dla planu)."""
    from src.data.dataset import load_split

    train = load_split("train", columns=["site", config.TARGET_COL])
    val = load_split("val", columns=["site", config.TARGET_COL])
    return SourceCounts.from_arrays(train["site"].to_numpy(), val["site"].to_numpy(),
                                    val[config.TARGET_COL].to_numpy(), local_site)


# ─────────────────────────────────────────
# Dostępność TFM
# ─────────────────────────────────────────
def tfm_status(model: str, device_option: str = "auto") -> dict[str, Any]:
    """
    {'available', 'reason', 'device' ('cuda'|'cpu'), 'weights' {file, sha256}, 'library_version', 'fake',
    'override'} — bez budowania estymatora (zależności, wagi offline, urządzenie).
    """
    import importlib.util

    from src.models.tfm_base import TFMWeightsError, adapter_for, fake_mode

    out: dict[str, Any] = {"model": model, "available": False, "reason": None, "device": None, "weights": None,
                           "library_version": None, "fake": fake_mode(), "override": None}
    if not out["fake"]:
        missing = [lib for lib in ("torch", model) if importlib.util.find_spec(lib) is None]
        if missing:
            out["reason"] = (f"brak pakietu {' i '.join(missing)} — zainstaluj torch (koło CUDA) i "
                             f"`pip install -r requirements-tfm.txt -c requirements-lock.txt` (docs/TFM.md)")
            return out
    adapter = adapter_for(model)
    defaults = dict(getattr(config, f"{model.upper()}_DEFAULT_PARAMS", {}))
    try:
        weights = adapter.resolve_weights(defaults)
    except TFMWeightsError as exc:
        out["reason"] = f"brak wag — {exc}"
        return out
    except Exception as exc:  # noqa: BLE001 — uszkodzona instalacja biblioteki
        out["reason"] = f"wagi niedostępne ({type(exc).__name__}: {str(exc)[:200]})"
        return out
    out["weights"] = {"file": weights.get("file"), "sha256": weights.get("sha256"), "size": weights.get("size")}
    out["library_version"] = adapter.library_version()
    out["override"] = adapter.weights_override()
    if out["fake"]:
        out["device"] = "cuda" if device_option == "gpu" else "cpu"
    elif device_option == "cpu":
        out["device"] = "cpu"
    else:
        from src.utils.hardware import device_for, torch_info

        if device_option == "gpu":
            info = torch_info()
            if not info.get("cuda_available"):
                out["reason"] = f"--device gpu, a torch nie widzi CUDA ({info.get('error') or 'koło bez CUDA?'})"
                return out
            out["device"] = "cuda"
        else:
            try:
                out["device"] = "cuda" if device_for(model) == "gpu" else "cpu"
            except RuntimeError as exc:
                out["reason"] = f"wymuszone GPU niedostępne: {str(exc)[:300]}"
                return out
    out["available"] = True
    return out


# ─────────────────────────────────────────
# Modele jednostek
# ─────────────────────────────────────────
@contextlib.contextmanager
def _drop_log_records(logger_name: str, fragment: str) -> Iterator[None]:
    """Pomija komunikaty loggera zawierające `fragment` (spodziewane ostrzeżenia w pętli jednostek)."""

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                return fragment not in record.getMessage()
            except Exception:  # noqa: BLE001
                return True

    logger = logging.getLogger(logger_name)
    flt = _Filter()
    logger.addFilter(flt)
    try:
        yield
    finally:
        logger.removeFilter(flt)


def _lgbm_jobs() -> int:
    """Wątki LightGBM: jak w treningu (LGBM_DEFAULT_PARAMS['n_jobs'] = config.N_JOBS; testy: 1)."""
    return int(config.LGBM_DEFAULT_PARAMS.get("n_jobs") or config.N_JOBS or 1)


def inner_folds(y: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Warstwowa K-krotna CV, K = max(2, min(5, liczność najrzadszej obecnej klasy))."""
    import warnings

    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y).astype(np.int64)
    counts = np.bincount(y, minlength=config.N_CLASSES)
    present = counts[counts > 0]
    k = max(2, min(LGBM_MAX_FOLDS, int(present.min()) if len(present) else 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed))
        return [(np.asarray(a), np.asarray(b)) for a, b in splitter.split(np.zeros(len(y)), y)]


def _suggest(trial: Any) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, (low, high, log_scale) in LGBM_SPACE.items():
        if name in LGBM_INT_PARAMS:
            params[name] = trial.suggest_int(name, int(low), int(high), log=log_scale)
        else:
            params[name] = trial.suggest_float(name, float(low), float(high), log=log_scale)
    return params


def tune_lightgbm_small(X: Any, y: np.ndarray, *, n_trials: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Optuna w pamięci na samej podpróbie: TPE (seed), wewnętrzna warstwowa CV z wczesnym zatrzymaniem
    w każdym foldzie (log loss foldu walidacyjnego), przycinanie medianą po foldach. Zwraca (parametry
    finalne z n_estimators = średnia najlepsza iteracja, opis strojenia).
    """
    import optuna
    from sklearn.metrics import log_loss

    from src.models import get_model

    y = np.asarray(y).astype(np.int64)
    folds = inner_folds(y, seed)
    fixed = {"objective": "multiclass", "num_class": config.N_CLASSES, "n_estimators": LGBM_MAX_ROUNDS,
             "subsample_freq": 1, "max_depth": -1, "random_state": int(seed), "n_jobs": _lgbm_jobs(),
             "verbosity": -1}
    labels = list(range(config.N_CLASSES))

    def objective(trial: Any) -> float:
        params = {**fixed, **_suggest(trial)}
        losses: list[float] = []
        iterations: list[int] = []
        for step, (tr, va) in enumerate(folds):
            model = get_model("lightgbm", params=params)
            model.es_rounds = LGBM_ES_ROUNDS
            model.fit_estimator(X.iloc[tr], y[tr], X.iloc[va], y[va])
            proba = model.predict_proba(X.iloc[va])
            losses.append(float(log_loss(y[va], proba, labels=labels)))
            iterations.append(int(model.best_iteration or params["n_estimators"]))
            trial.report(float(np.mean(losses)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("best_iterations", iterations)
        return float(np.mean(losses))

    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=int(seed)),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        )
        study.optimize(objective, n_trials=int(n_trials), catch=(Exception,), gc_after_trial=False)
    finally:
        optuna.logging.set_verbosity(verbosity)
    states = [t.state for t in study.trials]
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError(f"Strojenie LightGBM: żadna z {len(study.trials)} prób nie zakończyła się "
                           f"(błędy / przycięcia) — zob. log")
    best = study.best_trial
    iterations = [int(i) for i in best.user_attrs.get("best_iterations") or [LGBM_MAX_ROUNDS]]
    n_estimators = max(1, int(round(float(np.mean(iterations)))))
    final = {**fixed, **best.params, "n_estimators": n_estimators}
    info = {
        "best_params": dict(best.params),
        "inner_cv_log_loss": float(best.value),
        "n_trials": int(n_trials),
        "n_complete": len(complete),
        "n_pruned": sum(s == optuna.trial.TrialState.PRUNED for s in states),
        "n_fail": sum(s == optuna.trial.TrialState.FAIL for s in states),
        "k_folds": len(folds),
        "best_iterations": iterations,
        "n_estimators": n_estimators,
    }
    return final, info


def fit_predict_lgbm(X: Any, y: np.ndarray, X_eval: Any, params: Mapping[str, Any]) -> np.ndarray:
    """LightGBM na całej podpróbie bez wczesnego zatrzymania (liczba drzew ustalona) → (n_eval, 5)."""
    from src.models import get_model

    model = get_model("lightgbm", params=dict(params))
    with _drop_log_records("src.models.base", "brak zbioru early stopping"):
        model.fit_estimator(X, np.asarray(y))
    return np.asarray(model.predict_proba(X_eval), dtype=np.float64)


def fit_predict_tfm(model_name: str, X_ctx: Any, y_ctx: np.ndarray, X_eval: Any, *, device: str,
                    seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """TFM „z pudełka” przez wrapper: kontekst = cała podpróba (1 worek) → (prawdopodobieństwa, opis)."""
    from src.models import get_model

    params = {**TFM_OUT_OF_BOX[model_name], "n_ctx": int(len(X_ctx)), "n_bags": 1, "max_bags": 1,
              "bag_mode": "partition", "context_strategy": "stratified", "max_features": None,
              "random_state": int(seed), "device": "cuda" if str(device).startswith(("cuda", "gpu")) else "cpu"}
    model = get_model(model_name, params=params)
    model.context_strata = None                       # kontekst = wszystkie wiersze (warstwy bez znaczenia)
    try:
        model.fit_estimator(X_ctx, np.asarray(y_ctx))
        proba = model.predict_proba_many([X_eval])[0]
        info = model.tfm_info()
    finally:
        model.release()
    throughput = info.get("throughput") or {}
    compact = {
        "library_version": info.get("library_version"),
        "weights": {k: (info.get("weights") or {}).get(k) for k in ("file", "sha256")},
        "weights_override": info.get("weights_override"),
        "fake": info.get("fake"),
        "device": info.get("device"),
        "n_context": int(len(X_ctx)),
        "runtime": info.get("runtime"),
        "predict_s": info.get("predict_s"),
        "fit_s": throughput.get("fit_s_per_bag"),
        "rows_per_s": throughput.get("rows_per_s"),
        "peak_vram_mb": throughput.get("peak_vram_mb"),
        "events": [e for e in info.get("events") or [] if e.get("kind") != "weights_override"],
    }
    return np.asarray(proba, dtype=np.float64), compact


def _split_es(n_rows: int, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Pozycje (dopasowanie, early stopping) — 10% warstwowo; klasa z 1 wierszem → bez warstw."""
    from sklearn.model_selection import train_test_split

    positions = np.arange(n_rows)
    try:
        fit, es = train_test_split(positions, test_size=config.EARLY_STOPPING_FRACTION, random_state=int(seed),
                                   stratify=np.asarray(y))
    except ValueError:
        fit, es = train_test_split(positions, test_size=config.EARLY_STOPPING_FRACTION, random_state=int(seed))
    return np.sort(fit), np.sort(es)


def lightgbm_source(feature_set: str, tag: str) -> Any | None:
    """Najnowszy ukończony, strojony run lightgbm z tagiem (common.source_runs) albo None."""
    from src.analysis.common import source_runs

    try:
        return source_runs(feature_set, tag, ["lightgbm"]).get("lightgbm")
    except Exception as exc:  # noqa: BLE001 — brak katalogu runs / danych: odniesienie z parametrów domyślnych
        log.warning(f"Nie udało się odczytać runów źródłowych lightgbm ({type(exc).__name__}: {exc})")
        return None


def fit_predict_transported(data: SplitData, train_pos: np.ndarray, eval_pos: np.ndarray, local_prior: np.ndarray,
                            run: Any | None, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    LightGBM z innych oddziałów (parametry runa źródłowego albo domyślne repo) + early stopping na 10%
    tych wierszy → prawdopodobieństwa na lokalnych wierszach oceny: 'raw' i 'prior_known'.
    """
    from src.analysis.priors import apply_prior_shift, class_prior
    from src.models import get_model
    from src.models.train import refit_params

    features = list(data.features)
    if run is not None:
        from src.models.train import _refit_count_scaling, _refit_features

        params, _ = refit_params("lightgbm", run.info.get("params") or {}, seed)
        features = _refit_features(Path(run.path), run.info, data.feature_set)
        source = run.name
    else:
        params = {k: v for k, v in config.LGBM_DEFAULT_PARAMS.items() if k not in ("n_jobs", "random_state")}
        params["random_state"] = int(seed)
        source = None
    params["n_jobs"] = _lgbm_jobs()
    X_all = data.X["train"].iloc[train_pos]
    y_all = data.y["train"][train_pos]
    fit_idx, es_idx = _split_es(len(train_pos), y_all, seed)
    scaling: dict[str, Any] = {}
    if run is not None:
        scaling = _refit_count_scaling("lightgbm", params, run.info, len(fit_idx))
    model = get_model("lightgbm", params=params)
    model.fit_estimator(X_all.iloc[fit_idx][features], y_all[fit_idx], X_all.iloc[es_idx][features], y_all[es_idx])
    X_eval = data.X["val"].iloc[eval_pos][features]
    raw = np.asarray(model.predict_proba(X_eval), dtype=np.float64)
    train_prior = class_prior(y_all[fit_idx])
    known = apply_prior_shift(raw, local_prior, train_prior)
    info = {
        "params_source": source,
        "params_note": (f"parametry runa {source}" if source else
                        "brak strojonego runa lightgbm z tym tagiem — parametry domyślne repo (LGBM_DEFAULT_PARAMS)"),
        "n_fit": int(len(fit_idx)), "n_es": int(len(es_idx)), "n_features": len(features),
        "best_iteration": model.best_iteration, "count_scaling": scaling,
        "priors": {"train": train_prior.tolist(), "local": np.asarray(local_prior).tolist()},
    }
    return {"raw": raw, "prior_known": known}, info


# ─────────────────────────────────────────
# Metryki jednostki
# ─────────────────────────────────────────
def unit_metrics(y: np.ndarray, proba: np.ndarray, vc: np.ndarray | None = None,
                 admitted: np.ndarray | None = None) -> dict[str, Any]:
    """
    Metryki na wierszach oceny: common.evaluate (QWK, niedotriażowanie, krytyczne pominięcia, czułość ESI 1
    dla reguł argmax i koszt; log loss, makro-AUC, AUC przyjęcia; kopia `vc_`) + Brier, ECE (top-label),
    ECE ESI1 / ESI5 (jeden-przeciw-reszcie) i liczniki ESI 1 (k trafionych / n) do przedziałów Clopper–Pearsona.
    """
    from src.evaluation.metrics import decide
    from src.evaluation.statistics import ece_classwise, point_metrics

    y = np.asarray(y).astype(np.int64)
    proba = np.asarray(proba, dtype=np.float64)
    preds = {rule: decide(proba, rule) for rule in RULES}
    out: dict[str, Any] = dict(evaluate(y, proba, preds, vc, admitted))
    out.update(point_metrics(y, proba, ["brier", "ece"]))
    classwise = ece_classwise(y, proba)
    out["ece_esi1"] = classwise.get("ESI1")
    out["ece_esi5"] = classwise.get("ESI5")
    if vc is not None and np.asarray(vc).any():
        mask = np.asarray(vc, dtype=bool)
        for key, value in point_metrics(y[mask], proba[mask], ["brier", "ece"]).items():
            out[f"vc_{key}"] = value
    esi1 = y == 0
    counts: dict[str, Any] = {"n": int(esi1.sum())}
    for rule, pred in preds.items():
        counts[f"k_{rule}"] = int((pred[esi1] == 0).sum())
        counts[f"critical_{rule}"] = int((pred[esi1] >= 3).sum())
    out["esi1_counts"] = counts
    return out


# ─────────────────────────────────────────
# Statystyka
# ─────────────────────────────────────────
def holm(pvalues: Sequence[float | None]) -> list[float | None]:
    """Korekta Holma (rodzina = podane p; None / NaN pomijane): p_(i) · (m − i + 1), monotonicznie."""
    idx = [i for i, p in enumerate(pvalues) if p is not None and math.isfinite(float(p))]
    order = sorted(idx, key=lambda i: float(pvalues[i]))      # type: ignore[arg-type]
    m = len(order)
    out: list[float | None] = [None] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(pvalues[i])))   # type: ignore[arg-type]
        out[i] = running
    return out


def t_interval(values: Sequence[float | None], alpha: float = ALPHA) -> dict[str, Any]:
    """Średnia, SD (n−1), 95% przedział t (df = n − 1) i dwustronne p testu t (H0: średnia = 0)."""
    from scipy import stats

    x = np.array([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    n = int(len(x))
    out: dict[str, Any] = {"n": n, "mean": float(x.mean()) if n else None, "sd": None, "se": None, "df": None,
                           "ci_low": None, "ci_high": None, "p": None}
    if n < 2:
        return out
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd <= 1e-12 * max(1.0, abs(mean)):                    # identyczne wartości (szum zmiennoprzecinkowy)
        sd = 0.0
    se = sd / math.sqrt(n)
    crit = float(stats.t.ppf(1 - alpha / 2, n - 1))
    if se > 0:
        p = float(2 * stats.t.sf(abs(mean) / se, n - 1))
    else:
        p = 0.0 if mean != 0 else 1.0
    out.update({"sd": sd, "se": se, "df": n - 1, "ci_low": mean - crit * se, "ci_high": mean + crit * se, "p": p})
    return out


def two_level_draws(diffs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Poziom 1 dwupoziomowego bootstrapu. `diffs` (R × B): Δ powtórzenia r w losowaniu wierszy b (poziom 2,
    wspólny dla powtórzeń). Replikacja b: R powtórzeń losowanych ze zwracaniem, Δ*_b = średnia ich Δ[·, b].
    """
    diffs = np.asarray(diffs, dtype=float)
    if diffs.ndim != 2 or diffs.shape[0] < 1:
        raise ValueError(f"two_level_draws: oczekiwano macierzy (R, B), jest {diffs.shape}")
    import warnings

    n_rep, n_draws = diffs.shape
    picks = rng.integers(0, n_rep, size=(n_draws, n_rep))
    values = diffs[picks, np.arange(n_draws)[:, None]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # replikacja bez wartości → NaN (pomijana)
        return np.nanmean(values, axis=1)


def boot_summary(samples: np.ndarray, alpha: float = ALPHA) -> dict[str, Any]:
    """
    Percentylowy przedział (statystyki pozycyjne c i B−c+1, c = ⌈B·alpha/2⌉) i dwustronne p = 2·min(P(Δ ≤ 0),
    P(Δ ≥ 0)) z tych samych ogonów (jak statistics.paired_bootstrap_diff — przedział bez 0 ⇔ p < alpha).
    """
    valid = np.asarray(samples, dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0:
        return {"ci_low": None, "ci_high": None, "p": None, "n_valid": 0}
    b = len(valid)
    ordered = np.sort(valid)
    c = max(1, int(math.ceil(round(b * alpha / 2, 9))))
    n_le, n_ge = int((valid <= 0).sum()), int((valid >= 0).sum())
    p = float(min(1.0, max(2.0 * min(n_le, n_ge) / b, 1.0 / b)))
    return {"ci_low": float(ordered[c - 1]), "ci_high": float(ordered[b - c]), "p": p, "n_valid": b}


def _power(n: np.ndarray, fit: Mapping[str, Any]) -> np.ndarray:
    return float(fit["a"]) + float(fit["b"]) * np.power(np.asarray(n, dtype=float), -float(fit["c"]))


def crossover(fit_arm: Mapping[str, Any] | None, fit_base: Mapping[str, Any] | None, low: float, high: float,
              higher_is_better: bool = False) -> dict[str, Any]:
    """
    Punkt przecięcia krzywych a + b·n^(−c) modelu i odniesienia w [low, high] (siatka logarytmiczna + brentq).
    status: 'arm_better_below' (model lepszy poniżej n*), 'arm_better_above', 'multiple' (kilka przecięć —
    n* = pierwsze), 'arm_better_everywhere', 'base_better_everywhere', 'no_fit'.
    """
    if not fit_arm or not fit_base or not (high > low > 0):
        return {"status": "no_fit", "n_star": None}
    from scipy.optimize import brentq

    def gap(n: Any) -> Any:
        diff = _power(n, fit_arm) - _power(n, fit_base)
        return -diff if higher_is_better else diff          # < 0 ⇔ model lepszy

    grid = np.geomspace(low, high, 1024)
    better = gap(grid) < 0
    if better.all():
        return {"status": "arm_better_everywhere", "n_star": None}
    if not better.any():
        return {"status": "base_better_everywhere", "n_star": None}
    switches = np.flatnonzero(better[1:] != better[:-1])
    i = int(switches[0])
    try:
        n_star = float(brentq(lambda v: float(gap(np.array([v]))[0]), grid[i], grid[i + 1]))
    except ValueError:
        n_star = float(np.sqrt(grid[i] * grid[i + 1]))
    if len(switches) > 1:
        status = "multiple"
    else:
        status = "arm_better_below" if better[0] else "arm_better_above"
    return {"status": status, "n_star": n_star}


@contextlib.contextmanager
def _quiet(logger_name: str, level: int = logging.ERROR) -> Iterator[None]:
    logger = logging.getLogger(logger_name)
    previous = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous)


def crossover_bootstrap(per_size: Mapping[int, tuple[Sequence[float], Sequence[float]]], n_boot: int, seed: int,
                        higher_is_better: bool = False) -> dict[str, Any]:
    """
    n* z krzywych potęgowych dopasowanych do średnich po powtórzeniach + bootstrap po powtórzeniach
    (w każdym n losowane ze zwracaniem pary (model, odniesienie) tego samego powtórzenia).
    per_size: {n: (wartości modelu, wartości odniesienia)} — sparowane po powtórzeniach.
    """
    from src.analysis.learning_curve import fit_power_law

    sizes = sorted(n for n, (a, b) in per_size.items() if len(a) and len(a) == len(b))
    out: dict[str, Any] = {"sizes": sizes, "status": "no_fit", "n_star": None, "fit_arm": None, "fit_base": None,
                           "ci_low": None, "ci_high": None, "n_boot": 0, "share": {}, "in_range": None}
    if len(sizes) < 3:
        out["reason"] = f"za mało rozmiarów n ({len(sizes)} < 3) do krzywej potęgowej"
        return out
    low, high = float(min(sizes)), float(max(sizes)) * CROSSOVER_FACTOR
    hib = bool(higher_is_better)
    arm_means = [float(np.mean(per_size[n][0])) for n in sizes]
    base_means = [float(np.mean(per_size[n][1])) for n in sizes]
    with _quiet("src.analysis.learning_curve"):
        fit_arm = fit_power_law(sizes, arm_means, (), hib)
        fit_base = fit_power_law(sizes, base_means, (), hib)
    point = crossover(fit_arm, fit_base, low, high, hib)
    out.update({"fit_arm": fit_arm, "fit_base": fit_base, **point, "search_range": [low, high]})
    if point["n_star"] is not None:
        out["in_range"] = bool(point["n_star"] <= max(sizes))
    if n_boot <= 0 or fit_arm is None or fit_base is None:
        return out
    rng = np.random.default_rng(int(seed))
    stars: list[float] = []
    statuses: dict[str, int] = {}
    arrays = {n: (np.asarray(per_size[n][0], dtype=float), np.asarray(per_size[n][1], dtype=float)) for n in sizes}
    with _quiet("src.analysis.learning_curve"):
        for _ in range(int(n_boot)):
            a_means, b_means = [], []
            for n in sizes:
                a, b = arrays[n]
                pick = rng.integers(0, len(a), size=len(a))
                a_means.append(float(a[pick].mean()))
                b_means.append(float(b[pick].mean()))
            res = crossover(fit_power_law(sizes, a_means, (), hib), fit_power_law(sizes, b_means, (), hib),
                            low, high, hib)
            statuses[res["status"]] = statuses.get(res["status"], 0) + 1
            if res["status"] == point["status"] and res["n_star"] is not None:
                stars.append(float(res["n_star"]))
    out["n_boot"] = int(n_boot)
    out["share"] = {k: v / float(n_boot) for k, v in sorted(statuses.items())}
    out["agreement"] = out["share"].get(point["status"], 0.0)       # odsetek losowań z tym samym wynikiem
    min_repeats = min(len(arrays[n][0]) for n in sizes)
    out["min_repeats"] = int(min_repeats)
    # Bootstrap po R = 3 powtórzeniach jest zbyt wąski (jak poziom 1 porównań) — przedział tylko orientacyjny
    out["ci_indicative"] = bool(min_repeats < MIN_REPEATS_NSTAR_CI)
    if point["n_star"] is None:
        return out
    if stars and len(stars) >= 0.5 * n_boot:
        lo, hi = np.percentile(stars, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        out["ci_low"], out["ci_high"] = float(lo), float(hi)
        if out["ci_indicative"]:
            out["ci_note"] = (f"orientacyjny: najmniej {min_repeats} powtórzeń przy jednym z rozmiarów "
                              f"(< {MIN_REPEATS_NSTAR_CI}) — bootstrap po tak małej liczbie powtórzeń zaniża rozrzut")
    else:
        out["ci_note"] = (f"przedział nie wyznaczony: przecięcie tego samego typu w {len(stars)}/{n_boot} "
                          f"losowań (< 50%)")
    return out


def decision_p(p_boot: float | None, p_t: float | None, n_repeats: int) -> float | None:
    """
    p do werdyktu: max(p dwupoziomowego bootstrapu, p sparowanego testu t, df = R − 1) — wygrana wymaga,
    by OBA przedziały wykluczały 0. Przy R = 2–5 percentylowy poziom 1 bootstrapu (losowanie R powtórzeń) jest
    zbyt wąski — gdy wszystkie Δ mają ten sam znak, p bootstrapu spada do 1/B niezależnie od rozrzutu między
    podpróbami (antykonserwatywnie); test t z kolei pomija wspólny szum wierszy oceny. R < 2 → None
    (bez testu → werdykt 'na'). Bez bootstrapu (`--bootstrap 0`) — p testu t.
    """
    if int(n_repeats) < MIN_REPEATS_TEST:
        return None
    values = [float(p) for p in (p_boot, p_t) if p is not None and math.isfinite(float(p))]
    return max(values) if values else None


def _mid(stat_metric: str) -> float | None:
    from src.evaluation.report_sections import MINIMAL_IMPORTANT_DIFFERENCE

    return MINIMAL_IMPORTANT_DIFFERENCE.get(stat_metric)


def verdict(delta: float | None, p: float | None, metric_key: str, alpha: float = ALPHA) -> tuple[str, str]:
    """
    Reguła wygranej: p (dla metryki pierwszorzędowej — po korekcji Holma) < alpha I |Δ| > MID.
    Zwraca (kod 'win' | 'loss' | 'tie' | 'tie_small' | 'na', opis po polsku); Δ = model − odniesienie.
    """
    if delta is None or p is None or not math.isfinite(float(delta)):
        return "na", "—"
    stat_metric = COMPARE_METRICS.get(metric_key, (metric_key, "*"))[0]
    mid = _mid(stat_metric)
    lower_better = not HIGHER_IS_BETTER.get(metric_key, False)
    if p >= alpha:
        return "tie", "brak istotnej różnicy"
    if mid is not None and abs(delta) <= mid:
        return "tie_small", f"istotna, ale |Δ| ≤ MID {mid:g}"
    arm_better = (delta < 0) if lower_better else (delta > 0)
    suffix = "" if mid is not None else " (bez MID)"
    return ("win", "model lepszy" + suffix) if arm_better else ("loss", "odniesienie lepsze" + suffix)


# ─────────────────────────────────────────
# Jednostki i plan
# ─────────────────────────────────────────
def arm_base(arm: str) -> str:
    """Model bazowy ramienia: 'tabpfn+foreign' → 'tabpfn', 'lightgbm_tuned' → 'lightgbm'."""
    if arm.endswith(FOREIGN_SUFFIX):
        return arm[: -len(FOREIGN_SUFFIX)]
    return "lightgbm" if arm.startswith("lightgbm") else arm


def unit_name(scenario: str, n: int | None, repeat: int | None, arm: str) -> str:
    if n is None:
        return f"{scenario}/transported/{arm}"
    return f"{scenario}/n{int(n)}/r{int(repeat)}/{arm}"


def preds_file(unit: str, variant: str | None = None) -> str:
    stem = unit.replace("/", "__")
    return f"preds/{stem}{'__' + variant if variant else ''}.npy"


def arms_for(scenario: str, models: Sequence[str], foreign_rows: int) -> list[str]:
    arms = list(LGBM_ARMS) + list(models)
    if scenario == "site_local" and foreign_rows > 0:
        arms += [f"{m}{FOREIGN_SUFFIX}" for m in models]
    return arms


def build_units(cfg: SmallDataConfig, counts: SourceCounts, statuses: Mapping[str, Mapping[str, Any]],
                transport_source: str | None) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """
    Jednostki do policzenia (kolejność grup: LightGBM, TabPFN, TabICL) i pominięte (z powodem): rozmiar
    większy niż źródło, TFM niedostępny, kontekst > TFM_CPU_MAX_CONTEXT na CPU.
    """
    units: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    cpu_limit = int(config.TFM_CPU_MAX_CONTEXT)
    foreign_n = min(int(cfg.foreign_rows), counts.foreign_train)
    available_tfm = [m for m in cfg.models if statuses.get(m, {}).get("available")]
    for scenario in cfg.scenarios:
        source_rows = counts.n_train if scenario == "pooled" else counts.local_train
        arms = arms_for(scenario, available_tfm, foreign_n)
        for n, repeats in zip(cfg.sizes, cfg.repeats, strict=True):
            if n > source_rows:
                skipped.append({"unit": f"{scenario}/n{n}", "reason": f"n = {n:,} > {source_rows:,} wierszy źródła"})
                continue
            for r in range(repeats):
                for arm in arms:
                    base = arm_base(arm)
                    foreign = arm.endswith(FOREIGN_SUFFIX)
                    name = unit_name(scenario, n, r, arm)
                    context = n + (foreign_n if foreign else 0)
                    if base in config.TFM_MODELS:
                        device = statuses[base]["device"]
                        if device == "cpu" and context > cpu_limit:
                            skipped.append({"unit": name, "reason": f"CPU: kontekst {context:,} > TFM_CPU_MAX_CONTEXT="
                                                                    f"{cpu_limit:,} (uruchom na GPU)"})
                            continue
                    units.append({"unit": name, "model": arm, "arm": arm, "base_model": base, "scenario": scenario,
                                  "n": int(n), "repeat": int(r), "foreign": foreign, "context": int(context),
                                  "group": base, "source_run": "none",
                                  "seed": derive_seed(cfg.seed, scenario, n, r, base)})
        if scenario == "site_local" and cfg.transport_rows > 0:
            if counts.foreign_train == 0:
                skipped.append({"unit": unit_name(scenario, None, None, TRANSPORTED),
                                "reason": "brak wierszy innych oddziałów"})
            else:
                units.append({"unit": unit_name(scenario, None, None, TRANSPORTED), "model": TRANSPORTED,
                              "arm": TRANSPORTED, "base_model": "lightgbm", "scenario": scenario, "n": None,
                              "repeat": None, "foreign": False,
                              "context": int(min(cfg.transport_rows, counts.foreign_train)), "group": "lightgbm",
                              "source_run": transport_source or "none",
                              "seed": derive_seed(cfg.seed, scenario, "transported")})
    for model in cfg.models:
        if model not in available_tfm:
            skipped.append({"unit": f"*/{model}", "reason": str(statuses.get(model, {}).get("reason"))})
    group_order = {"lightgbm": 0, "tabpfn": 1, "tabicl": 2}
    units.sort(key=lambda u: group_order.get(u["group"], 3))           # stabilnie: w grupie kolejność planu
    return units, skipped


def benchmark_unusable_reason(data: Mapping[str, Any] | None) -> str | None:
    """
    Czy benchmark 13_tfm_setup NIE nadaje się do szacunków GPU (powód) — pomiar na CPU, atrapa
    (SOR_AI_TFM_FAKE) albo przekierowane (losowe) wagi; brak klucza = format tolerancyjny (używany).
    """
    if not isinstance(data, Mapping):
        return "brak pliku"
    if data.get("mechanics_only"):
        return f"test mechaniki ({data['mechanics_only']})"
    if data.get("fake"):
        return "atrapa estymatora (SOR_AI_TFM_FAKE)"
    if data.get("device") not in (None, "cuda"):
        return f"pomiar na '{data.get('device')}', nie na GPU"
    weights = data.get("weights") if isinstance(data.get("weights"), Mapping) else {}
    if any(isinstance(w, Mapping) and w.get("override") for w in weights.values()):
        return "przekierowane wagi (SOR_AI_*_MODEL_PATH)"
    return None


def _benchmark_rates(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    """
    {model: [(n_ctx, rows_per_s, fit_s)]} z benchmark.json 13_tfm_setup (`results`: rekordy {model, n_ctx,
    n_estimators, status, fit_s, rows_per_s, …}; format tolerancyjny), przeliczone na liczbę estymatorów
    eksperymentu A (koszt ~ liniowy w estymatorach); rekordy ze statusem ≠ ok pomijane. Pomiar na CPU,
    z atrapą albo przekierowanymi wagami (`benchmark_unusable_reason`) — pomijany w całości.
    """
    data = read_json(path)
    if not data:
        return {}
    reason = benchmark_unusable_reason(data)
    if reason is not None:
        log.info(f"Benchmark {path} pominięty w szacunkach: {reason}")
        return {}
    records: list[dict[str, Any]] = []
    if isinstance(data.get("results"), list):
        records += [dict(r) for r in data["results"] if isinstance(r, Mapping)]
    models = data.get("models") if isinstance(data.get("models"), Mapping) else data
    for model in config.TFM_MODELS:
        entry = models.get(model) if isinstance(models, Mapping) else None
        rows = entry.get("results") or entry.get("points") if isinstance(entry, Mapping) else entry
        if isinstance(rows, list):
            records += [{"model": model, **r} for r in rows if isinstance(r, Mapping)]
    out: dict[str, list[tuple[float, float, float]]] = {}
    for r in records:
        if r.get("status") not in (None, "ok"):
            continue
        try:
            model, n_ctx, rate = str(r["model"]), float(r["n_ctx"]), float(r["rows_per_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if model in config.TFM_MODELS and n_ctx > 0 and rate > 0:
            try:
                bench_est = float(r.get("n_estimators") or _UNIT_ESTIMATORS[model])
            except (TypeError, ValueError):
                bench_est = float(_UNIT_ESTIMATORS[model])
            scale = _UNIT_ESTIMATORS[model] / max(bench_est, 1.0)
            out.setdefault(model, []).append((n_ctx, rate / scale, float(r.get("fit_s") or 0.0) * scale))
    return {m: sorted(v) for m, v in out.items()}


def _interp_log(points: Sequence[tuple[float, float]], x: float) -> float:
    xs = np.log([p[0] for p in points])
    ys = np.log([max(p[1], 1e-9) for p in points])
    return float(np.exp(np.interp(math.log(max(x, 1.0)), xs, ys)))


def estimate_unit_s(u: Mapping[str, Any], n_eval: int, lgbm_trials: int,
                    rates: Mapping[str, Sequence[tuple[float, float, float]]], device: str | None) -> float | None:
    """Orientacyjny czas jednostki (s): benchmark 13_tfm_setup → plan pamięci (GPU) → None (CPU bez pomiaru)."""
    arm, n = str(u["arm"]), int(u.get("n") or u.get("context") or 0)

    def lgbm_fit(rows: float) -> float:
        return 0.3 + rows / 4_000.0

    if arm == "lightgbm_tuned":
        k = LGBM_MAX_FOLDS if n >= 1_000 else 3
        return lgbm_trials * k * lgbm_fit(n * (k - 1) / k) * 0.7 + lgbm_fit(n)
    if arm == "lightgbm_default":
        return 0.2 + n / 20_000.0
    if arm == TRANSPORTED:
        return 30.0 + float(u.get("context") or 0) / 1_000.0
    base, context = str(u["base_model"]), int(u.get("context") or n)
    measured = rates.get(base)
    if measured:
        rate = _interp_log([(p[0], p[1]) for p in measured], context)
        fit = _interp_log([(p[0], max(p[2], 0.01)) for p in measured], context)
        return fit + n_eval / rate
    if device == "cuda":
        table = sorted(_MEMPLAN_UNIT_S[base].items())
        return _interp_log([(float(a), float(b)) for a, b in table], context) * max(n_eval, 1) / 20_000.0
    return None


def lightgbm_eta_s(cfg: SmallDataConfig | None = None, counts: SourceCounts | None = None) -> float:
    """
    Szacunek części LightGBM planu (strojony + domyślny + przeniesiony; ta sama formuła co `--plan`), bez
    wczytywania danych: brak `counts` → źródła ≥ największego rozmiaru (prawdziwe dane: żaden rozmiar nie
    odpada) i pełne `transport_rows` — górne oszacowanie. Używane przez orkiestrator (ETA kroku small_data)
    i 13_tfm_setup (szacunek eksperymentu A); jednostki TFM liczone osobno (benchmark).
    """
    cfg = cfg or SmallDataConfig()
    sizes, repeats = parse_sizes_repeats(cfg.sizes, cfg.repeats)
    big = max(sizes) + int(cfg.transport_rows) + 1
    counts = counts or SourceCounts(n_train=big, n_val=big, sites_train={}, sites_val={}, local_train=big,
                                    local_val=big, foreign_train=big, local_esi1_val=0)
    plan_cfg = SmallDataConfig(models=[], scenarios=list(cfg.scenarios), sizes=sizes, repeats=repeats,
                               foreign_rows=0, transport_rows=int(cfg.transport_rows), lgbm_trials=int(cfg.lgbm_trials))
    units, _ = build_units(plan_cfg, counts, {}, None)
    return float(sum(estimate_unit_s(u, 0, int(cfg.lgbm_trials), {}, None) or 0.0 for u in units))


# ─────────────────────────────────────────
# Wyniki: zbieranie i statystyka
# ─────────────────────────────────────────
def _mean_ci(values: Sequence[float | None]) -> dict[str, Any]:
    t = t_interval(values)
    clean = [None if v is None or not math.isfinite(float(v)) else float(v) for v in values]
    return {"mean": t["mean"], "sd": t["sd"], "ci_low": t["ci_low"], "ci_high": t["ci_high"], "n": t["n"],
            "values": clean}


def comparisons_for(scenario: str, arms: Sequence[str]) -> list[tuple[str, str]]:
    """Pary (model, odniesienie): każdy model vs lightgbm_tuned; w site_local też TFM+foreign vs TFM."""
    pairs = [(arm, BASELINE) for arm in arms if arm not in (BASELINE, TRANSPORTED)]
    pairs += [(arm, arm_base(arm)) for arm in arms if arm.endswith(FOREIGN_SUFFIX) and arm_base(arm) in arms]
    return pairs


def collect_statistics(store_root: Path, scenario: str, done: Sequence[Mapping[str, Any]], y_eval: np.ndarray,
                       n_boot: int, seed: int) -> dict[str, Any]:
    """
    Punkty krzywych (średnia ± 95% t po powtórzeniach), porównania sparowane (t + dwupoziomowy bootstrap,
    Holm między n dla log loss), krzywe potęgowe i n* — dla jednego scenariusza.
    """
    from src.evaluation.statistics import bootstrap_samples

    by_n: dict[int, dict[int, dict[str, Mapping[str, Any]]]] = {}
    for u in done:
        if u.get("n") is None:
            continue
        by_n.setdefault(int(u["n"]), {}).setdefault(int(u["repeat"]), {})[str(u["arm"])] = u
    arms = list(dict.fromkeys(str(u["arm"]) for u in done if u.get("n") is not None))
    pairs = comparisons_for(scenario, arms)
    points: dict[str, dict[str, Any]] = {}
    for arm in arms:
        for n in sorted(by_n):
            rows = [by_n[n][r][arm] for r in sorted(by_n[n]) if arm in by_n[n][r]]
            if not rows:
                continue
            metrics = {key: _mean_ci([(row.get("metrics") or {}).get(key) for row in rows])
                       for key in (*TABLE_METRICS, "recall_esi1_argmax", "critical_miss_cost", "vc_log_loss",
                                   "vc_qwk_argmax", "vc_qwk_cost", "vc_undertriage_cost", "admission_auc")}
            esi1 = [(row.get("metrics") or {}).get("esi1_counts") or {} for row in rows]
            points.setdefault(arm, {})[str(n)] = {"R": len(rows), "metrics": metrics, "esi1_counts": esi1,
                                                  "repeats": [int(row["repeat"]) for row in rows]}

    stat_metrics = sorted({m for m, _ in COMPARE_METRICS.values()})
    comparisons: list[dict[str, Any]] = []
    for n in sorted(by_n):
        reps = by_n[n]
        point_d: dict[tuple[str, str, str], dict[int, float]] = {}
        boot_d: dict[tuple[str, str, str], dict[int, np.ndarray]] = {}
        for r in sorted(reps):
            present = reps[r]
            needed = sorted({a for pair in pairs if pair[0] in present and pair[1] in present for a in pair})
            if not needed:
                continue
            probas = {arm: np.load(store_root / present[arm]["preds"]).astype(np.float64) for arm in needed}
            if n_boot > 0:
                bs = bootstrap_samples(y_eval, probas, rules=RULES, metrics=stat_metrics, n_boot=int(n_boot),
                                       seed=derive_seed(seed, scenario, "rows"))
                point_of = bs.point
            else:
                from src.evaluation.statistics import point_metrics

                bs = None
                point_of = {}
                for arm, p in probas.items():
                    vals: dict[tuple[str, str], float] = {}
                    for rule in RULES:
                        for metric, value in point_metrics(y_eval, p, stat_metrics, rule).items():
                            key = ("*", metric) if metric in ("log_loss", "brier", "ece") else (rule, metric)
                            vals[key] = value
                    point_of[arm] = vals
            for arm, base in pairs:
                if arm not in probas or base not in probas:
                    continue
                for key, (metric, rule) in COMPARE_METRICS.items():
                    k = ("*", metric) if rule == "*" else (rule, metric)
                    point_d.setdefault((arm, base, key), {})[r] = float(point_of[arm][k] - point_of[base][k])
                    if bs is not None:
                        boot_d.setdefault((arm, base, key), {})[r] = bs.samples[arm][k] - bs.samples[base][k]
            del probas
        for (arm, base, key), by_rep in point_d.items():
            reps_used = sorted(by_rep)
            deltas = [by_rep[r] for r in reps_used]
            t = t_interval(deltas)
            row: dict[str, Any] = {"arm": arm, "base": base, "n": int(n), "metric": key, "R": len(reps_used),
                                   "repeats": reps_used, "deltas": deltas, "delta": t["mean"], "sd": t["sd"],
                                   "t_ci_low": t["ci_low"], "t_ci_high": t["ci_high"], "p_t": t["p"],
                                   "boot_ci_low": None, "boot_ci_high": None, "p_boot": None}
            if (arm, base, key) in boot_d:
                diffs = np.stack([boot_d[(arm, base, key)][r] for r in reps_used])
                rng = np.random.default_rng(derive_seed(seed, scenario, n, len(reps_used), "repeats"))
                summary = boot_summary(two_level_draws(diffs, rng))
                row.update({"boot_ci_low": summary["ci_low"], "boot_ci_high": summary["ci_high"],
                            "p_boot": summary["p"], "n_boot_valid": summary["n_valid"]})
            comparisons.append(row)

    # p werdyktu = max(p bootstrapu, p testu t) (R ≥ 2); Holm między rozmiarami n (metryka pierwszorzędowa,
    # każda para osobno); werdykty
    for c in comparisons:
        c["p"] = decision_p(c["p_boot"], c["p_t"], c["R"])
    for arm, base in pairs:
        rows = [c for c in comparisons if c["arm"] == arm and c["base"] == base and c["metric"] == PRIMARY_METRIC]
        adjusted = holm([c["p"] for c in rows])
        for c, p_adj in zip(rows, adjusted, strict=True):
            c["p_holm"] = p_adj
    for c in comparisons:
        p_used = c.get("p_holm") if c["metric"] == PRIMARY_METRIC else c["p"]
        if c["R"] < MIN_REPEATS_TEST:
            c["verdict"], c["verdict_text"] = "na", f"R = {c['R']} < {MIN_REPEATS_TEST} — bez testu"
        else:
            c["verdict"], c["verdict_text"] = verdict(c["delta"], p_used, c["metric"])
        c["mid"] = _mid(COMPARE_METRICS[c["metric"]][0])

    # Krzywe potęgowe i n* (metryka pierwszorzędowa)
    from src.analysis.learning_curve import fit_power_law

    power: dict[str, Any] = {}
    for arm in arms:
        pts = points.get(arm) or {}
        sizes = sorted(int(n) for n in pts)
        with _quiet("src.analysis.learning_curve"):
            power[arm] = fit_power_law(sizes, [pts[str(n)]["metrics"][PRIMARY_METRIC]["mean"] for n in sizes],
                                       (), False)
    crossovers: dict[str, Any] = {}
    for arm, base in pairs:
        per_size: dict[int, tuple[list[float], list[float]]] = {}
        for n in sorted(by_n):
            a_vals, b_vals = [], []
            for r in sorted(by_n[n]):
                ua, ub = by_n[n][r].get(arm), by_n[n][r].get(base)
                va = (ua or {}).get("metrics", {}).get(PRIMARY_METRIC) if ua else None
                vb = (ub or {}).get("metrics", {}).get(PRIMARY_METRIC) if ub else None
                if va is not None and vb is not None:
                    a_vals.append(float(va))
                    b_vals.append(float(vb))
            if a_vals:
                per_size[n] = (a_vals, b_vals)
        crossovers[f"{arm}|{base}"] = {"arm": arm, "base": base, **crossover_bootstrap(
            per_size, int(n_boot), derive_seed(seed, scenario, arm, base, "crossover"))}
    return {"points": points, "comparisons": comparisons, "power_law": power, "crossover": crossovers,
            "arms": arms}


# ─────────────────────────────────────────
# Wykresy
# ─────────────────────────────────────────
def _arm_label(arm: str) -> str:
    base = arm_base(arm)
    if base in TFM_DISPLAY:
        return TFM_DISPLAY[base] + (" + obce oddziały" if arm.endswith(FOREIGN_SUFFIX) else "")
    return {"lightgbm_tuned": "LightGBM (strojony)", "lightgbm_default": "LightGBM (domyślny)",
            TRANSPORTED: "LightGBM przeniesiony"}.get(arm, arm)


def _size_ticks(ax: Any, points: Mapping[str, Mapping[str, Any]]) -> None:
    """Oś n w skali log z podziałką w rozmiarach eksperymentu (500, 1 000, …) zamiast potęg dziesięciu."""
    from matplotlib.ticker import NullLocator

    sizes = sorted({int(n) for by_n in points.values() for n in by_n})
    ax.set_xscale("log")
    if sizes:
        ax.set_xticks(sizes)
        ax.set_xticklabels([_n(n) for n in sizes], fontsize=8)
        ax.xaxis.set_minor_locator(NullLocator())


def plot_curves(scenario: str, block: Mapping[str, Any], path: Path, title: str) -> Path | None:
    """Krzywe (średnia ± 95% t po powtórzeniach) log loss / QWK / niedotriażowania vs n; potęgowa dla log loss."""
    from src.evaluation import visualizations  # noqa: F401, I001 — najpierw backend Agg (poza notebookiem)
    import matplotlib.pyplot as plt

    points = block.get("points") or {}
    if not points:
        return None
    fig, axes = plt.subplots(1, len(CURVE_METRICS), figsize=(5.0 * len(CURVE_METRICS), 4.3))
    transported = (block.get("transported") or {}).get("variants") or {}
    for ax, (metric, label) in zip(np.atleast_1d(axes), CURVE_METRICS, strict=True):
        for arm, by_n in points.items():
            sizes = sorted(int(n) for n in by_n)
            stats = [by_n[str(n)]["metrics"].get(metric) or {} for n in sizes]
            xs = np.array([n for n, s in zip(sizes, stats, strict=True) if s.get("mean") is not None], dtype=float)
            if not len(xs):
                continue
            ys = np.array([s["mean"] for s in stats if s.get("mean") is not None], dtype=float)
            lo = np.array([s["ci_low"] if s.get("ci_low") is not None else s["mean"] for s in stats
                           if s.get("mean") is not None], dtype=float)
            hi = np.array([s["ci_high"] if s.get("ci_high") is not None else s["mean"] for s in stats
                           if s.get("mean") is not None], dtype=float)
            style, marker = ARM_STYLE.get(arm, ("-", "o"))
            color = ARM_COLORS.get(arm, "#52514e")
            ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt=marker, linestyle=style, color=color, lw=1.6, ms=5,
                        capsize=3, elinewidth=1.0, label=_arm_label(arm))
            fit = (block.get("power_law") or {}).get(arm) if metric == PRIMARY_METRIC else None
            if fit:
                grid = np.geomspace(xs.min(), xs.max(), 100)
                ax.plot(grid, _power(grid, fit), color=color, lw=1.0, alpha=0.5, linestyle="-")
        for variant, ls in (("raw", "-."), ("prior_known", ":")):
            value = (transported.get(variant) or {}).get(metric)
            if value is not None:
                ax.axhline(float(value), color=ARM_COLORS[TRANSPORTED], lw=1.4, linestyle=ls,
                           label=f"{_arm_label(TRANSPORTED)} ({'surowy' if variant == 'raw' else 'korekta priorów'})")
        _size_ticks(ax, points)
        ax.set_xlabel("n — wiersze uczące (lokalne etykiety)")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, max(1, len(labels))), fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    rows = math.ceil(len(labels) / 4) if labels else 0
    fig.tight_layout(rect=(0, 0.045 * rows + 0.02, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_deltas(scenario: str, block: Mapping[str, Any], path: Path, title: str) -> Path | None:
    """Δ (model − LightGBM strojony) vs n z 95% CI dwupoziomowego bootstrapu (albo t) i pasmem ±MID."""
    from src.evaluation import visualizations  # noqa: F401, I001 — najpierw backend Agg
    import matplotlib.pyplot as plt

    rows = [c for c in block.get("comparisons") or [] if c["base"] == BASELINE]
    if not rows:
        return None
    panels = (("log_loss", "Δ log loss (< 0 = model lepszy)"), ("qwk_argmax", "Δ QWK argmax (> 0 = model lepszy)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.3))
    arms = list(dict.fromkeys(c["arm"] for c in rows))
    for ax, (metric, label) in zip(axes, panels, strict=True):
        mid = _mid(COMPARE_METRICS[metric][0])
        if mid:
            ax.axhspan(-mid, mid, color="#9a9a96", alpha=0.15, lw=0, label="±MID (minimalna istotna różnica)")
            label = f"{label}; MID {mid:g}"
        ax.axhline(0.0, color="#52514e", lw=1.0)
        for j, arm in enumerate(arms):
            sel = sorted((c for c in rows if c["arm"] == arm and c["metric"] == metric and c["delta"] is not None),
                         key=lambda c: c["n"])
            if not sel:
                continue
            offset = 1.0 + 0.04 * (j - (len(arms) - 1) / 2)
            xs = np.array([c["n"] for c in sel], dtype=float) * offset
            ys = np.array([c["delta"] for c in sel], dtype=float)
            lo = np.array([c["boot_ci_low"] if c["boot_ci_low"] is not None else
                           (c["t_ci_low"] if c["t_ci_low"] is not None else c["delta"]) for c in sel], dtype=float)
            hi = np.array([c["boot_ci_high"] if c["boot_ci_high"] is not None else
                           (c["t_ci_high"] if c["t_ci_high"] is not None else c["delta"]) for c in sel], dtype=float)
            style, marker = ARM_STYLE.get(arm, ("-", "o"))
            ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt=marker, linestyle=style, color=ARM_COLORS.get(arm, "#52514e"),
                        lw=1.4, ms=5, capsize=3, elinewidth=1.0, label=_arm_label(arm))
        _size_ticks(ax, {"_": {str(c["n"]): None for c in rows}})
        ax.set_xlabel("n — wiersze uczące")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, max(1, len(labels))), fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    rows = math.ceil(len(labels) / 4) if labels else 0
    fig.tight_layout(rect=(0, 0.045 * rows + 0.02, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ─────────────────────────────────────────
# Raport (po polsku)
# ─────────────────────────────────────────
def _f(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "—" if not math.isfinite(v) else f"{v:.{digits}f}"


def _ci(low: Any, high: Any, digits: int = 4) -> str:
    if low is None or high is None:
        return "—"
    return f"[{_f(low, digits)}; {_f(high, digits)}]"


def _p(value: Any) -> str:
    if value is None:
        return "—"
    v = float(value)
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def _n(value: Any) -> str:
    return "—" if value is None else f"{int(round(float(value))):,}".replace(",", " ")


SCENARIO_TITLES = {"pooled": "pooled — wszystkie oddziały", "site_local": "site_local — jeden oddział"}
_STATUS_TEXT = {
    "arm_better_below": "model lepszy PONIŻEJ n*", "arm_better_above": "model lepszy POWYŻEJ n*",
    "multiple": "kilka przecięć (n* = pierwsze)", "arm_better_everywhere": "model lepszy w całym zakresie",
    "base_better_everywhere": "odniesienie lepsze w całym zakresie", "no_fit": "brak dopasowania",
}


def _esi1_rows(scenario: str, block: Mapping[str, Any], n_esi1: int) -> list[str]:
    from src.evaluation.statistics import clopper_pearson

    lines = []
    for arm, by_n in (block.get("points") or {}).items():
        for n in sorted(int(k) for k in by_n):
            if n < RECALL_MIN_N:
                continue
            counts = by_n[str(n)].get("esi1_counts") or []
            cells = []
            for key in ("k_argmax", "k_cost", "critical_cost"):
                ks = [c.get(key) for c in counts if c.get(key) is not None]
                if not ks or not n_esi1:
                    cells.append("—")
                    continue
                k_mean = float(np.mean(ks))
                low, high = clopper_pearson(int(round(k_mean)), int(n_esi1))
                cells.append(f"{k_mean / n_esi1:.3f} {_ci(low, high, 3)} (zakres {min(ks) / n_esi1:.3f}–"
                             f"{max(ks) / n_esi1:.3f})")
            lines.append(f"| {_arm_label(arm)} | {_n(n)} | {len(counts)} | " + " | ".join(cells) + " |")
    transported = block.get("transported") or {}
    for variant, counts in (transported.get("esi1_counts") or {}).items():
        cells = []
        for key in ("k_argmax", "k_cost", "critical_cost"):
            k, total = counts.get(key), counts.get("n")
            if k is None or not total:
                cells.append("—")
                continue
            low, high = clopper_pearson(int(k), int(total))
            cells.append(f"{k / total:.3f} {_ci(low, high, 3)}")
        label = f"{_arm_label(TRANSPORTED)} ({'surowy' if variant == 'raw' else 'korekta priorów'})"
        lines.append(f"| {label} | {_n(transported.get('n_fit'))} | 1 | " + " | ".join(cells) + " |")
    return lines


def decision_lines(results: Mapping[str, Any]) -> list[str]:
    """Wniosek „TFM wygrywa poniżej n* …” z werdyktów (Holm + MID) i punktu przecięcia krzywych."""
    lines: list[str] = []
    for scenario, block in (results.get("scenarios") or {}).items():
        for arm in block.get("arms") or []:
            if arm_base(arm) not in config.TFM_MODELS:
                continue
            rows = sorted((c for c in block.get("comparisons") or [] if c["arm"] == arm and c["base"] == BASELINE
                           and c["metric"] == PRIMARY_METRIC), key=lambda c: c["n"])
            if not rows:
                continue
            groups = {code: [c["n"] for c in rows if c["verdict"] == code] for code in ("win", "loss", "tie", "tie_small")}
            parts = []
            if groups["win"]:
                parts.append("wygrywa przy n ∈ {" + ", ".join(_n(n) for n in groups["win"]) + "}")
            if groups["loss"]:
                parts.append("przegrywa przy n ∈ {" + ", ".join(_n(n) for n in groups["loss"]) + "}")
            ties = groups["tie"] + groups["tie_small"]
            if ties:
                parts.append("bez rozstrzygnięcia przy n ∈ {" + ", ".join(_n(n) for n in sorted(ties)) + "}")
            cross = (block.get("crossover") or {}).get(f"{arm}|{BASELINE}") or {}
            if cross.get("n_star") is not None:
                where = "w zakresie danych" if cross.get("in_range") else "EKSTRAPOLACJA poza zakres n"
                ci = (f"95% CI {_n(cross['ci_low'])}–{_n(cross['ci_high'])}"
                      + (" — ORIENTACYJNY (mało powtórzeń)" if cross.get("ci_indicative") else "")
                      if cross.get("ci_low") is not None else cross.get("ci_note") or "bez przedziału")
                cross_text = (f"przecięcie krzywych potęgowych n* ≈ {_n(cross['n_star'])} ({ci}; {where}; "
                              f"{_STATUS_TEXT.get(cross['status'], cross['status'])})")
            else:
                cross_text = (f"krzywe potęgowe: {_STATUS_TEXT.get(cross.get('status'), '—')}"
                              + (f" ({cross['reason']})" if cross.get("reason") else ""))
            wins = groups["win"]
            if wins and not groups["loss"]:
                summary = f"**TFM wygrywa do n = {_n(max(wins))}**" + (" (dalej bez rozstrzygnięcia)" if ties else "")
            elif wins and groups["loss"] and max(wins) < min(groups["loss"]):
                summary = (f"**TFM wygrywa poniżej n ≈ {_n(min(groups['loss']))}** (wygrane do {_n(max(wins))}, "
                           f"przegrane od {_n(min(groups['loss']))})")
            elif wins:
                summary = "**wynik niejednoznaczny** (wygrane i przegrane przeplatają się)"
            elif groups["loss"]:
                summary = "**TFM nie wygrywa przy żadnym n** (strojony LightGBM lepszy lub remis)"
            else:
                summary = "**brak rozstrzygnięcia przy żadnym n**"
            lines.append(f"- {scenario} · {_arm_label(arm)}: {summary}; " + "; ".join(parts) + f"; {cross_text}.")
        for arm in block.get("arms") or []:
            if not arm.endswith(FOREIGN_SUFFIX):
                continue
            own = arm_base(arm)
            rows = sorted((c for c in block.get("comparisons") or [] if c["arm"] == arm and c["base"] == own
                           and c["metric"] == PRIMARY_METRIC), key=lambda c: c["n"])
            if not rows:
                continue
            helps = [c["n"] for c in rows if c["verdict"] == "win"]
            hurts = [c["n"] for c in rows if c["verdict"] == "loss"]
            text = (("pomagają przy n ∈ {" + ", ".join(_n(n) for n in helps) + "}" if helps else "")
                    + ("; " if helps and hurts else "")
                    + ("SZKODZĄ (transfer negatywny) przy n ∈ {" + ", ".join(_n(n) for n in hurts) + "}" if hurts else ""))
            lines.append(f"- {scenario} · obce oddziały w kontekście {_arm_label(own)} (vs sam lokalny kontekst): "
                         + (text or "bez istotnego wpływu przy żadnym n") + ".")
    return lines or ["- brak porównań TFM (TFM pominięte albo brak gotowych jednostek)."]


def methods_paragraph(results: Mapping[str, Any]) -> str:
    cfg = results.get("config") or {}
    sizes = ", ".join(_n(n) for n in cfg.get("sizes") or [])
    repeats = ", ".join(str(r) for r in cfg.get("repeats") or [])
    mid = _mid("log_loss")
    mid_text = f"{mid:g}".replace(".", ",")                  # zapis polski (przecinek dziesiętny)
    eval_rows = cfg.get("eval_rows")
    eval_text = "całym zbiorze walidacyjnym" if not eval_rows else f"stałej, warstwowej próbie {_n(eval_rows)} wizyt zbioru walidacyjnego"
    local_eval = cfg.get("local_eval_rows")
    local_text = ("wszystkich wizytach walidacyjnych tego oddziału" if not local_eval
                  else f"warstwowej próbie {_n(local_eval)} wizyt walidacyjnych tego oddziału")
    tfm = ", ".join(TFM_DISPLAY.get(m, m) for m in cfg.get("models") or []) or "(brak — pominięte)"
    return (
        "Eksperyment A (mała liczba lokalnych etykiet). Sprawdzono, czy wstępnie wytrenowane tabularne modele "
        f"fundamentalne ({tfm}) użyte „z pudełka” przewyższają gradientowy boosting LightGBM, gdy szpital dysponuje "
        "niewielką liczbą oznaczonych wizyt. W scenariuszu „pooled” z wizyt zbioru uczącego wszystkich oddziałów "
        f"losowano podpróby o liczności n ∈ {{{sizes}}} (warstwowo względem ESI × oddział; liczba powtórzeń R dla "
        f"kolejnych n: {repeats}), a ocenę prowadzono na {eval_text}. W scenariuszu „site_local” podpróby "
        f"pochodziły wyłącznie z oddziału {cfg.get('local_site')} (warstwowo względem ESI), a ocenę prowadzono na "
        f"{local_text}. Każdą podpróbę losowano raz i używano jej dla wszystkich modeli (układ sparowany); w każdej "
        f"zagwarantowano co najmniej {MIN_PER_CLASS} wizyty z każdej klasy ESI. LightGBM strojono wyłącznie na "
        f"podpróbie (Optuna z samplerem TPE, liczba prób: {cfg.get('lgbm_trials')}; wewnętrzna warstwowa K-krotna walidacja "
        "krzyżowa, K = max(2, min(5, liczność najrzadszej klasy)); funkcja celu: log loss; liczba drzew równa "
        "średniej najlepszej iteracji wczesnego zatrzymania w foldach); jako odniesienie podano też LightGBM "
        "z domyślnymi ustawieniami biblioteki. Modele fundamentalne zastosowano bez strojenia, z domyślnymi "
        "ustawieniami bibliotek (kontekst = cała podpróba). W scenariuszu lokalnym sprawdzono ponadto, czy "
        f"dołączenie do kontekstu {_n(cfg.get('foreign_rows'))} wizyt z innych oddziałów poprawia wynik, czy go "
        "pogarsza (transfer negatywny), oraz model LightGBM wytrenowany na innych oddziałach i przeniesiony bez zmian, także "
        "z korektą znanego lokalnego rozkładu ESI (Saerens i in., 2002). Pierwszorzędową, ustaloną przed analizą "
        "miarą była wieloklasowa strata logarytmiczna (log loss); drugorzędowe: wynik Briera, ECE (ogółem oraz "
        "dla ESI 1 i ESI 5), makro-AUC, ważona kwadratowo kappa (QWK) i odsetek niedotriażowania dla reguły "
        "argmax i reguły kosztowej. Dla każdego n i modelu obliczono sparowane różnice Δ względem strojonego "
        "LightGBM w każdym powtórzeniu, ich średnią z 95% przedziałem t (df = R − 1) oraz dwupoziomowy bootstrap "
        f"({cfg.get('n_boot')} losowań: powtórzenia ze zwracaniem, a następnie wizyty oceniane warstwowo względem "
        "ESI). Do wnioskowania przyjęto wartość p równą większej z wartości p bootstrapu i sparowanego testu t "
        "(różnicę uznawano za istotną tylko wtedy, gdy oba przedziały wykluczały zero; przy jednym powtórzeniu "
        "testu nie przeprowadzano), ponieważ przy kilku powtórzeniach percentylowy bootstrap po powtórzeniach "
        "zaniża niepewność. Wartości p miary pierwszorzędowej skorygowano metodą Holma względem liczby rozmiarów "
        f"n. Przewagę modelu ogłaszano wyłącznie, gdy skorygowane p < 0,05 i |Δ| > {mid_text} (minimalna istotna "
        "różnica). Przebieg log loss w funkcji n opisano krzywą a + b·n^(−c), a punkt przecięcia krzywych n* "
        "wyznaczono wraz z 95% przedziałem bootstrap po powtórzeniach (orientacyjnym, gdy dla któregoś rozmiaru "
        f"wykonano mniej niż {MIN_REPEATS_NSTAR_CI} powtórzeń). Czułość dla ESI 1 raportowano wyłącznie dla "
        f"n ≥ {_n(RECALL_MIN_N)}, z dokładnym przedziałem Clopper–Pearsona."
    )


def report_lines(results: Mapping[str, Any]) -> list[str]:
    """Pełny raport report.md (po polsku)."""
    cfg = results.get("config") or {}
    mid = _mid("log_loss")
    lines = [f"# Eksperyment A — modele TFM przy małej liczbie danych · {results['feature_set']} · tag {results['tag']}",
             "", f"_Wygenerowano: {results.get('created')}; odcisk danych {results.get('data_fingerprint')}_", ""]
    warnings_top = []
    for model, status in (results.get("tfm_status") or {}).items():
        if status.get("fake"):
            warnings_top.append(f"**{TFM_DISPLAY.get(model, model)}: ATRAPA estymatora (SOR_AI_TFM_FAKE=1)** — wyniki "
                                f"TFM bezwartościowe, tylko test mechaniki.")
        elif status.get("override"):
            warnings_top.append(f"**{TFM_DISPLAY.get(model, model)}: PRZEKIEROWANIE WAG ({status['override']})** — "
                                f"jeśli to wagi testowe (losowe), wyniki TFM są bezwartościowe.")
    for model, entries in (results.get("tfm_provenance") or {}).items():
        name = TFM_DISPLAY.get(model, model)
        if any(e.get("fake") for e in entries) and not (results.get("tfm_status") or {}).get(model, {}).get("fake"):
            warnings_top.append(f"**{name}: część jednostek policzona ATRAPĄ estymatora** — wyniki TFM bezwartościowe.")
        overrides = sorted({str(e.get("override")) for e in entries if e.get("override") and not e.get("fake")})
        if overrides and not (results.get("tfm_status") or {}).get(model, {}).get("override"):
            warnings_top.append(f"**{name}: jednostki z PRZEKIEROWANIEM WAG ({', '.join(overrides)})**.")
        if len({json.dumps(clean_json(e.get("weights")), sort_keys=True) for e in entries}) > 1:
            warnings_top.append(f"**{name}: jednostki policzone RÓŻNYMI wagami** — zob. results.json `tfm_provenance`.")
    if warnings_top:
        lines += ["> ⚠ " + w for w in warnings_top] + [""]
    lines += ["## Kryteria ustalone przed analizą (pre-rejestracja)", "",
              "- metryka pierwszorzędowa: **log loss** (wiersze oceny); odniesienie: **LightGBM strojony na podpróbie**",
              "- Δ = model − odniesienie w każdym powtórzeniu (układ sparowany); średnia z 95% CI t (df = R − 1)",
              f"- dwupoziomowy bootstrap: {cfg.get('n_boot')} losowań (powtórzenia ze zwracaniem → wiersze oceny "
              "warstwowo wg ESI); p z tych samych ogonów co przedział",
              "- p werdyktu = **max(p bootstrapu, p testu t)** (R ≥ 2; R = 1 → bez werdyktu) — wygrana wymaga, by "
              "przedział t (df = R − 1) **i** przedział bootstrapu wykluczały 0 (przy 2–5 powtórzeniach sam "
              "bootstrap jest zbyt wąski)",
              f"- korekta Holma między rozmiarami n (każda para model–odniesienie osobno); **wygrana** tylko gdy "
              f"p_Holm < 0.05 **i** |Δ| > MID = {mid:g}; metryki drugorzędowe bez korekty (eksploracyjnie; MID: "
              "QWK 0.01, niedotriażowanie 0.01; Brier / ECE bez MID)",
              f"- czułość ESI 1 raportowana tylko dla n ≥ {_n(RECALL_MIN_N)} (przedział Clopper–Pearsona)",
              f"- krzywa a + b·n^(−c) (≥ 3 rozmiary; przy 3 c = 0.5) i przecięcie n* (szukane do "
              f"{CROSSOVER_FACTOR:g}× największego n); 95% CI n* z bootstrapu po powtórzeniach — orientacyjny, gdy "
              f"któryś rozmiar ma < {MIN_REPEATS_NSTAR_CI} powtórzeń", ""]
    lines += ["## Ustawienia", "",
              f"- scenariusze: {', '.join(cfg.get('scenarios') or [])}; rozmiary n: {', '.join(_n(n) for n in cfg.get('sizes') or [])}; "
              f"powtórzenia: {', '.join(str(r) for r in cfg.get('repeats') or [])}; seed {cfg.get('seed')}",
              f"- ocena pooled: {('cały VAL' if not cfg.get('eval_rows') else _n(cfg.get('eval_rows')) + ' wierszy VAL (warstwy ESI × oddział)')}; "
              f"site_local (oddział {cfg.get('local_site')}): "
              f"{('wszystkie wiersze VAL oddziału' if not cfg.get('local_eval_rows') else _n(cfg.get('local_eval_rows')) + ' wierszy VAL oddziału')}",
              f"- LightGBM strojony: liczba prób Optuny na jednostkę = {cfg.get('lgbm_trials')}; obce wiersze w kontekście: "
              f"{_n(cfg.get('foreign_rows'))}; model przeniesiony: ≤ {_n(cfg.get('transport_rows'))} wierszy innych oddziałów",
              f"- jednostki: {results.get('n_done')}/{results.get('n_units')} gotowe"
              + (f", błędy: {len(results.get('failures') or [])}" if results.get("failures") else ""), ""]
    lines += ["| model TFM | dostępny | urządzenie | biblioteka | wagi | powód pominięcia |", "|---|---|---|---|---|---|"]
    for model in cfg.get("models_requested") or []:
        st = (results.get("tfm_status") or {}).get(model) or {}
        weights = st.get("weights") or {}
        sha = str(weights.get("sha256") or "")[:12]
        lines.append(f"| {TFM_DISPLAY.get(model, model)} | {'tak' if st.get('available') else 'NIE'} | "
                     f"{st.get('device') or '—'} | {st.get('library_version') or '—'} | "
                     f"{weights.get('file') or '—'}{' (' + sha + ')' if sha else ''} | {st.get('reason') or '—'} |")
    if not cfg.get("models_requested"):
        lines.append("| — | — | — | — | — | nie zażądano TFM (--models none) |")
    lines.append("")
    if results.get("skipped_units"):
        reasons: dict[str, int] = {}
        for s in results["skipped_units"]:
            reasons[s["reason"]] = reasons.get(s["reason"], 0) + 1
        lines += ["Pominięte jednostki: " + "; ".join(f"{k} ({v}×)" for k, v in reasons.items()), ""]

    lines += ["## Decyzja", ""] + decision_lines(results) + [""]

    for scenario, block in (results.get("scenarios") or {}).items():
        ev = block.get("eval") or {}
        lines += [f"## Scenariusz {SCENARIO_TITLES.get(scenario, scenario)}", "",
                  f"Zbiór oceny: {_n(ev.get('n'))} wizyt VAL, w tym {_n(ev.get('n_esi1'))} ESI 1; "
                  f"źródło podprób: {_n(block.get('source_rows'))} wierszy TRAIN.", ""]
        if scenario == "site_local" and ev.get("n_esi1") is not None and ev["n_esi1"] < SMALL_COUNT_ESI1:
            lines += [f"> ⚠ Tylko {ev['n_esi1']} wizyt ESI 1 w zbiorze oceny — metryki ESI 1 (czułość, krytyczne "
                      "pominięcia, ECE ESI1) są bardzo niepewne; przedziały Clopper–Pearsona poniżej, wnioski "
                      "o ESI 1 wyłącznie orientacyjne.", ""]
        lines += ["### Metryki (średnia ± SD po powtórzeniach)", "",
                  "| model | n | R | " + " | ".join(TABLE_METRICS.values()) + " |",
                  "|---|---:|---:|" + "---:|" * len(TABLE_METRICS)]
        for arm, by_n in (block.get("points") or {}).items():
            for n in sorted(int(k) for k in by_n):
                entry = by_n[str(n)]
                cells = []
                for key in TABLE_METRICS:
                    s = entry["metrics"].get(key) or {}
                    if key.startswith("recall_esi1") and n < RECALL_MIN_N:
                        cells.append("— (n < 5000)")
                    elif s.get("mean") is None:
                        cells.append("—")
                    else:
                        cells.append(_f(s["mean"]) + (f" ± {_f(s['sd'])}" if s.get("sd") is not None else ""))
                lines.append(f"| {_arm_label(arm)} | {_n(n)} | {entry['R']} | " + " | ".join(cells) + " |")
        transported = block.get("transported")
        if transported:
            for variant, metrics in (transported.get("variants") or {}).items():
                cells = [_f(metrics.get(k)) for k in TABLE_METRICS]
                label = f"{_arm_label(TRANSPORTED)} ({'surowy' if variant == 'raw' else 'korekta priorów'})"
                lines.append(f"| {label} | {_n(transported.get('n_fit'))} | 1 | " + " | ".join(cells) + " |")
        lines.append("")
        if transported:
            lines += [f"Model przeniesiony: {transported.get('params_note')}; uczony na {_n(transported.get('n_fit'))} "
                      f"wierszach innych oddziałów (+ {_n(transported.get('n_es'))} na early stopping); korekta priorów: "
                      f"P·π_lokalny/π_trening (π_lokalny = rozkład ESI w TRAIN oddziału — „znany” z rejestru).", ""]
        lines += ["### Δ log loss vs LightGBM strojony (metryka pierwszorzędowa)", "",
                  "| model | odniesienie | n | R | Δ | SD | 95% CI t | 95% CI bootstrap 2-poz. | p (t) | p (bootstrap) "
                  "| p = max | p Holm | werdykt |",
                  "|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|"]
        prim = [c for c in block.get("comparisons") or [] if c["metric"] == PRIMARY_METRIC]
        for c in sorted(prim, key=lambda c: (c["base"] != BASELINE, c["arm"], c["n"])):
            lines.append(f"| {_arm_label(c['arm'])} | {_arm_label(c['base'])} | {_n(c['n'])} | {c['R']} | "
                         f"{_f(c['delta'])} | {_f(c['sd'])} | {_ci(c['t_ci_low'], c['t_ci_high'])} | "
                         f"{_ci(c['boot_ci_low'], c['boot_ci_high'])} | {_p(c.get('p_t'))} | {_p(c['p_boot'])} | "
                         f"{_p(c.get('p'))} | {_p(c.get('p_holm'))} | {c['verdict_text']} |")
        lines += ["", "### Metryki drugorzędowe — Δ vs LightGBM strojony (95% CI bootstrap 2-poz.; bez korekty)", "",
                  "| model | n | Δ QWK argmax | Δ QWK koszt | Δ undertriage argmax | Δ undertriage koszt | Δ Brier | Δ ECE |",
                  "|---|---:|---|---|---|---|---|---|"]
        second = {(c["arm"], c["n"], c["metric"]): c for c in block.get("comparisons") or [] if c["base"] == BASELINE}
        for arm, n in sorted({(c["arm"], c["n"]) for c in prim if c["base"] == BASELINE}):
            cells = []
            for key in ("qwk_argmax", "qwk_cost", "undertriage_argmax", "undertriage_cost", "brier", "ece"):
                c = second.get((arm, n, key))
                if not c:
                    cells.append("—")
                    continue
                mark = {"win": " ✔", "loss": " ✘"}.get(c["verdict"], "")
                cells.append(f"{_f(c['delta'])} {_ci(c['boot_ci_low'], c['boot_ci_high'])}{mark}")
            lines.append(f"| {_arm_label(arm)} | {_n(n)} | " + " | ".join(cells) + " |")
        lines += ["", "✔ = model istotnie lepszy (i |Δ| > MID, gdy MID określone), ✘ = istotnie gorszy.", "",
                  "### Krzywe a + b·n^(−c) (log loss) i przecięcie z LightGBM strojonym", "",
                  "| model | a | b | c | RMSE | wiarygodne | n* | 95% CI n* | wynik |", "|---|---:|---:|---:|---:|---|---:|---|---|"]
        for arm, fit in (block.get("power_law") or {}).items():
            cross = (block.get("crossover") or {}).get(f"{arm}|{BASELINE}") or {}
            if fit is None:
                lines.append(f"| {_arm_label(arm)} | — | — | — | — | — | — | — | za mało rozmiarów n (< 3) |")
                continue
            reasons = "tak" if fit.get("reliable") else "nie (" + "; ".join(fit.get("reasons") or []) + ")"
            ci = (_ci(cross.get("ci_low"), cross.get("ci_high"), 0)
                  + (" orientacyjny" if cross.get("ci_indicative") else "")
                  if cross.get("ci_low") is not None else (cross.get("ci_note") or "—"))
            outcome = _STATUS_TEXT.get(cross.get("status"), "—") if cross else "odniesienie"
            if cross.get("n_star") is not None and cross.get("in_range") is False:
                outcome += " — ekstrapolacja"
            if cross.get("agreement") is not None:
                outcome += f" (zgodność bootstrapu {100 * cross['agreement']:.0f}%)"
            lines.append(f"| {_arm_label(arm)} | {_f(fit['a'])} | {_f(fit['b'])} | {_f(fit['c'], 3)} | {_f(fit['rmse'])} | "
                         f"{reasons} | {_n(cross.get('n_star'))} | {ci} | {outcome} |")
        lines += ["", f"### ESI 1 (n ≥ {_n(RECALL_MIN_N)}; średnia po powtórzeniach, 95% CI Clopper–Pearsona dla "
                  f"średniej liczby trafień z {_n(ev.get('n_esi1'))} wizyt ESI 1)", "",
                  "| model | n | R | czułość ESI1 argmax | czułość ESI1 koszt | krytyczne pominięcia ESI1 (koszt) |",
                  "|---|---:|---:|---|---|---|"]
        esi_rows = _esi1_rows(scenario, block, int(ev.get("n_esi1") or 0))
        lines += esi_rows or [f"| — | brak rozmiarów n ≥ {_n(RECALL_MIN_N)} | | | | |"]
        lines.append("")
        for fig in block.get("figures") or []:
            lines.append(f"![{scenario}]({fig})")
        lines.append("")
    lines += ["## Akapit metod (do pracy)", "", methods_paragraph(results), "",
              "## Zastrzeżenia", "",
              "- Deklaracje producentów (TabPFN, TabICL: przewaga nad strojonym boostingiem do ~10–50 tys. wierszy) "
              "pochodzą z ich własnych benchmarków na innych danych — ten eksperyment sprawdza je na danych SOR; "
              "wynik dotyczy tych danych i tych wersji wag.",
              "- Rzadkie klasy: ESI 1 ≈ 1% wizyt. Przy małym n podpróba ma kilka wizyt ESI 1 (gwarancja ≥ 2 lekko je "
              "nadreprezentuje przy najmniejszych n); metryki ESI 1 przy małym n i w VAL jednego oddziału "
              "(kilkadziesiąt wizyt ESI 1) są bardzo niepewne.",
              "- Mało powtórzeń przy dużych n: sam percentylowy bootstrap po R = 2–5 powtórzeniach zaniżałby niepewność "
              "(fałszywe „wygrane”), dlatego werdykt używa max(p bootstrapu, p testu t) — kosztem mocy; przedział n* "
              "przy R < 5 jest tylko orientacyjny.",
              "- Zbiór oceny jest stały (VAL); TEST nie był używany. LightGBM strojony wewnętrzną CV na tej samej "
              "podpróbie (bez dostępu do VAL) — TFM bez strojenia: porównanie „z pudełka” vs „strojony” jest zamierzone.",
              "- Kontekst TFM większy od zakresu pretreningu (TabICL: ~48 tys. wierszy; n = 50 000 + obce wiersze) "
              "może pogarszać wynik; GPU (autocast fp16) i CPU dają nieco inne prawdopodobieństwa.",
              "- Ekstrapolacja n* poza zakres n jest tylko orientacyjna (oznaczona w tabeli).", "",
              "## Pliki", "",
              "- results.json — wszystkie liczby; units.csv — metryki jednostek; units/*.json — jednostki (wznawianie);",
              "- preds/*.npy — prawdopodobieństwa wierszy oceny (kolejność = samples/<scenariusz>_eval.npy);",
              "- samples/*.npy — row_id podprób (wspólne dla modeli); figures/*.png — wykresy.", ""]
    return lines


def summary_lines(results: Mapping[str, Any]) -> list[str]:
    """Krótkie podsumowanie summary.md (orkiestrator)."""
    lines = [f"# Eksperyment A (TFM, mało danych) — {results['feature_set']} · tag {results['tag']}", "",
             f"- jednostki {results.get('n_done')}/{results.get('n_units')}"
             + (f", błędy: {len(results.get('failures') or [])}" if results.get("failures") else "")]
    for model, reason in (results.get("skipped_models") or {}).items():
        lines.append(f"- pominięto {TFM_DISPLAY.get(model, model)}: {reason}")
    lines += ["", "## Decyzja (log loss, Holm + MID)", ""] + decision_lines(results)
    lines += ["", "Pełny raport: report.md"]
    return lines


def units_table(store: UnitStore, units: Sequence[Mapping[str, Any]], expect: Callable[[Mapping[str, Any]], dict]) -> Any:
    """Płaska tabela jednostek (units.csv): klucze + metryki."""
    import pandas as pd

    rows = []
    for u in units:
        payload = store.valid(str(u["unit"]), expect(u))
        base = {"unit": u["unit"], "scenario": u["scenario"], "arm": u["arm"], "n": u.get("n"),
                "repeat": u.get("repeat"), "context": u.get("context"), "status": "done" if payload else "pending"}
        if payload:
            base["duration_s"] = payload.get("duration_s")
            if u["arm"] == TRANSPORTED:
                for variant, metrics in (payload.get("variants") or {}).items():
                    rows.append({**base, "variant": variant,
                                 **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}})
                continue
            base.update({k: v for k, v in (payload.get("metrics") or {}).items() if not isinstance(v, (dict, list))})
        rows.append(base)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# Przebieg
# ─────────────────────────────────────────
@contextlib.contextmanager
def tfm_group_session(model: str, fake: bool) -> Iterator[None]:
    """
    Grupa jednostek jednego TFM: TabPFN — TABPFN_MODEL_CACHE_SIZE=1 na całą grupę (wagi ładowane raz),
    na końcu wyczyszczenie pamięci podręcznej modelu i zwolnienie pamięci GPU (przed TabICL).
    """
    from src.models.tfm_base import free_device_memory

    set_here = model == "tabpfn" and not fake and "TABPFN_MODEL_CACHE_SIZE" not in os.environ
    if set_here:
        os.environ["TABPFN_MODEL_CACHE_SIZE"] = "1"
    try:
        yield
    finally:
        if set_here:
            os.environ.pop("TABPFN_MODEL_CACHE_SIZE", None)
        if model == "tabpfn" and not fake:
            with contextlib.suppress(Exception):
                from tabpfn import model_loading

                model_loading.clear_built_model_cache()
                cached = getattr(model_loading, "_load_checkpoint_cached", None)
                if cached is not None and hasattr(cached, "cache_clear"):
                    cached.cache_clear()
        free_device_memory()


def plan_lines(cfg: SmallDataConfig, units: Sequence[Mapping[str, Any]], skipped: Sequence[Mapping[str, str]],
               statuses: Mapping[str, Mapping[str, Any]], counts: SourceCounts, root: Path,
               benchmark: Path) -> list[str]:
    """Plan (--plan): liczba jednostek na scenariusz i model + orientacyjny czas."""
    rates = _benchmark_rates(benchmark)
    eval_n = {"pooled": counts.n_val if not cfg.eval_rows else min(cfg.eval_rows, counts.n_val),
              "site_local": counts.local_val if not cfg.local_eval_rows else min(cfg.local_eval_rows, counts.local_val)}
    # liczba ESI 1 jest znana bez wczytywania cech tylko dla CAŁEGO VAL oddziału — przy próbie podajemy ją jako tło
    local_esi1 = (f"{counts.local_esi1_val} ESI 1" if eval_n["site_local"] >= counts.local_val else
                  f"próba warstwowa; cały VAL oddziału: {counts.local_val:,} wierszy, {counts.local_esi1_val} ESI 1")
    lines = [f"Plan eksperymentu A → {root}",
             f"  scenariusze: {', '.join(cfg.scenarios)}; n = {cfg.sizes}; powtórzenia = {cfg.repeats}; seed {cfg.seed}",
             f"  ocena: pooled {eval_n['pooled']:,} wierszy VAL; site_local (oddział {cfg.local_site}) "
             f"{eval_n['site_local']:,} wierszy VAL ({local_esi1})",
             f"  szacunek czasu: {'benchmark ' + str(benchmark) if rates else 'bez pomiaru (plan pamięci, GPU; orientacyjnie)'}"]
    for model in cfg.models:
        st = statuses.get(model) or {}
        lines.append(f"  {TFM_DISPLAY.get(model, model)}: " + (f"urządzenie {st.get('device')}, wagi "
                     f"{(st.get('weights') or {}).get('file')}" if st.get("available") else f"POMINIĘTY — {st.get('reason')}"))
    table: dict[tuple[str, str], list[float | None]] = {}
    for u in units:
        device = (statuses.get(u["base_model"]) or {}).get("device")
        est = estimate_unit_s(u, eval_n[u["scenario"]], cfg.lgbm_trials, rates, device)
        table.setdefault((u["scenario"], u["arm"]), []).append(est)
    total = 0.0
    unknown = False
    lines.append(f"  {'scenariusz':<11} {'model':<22} {'jednostek':>9} {'czas':>12}")
    for (scenario, arm), ests in table.items():
        known = [e for e in ests if e is not None]
        unknown |= len(known) < len(ests)
        total += sum(known)
        text = format_duration(sum(known)) if known else "?"
        lines.append(f"  {scenario:<11} {arm:<22} {len(ests):>9} {text:>12}" + (" (+ ?)" if len(known) < len(ests) and known else ""))
    lines.append(f"  RAZEM: {len(units)} jednostek, ≈ {format_duration(total)}" + (" + jednostki bez szacunku (CPU)" if unknown else ""))
    if skipped:
        lines.append(f"  pominięte: {len(skipped)} — " + "; ".join(sorted({s['reason'] for s in skipped}))[:600])
    return lines


def run_small_data(cfg: SmallDataConfig, plan_only: bool = False) -> dict[str, Any]:
    """
    Eksperyment A (wznawialny) → results.json, units.csv, report.md, summary.md, figures/. plan_only=True →
    tylko plan (bez danych cech, bez zapisu) w `results['plan_lines']`.
    """
    from src.data.dataset import data_fingerprint

    cfg = validate_config(cfg)
    tag = cfg.resolved_tag()
    root = analysis_path(cfg.feature_set, tag)
    benchmark = analyses_root() / "tfm_setup" / "benchmark.json"
    statuses = {m: tfm_status(m, cfg.device) for m in cfg.models}
    for model, st in statuses.items():
        if not st["available"]:
            log.warning(f"{TFM_DISPLAY.get(model, model)} POMINIĘTY w eksperymencie A: {st['reason']} "
                        f"(części LightGBM liczone normalnie)")
    source_run = lightgbm_source(cfg.feature_set, tag) if "site_local" in cfg.scenarios and cfg.transport_rows else None
    fingerprint = data_fingerprint()

    if plan_only:
        counts = light_counts(cfg.local_site)
        if "site_local" in cfg.scenarios and counts.local_train == 0:
            raise AnalysisArgumentError(f"Oddziału '{cfg.local_site}' nie ma w TRAIN (dostępne: "
                                        f"{', '.join(counts.sites_train)})")
        units, skipped = build_units(cfg, counts, statuses, source_run.name if source_run else None)
        lines = plan_lines(cfg, units, skipped, statuses, counts, root, benchmark)
        return {"plan_lines": lines, "n_units": len(units), "skipped_units": skipped,
                "skipped_models": {m: st["reason"] for m, st in statuses.items() if not st["available"]}}

    log.info(f"Eksperyment A: wczytuję TRAIN i VAL (zestaw {cfg.feature_set})")
    data = SplitData(cfg.feature_set)
    counts = SourceCounts.from_arrays(data.site["train"], data.site["val"], data.y["val"], cfg.local_site)
    if "site_local" in cfg.scenarios and counts.local_train == 0:
        raise AnalysisArgumentError(f"Oddziału '{cfg.local_site}' nie ma w TRAIN (dostępne: "
                                    f"{', '.join(counts.sites_train)})")
    units, skipped = build_units(cfg, counts, statuses, source_run.name if source_run else None)

    # ── próby (raz; wspólne dla modeli) ──
    bank = SampleBank(root, {"seed": cfg.seed, "data_fingerprint": fingerprint, "local_site": cfg.local_site,
                             "feature_set": cfg.feature_set})
    y_tr, y_va = data.y["train"], data.y["val"]
    strata_tr, strata_va = data.strata("train"), data.strata("val")
    local_tr = np.flatnonzero(data.site["train"] == cfg.local_site)
    foreign_tr = np.flatnonzero(data.site["train"] != cfg.local_site)
    local_va = np.flatnonzero(data.site["val"] == cfg.local_site)
    all_tr, all_va = np.arange(len(y_tr)), np.arange(len(y_va))
    eval_pos: dict[str, np.ndarray] = {}
    if "pooled" in cfg.scenarios:
        m = len(all_va) if not cfg.eval_rows else min(cfg.eval_rows, len(all_va))
        eval_pos["pooled"] = bank.get("pooled_eval", data.row_id["val"], lambda m=m: draw_sample(
            all_va, strata_va, y_va, m, derive_seed(cfg.seed, "pooled", "eval"), 0), m)
    if "site_local" in cfg.scenarios:
        m = len(local_va) if not cfg.local_eval_rows else min(cfg.local_eval_rows, len(local_va))
        eval_pos["site_local"] = bank.get("site_local_eval", data.row_id["val"], lambda m=m: draw_sample(
            local_va, y_va[local_va], y_va, m, derive_seed(cfg.seed, "site_local", "eval"), 0), m)
    ctx: dict[tuple[str, int, int], np.ndarray] = {}
    foreign: dict[tuple[int, int], np.ndarray] = {}
    foreign_n = min(cfg.foreign_rows, len(foreign_tr))
    for u in units:
        if u["n"] is None:
            continue
        key = (u["scenario"], int(u["n"]), int(u["repeat"]))
        if key not in ctx:
            scenario, n, r = key
            pool, labels = (all_tr, strata_tr) if scenario == "pooled" else (local_tr, y_tr[local_tr].astype(str))
            ctx[key] = bank.get(f"{scenario}_n{n}_r{r}", data.row_id["train"], lambda p=pool, lab=labels, n=n, s=scenario, r=r:
                                draw_sample(p, lab, y_tr, n, derive_seed(cfg.seed, s, n, r, "sample")), n)
        if u["foreign"] and (key[1], key[2]) not in foreign:
            n, r = key[1], key[2]
            foreign[(n, r)] = bank.get(f"site_local_n{n}_r{r}_foreign", data.row_id["train"],
                                       lambda n=n, r=r: draw_sample(foreign_tr, strata_tr[foreign_tr], y_tr, foreign_n,
                                                                    derive_seed(cfg.seed, "foreign", n, r)), foreign_n)
    transport_pos = None
    if any(u["arm"] == TRANSPORTED for u in units):
        m = min(cfg.transport_rows, len(foreign_tr))
        transport_pos = bank.get("site_local_transport", data.row_id["train"], lambda m=m: draw_sample(
            foreign_tr, strata_tr[foreign_tr], y_tr, m, derive_seed(cfg.seed, "transport")), m)
    bank.write_manifest()
    log.info(f"Próby: {bank.reused} użyte ponownie, {bank.drawn} wylosowane → {bank.dir}")

    def sample_key(u: Mapping[str, Any]) -> str:
        parts = [ids_hash(data.row_id["val"][eval_pos[u["scenario"]]])]
        if u["n"] is None:
            parts.append(ids_hash(data.row_id["train"][transport_pos]))
        else:
            parts.append(ids_hash(data.row_id["train"][ctx[(u["scenario"], int(u["n"]), int(u["repeat"]))]]))
            if u["foreign"]:
                parts.append(ids_hash(data.row_id["train"][foreign[(int(u["n"]), int(u["repeat"]))]]))
        return ":".join(parts)

    def arm_config(u: Mapping[str, Any]) -> dict[str, Any]:
        arm, base = str(u["arm"]), str(u["base_model"])
        if arm == "lightgbm_tuned":
            return {"trials": int(cfg.lgbm_trials), "space": LGBM_SPACE_VERSION, "max_rounds": LGBM_MAX_ROUNDS,
                    "es_rounds": LGBM_ES_ROUNDS}
        if arm == "lightgbm_default":
            return {"params": LGBM_LIBRARY_DEFAULTS}
        if arm == TRANSPORTED:
            return {"transport_rows": int(cfg.transport_rows)}
        st = statuses[base]
        return {"params": TFM_OUT_OF_BOX[base], "weights": st.get("weights"), "library_version": st.get("library_version"),
                "fake": st.get("fake"), "device": st.get("device"), "foreign_rows": int(foreign_n) if u["foreign"] else 0}

    expect_cache: dict[str, dict[str, Any]] = {}

    def expect(u: Mapping[str, Any]) -> dict[str, Any]:
        name = str(u["unit"])
        if name not in expect_cache:
            expect_cache[name] = {"source_run": u["source_run"], "data_fingerprint": fingerprint,
                                  "version": SMALL_DATA_VERSION, "samples": sample_key(u), "config": arm_config(u)}
        return expect_cache[name]

    store = UnitStore(root)
    # Wyniki kumulatywne: gotowe jednostki TFM z poprzedniego planu, których to wywołanie nie liczy (model
    # niezażądany / niedostępny tutaj, kontekst za duży na CPU) — o ile te same próby i dane; dla modelu
    # liczonego teraz także te same wagi (bez mieszania np. wag testowych z prawdziwymi)
    planned_names = {str(u["unit"]) for u in units}
    kept: list[dict[str, Any]] = []
    for old in store.planned_units():
        base = str(old.get("base_model") or "")
        if str(old["unit"]) in planned_names or base not in config.TFM_MODELS:
            continue
        key = (old.get("scenario"), old.get("n"), old.get("repeat"))
        if old.get("n") is None or (key[0], int(key[1]), int(key[2])) not in ctx:
            continue
        if old.get("foreign") and (int(key[1]), int(key[2])) not in foreign:
            continue
        payload = store.load(str(old["unit"])) or {}
        old_cfg = payload.get("config") or {}
        current = statuses.get(base) or {}
        same_weights = not current.get("available") or all(
            clean_json(old_cfg.get(k)) == clean_json(current.get(k)) for k in ("weights", "library_version", "fake"))
        if (same_weights and payload.get("status") == "done" and payload.get("data_fingerprint") == fingerprint
                and payload.get("version") == SMALL_DATA_VERSION and payload.get("samples") == sample_key(old)
                and (root / str(payload.get("preds", ""))).exists()):
            expect_cache[str(old["unit"])] = {k: payload.get(k) for k in ("source_run", "data_fingerprint", "version",
                                                                          "samples", "config")}
            kept.append(dict(old))
    if kept:
        log.info(f"Wyniki kumulatywne: {len(kept)} gotowych jednostek TFM z poprzedniego wywołania")
    all_units = units + kept
    store.write_plan({"analysis": ANALYSIS, "feature_set": cfg.feature_set, "tag": tag, "config": asdict(cfg),
                      "units": [{k: u[k] for k in ("unit", "model", "arm", "base_model", "scenario", "n", "repeat",
                                                   "foreign", "context", "source_run")} for u in all_units],
                      "skipped": skipped})

    # Jednostka „gotowa”, ale bez pliku predykcji → licz od nowa
    for u in units:
        payload = store.valid(str(u["unit"]), expect(u))
        if payload is not None and not all((root / p).exists() for p in _payload_preds(payload)):
            with contextlib.suppress(OSError):
                store.path(str(u["unit"])).unlink()

    local_prior = None
    if "site_local" in cfg.scenarios:
        from src.analysis.priors import class_prior

        local_prior = class_prior(y_tr[local_tr])

    def compute(u: Mapping[str, Any]) -> dict[str, Any]:
        scenario, arm, base = str(u["scenario"]), str(u["arm"]), str(u["base_model"])
        ev = eval_pos[scenario]
        X_eval = data.X["val"].iloc[ev]
        y_eval = y_va[ev]
        vc = None if data.vc["val"] is None else data.vc["val"][ev]
        admitted = None if data.admitted["val"] is None else data.admitted["val"][ev]
        seed = int(u["seed"])
        if arm == TRANSPORTED:
            probas, info = fit_predict_transported(data, transport_pos, ev, local_prior, source_run, seed)
            out: dict[str, Any] = {"arm": arm, "scenario": scenario, "variants": {}, "preds_variants": {},
                                   "info": info, "n_fit": info["n_fit"]}
            for variant, p in probas.items():
                rel = preds_file(str(u["unit"]), variant)
                _save_npy(root / rel, p.astype(np.float32))
                out["variants"][variant] = unit_metrics(y_eval, p, vc, admitted)
                out["preds_variants"][variant] = rel
            return out
        pos = ctx[(scenario, int(u["n"]), int(u["repeat"]))]
        if u["foreign"]:
            pos = np.concatenate([pos, foreign[(int(u["n"]), int(u["repeat"]))]])
        X_ctx = data.X["train"].iloc[pos]
        y_ctx = y_tr[pos]
        info: dict[str, Any] = {}
        if arm == "lightgbm_tuned":
            params, info = tune_lightgbm_small(X_ctx, y_ctx, n_trials=cfg.lgbm_trials, seed=seed)
            proba = fit_predict_lgbm(X_ctx, y_ctx, X_eval, params)
        elif arm == "lightgbm_default":
            proba = fit_predict_lgbm(X_ctx, y_ctx, X_eval, {**LGBM_LIBRARY_DEFAULTS, "random_state": seed,
                                                             "n_jobs": _lgbm_jobs()})
        else:
            proba, info = fit_predict_tfm(base, X_ctx, y_ctx, X_eval, device=statuses[base]["device"], seed=seed)
        rel = preds_file(str(u["unit"]))
        _save_npy(root / rel, proba.astype(np.float32))
        return {"arm": arm, "scenario": scenario, "n": int(u["n"]), "repeat": int(u["repeat"]),
                "n_context": int(len(pos)), "class_counts": np.bincount(y_ctx, minlength=config.N_CLASSES).tolist(),
                "metrics": unit_metrics(y_eval, proba, vc, admitted), "info": info, "preds": rel}

    rates = _benchmark_rates(benchmark)

    def estimate(u: Mapping[str, Any]) -> float | None:
        device = (statuses.get(u["base_model"]) or {}).get("device")
        return estimate_unit_s(u, len(eval_pos[u["scenario"]]), cfg.lgbm_trials, rates, device)

    failures: list[dict[str, str]] = []
    groups = list(dict.fromkeys(u["group"] for u in units))
    for group in groups:
        todo = [u for u in units if u["group"] == group]
        fake = bool((statuses.get(group) or {}).get("fake"))
        session = tfm_group_session(group, fake) if group in config.TFM_MODELS else contextlib.nullcontext()
        log.info(f"Eksperyment A — grupa {group}: {len(todo)} jednostek")
        with session:
            failures += execute_units(store, todo, expect, compute, f"{ANALYSIS}/{group}", force=cfg.force,
                                      estimate_s=estimate)

    results = assemble_results(cfg, tag, root, store, all_units, expect, skipped, statuses, failures, data, eval_pos,
                               counts, fingerprint)
    return results


def _payload_preds(payload: Mapping[str, Any]) -> list[str]:
    if payload.get("preds"):
        return [str(payload["preds"])]
    return [str(p) for p in (payload.get("preds_variants") or {}).values()]


def assemble_results(cfg: SmallDataConfig, tag: str, root: Path, store: UnitStore, units: Sequence[Mapping[str, Any]],
                     expect: Callable[[Mapping[str, Any]], dict], skipped: Sequence[Mapping[str, str]],
                     statuses: Mapping[str, Mapping[str, Any]], failures: Sequence[Mapping[str, str]], data: SplitData,
                     eval_pos: Mapping[str, np.ndarray], counts: SourceCounts, fingerprint: str) -> dict[str, Any]:
    """Statystyka, wykresy, results.json, units.csv, report.md, summary.md."""
    done_by_scenario: dict[str, list[dict[str, Any]]] = {}
    n_done = 0
    for u in units:
        payload = store.valid(str(u["unit"]), expect(u))
        if payload is None or not all((root / p).exists() for p in _payload_preds(payload)):
            continue
        n_done += 1
        done_by_scenario.setdefault(str(u["scenario"]), []).append({**dict(u), **payload})
    scenarios: dict[str, Any] = {}
    figures_dir = root / "figures"
    for scenario in cfg.scenarios:
        ev = eval_pos[scenario]
        y_eval = data.y["val"][ev]
        done = done_by_scenario.get(scenario, [])
        t0 = time.time()
        block = collect_statistics(root, scenario, done, y_eval, cfg.n_boot, cfg.seed)
        log.info(f"Statystyka {scenario}: {len(block['comparisons'])} porównań, {time.time() - t0:.1f} s")
        block["eval"] = {"n": int(len(ev)), "n_esi1": int((y_eval == 0).sum()),
                         "class_counts": np.bincount(y_eval, minlength=config.N_CLASSES).tolist()}
        block["source_rows"] = counts.n_train if scenario == "pooled" else counts.local_train
        trans = next((u for u in done if u["arm"] == TRANSPORTED), None)
        if trans is not None:
            block["transported"] = {
                "variants": {v: {k: m.get(k) for k in (*TABLE_METRICS, "recall_esi1_argmax", "critical_miss_cost")}
                             for v, m in (trans.get("variants") or {}).items()},
                "esi1_counts": {v: m.get("esi1_counts") for v, m in (trans.get("variants") or {}).items()},
                "n_fit": trans.get("n_fit"), "n_es": (trans.get("info") or {}).get("n_es"),
                "params_note": (trans.get("info") or {}).get("params_note"), "source_run": trans.get("source_run"),
                "priors": (trans.get("info") or {}).get("priors"),
            }
        title_tail = f"{cfg.feature_set} · tag {tag}"
        figs = []
        for kind, plotter in (("curves", plot_curves), ("delta", plot_deltas)):
            try:
                path = plotter(scenario, block, figures_dir / f"{kind}_{scenario}.png",
                               f"Eksperyment A — {SCENARIO_TITLES.get(scenario, scenario)} · {title_tail}")
                if path:
                    figs.append(str(path.relative_to(root)).replace("\\", "/"))
            except Exception as exc:  # noqa: BLE001 — wykres nie może unieważnić wyników
                log.warning(f"Wykres {kind} ({scenario}) nieudany: {type(exc).__name__}: {exc}")
        block["figures"] = figs
        scenarios[scenario] = block
    skipped_models = {m: st["reason"] for m, st in statuses.items() if not st["available"]}
    provenance: dict[str, list[dict[str, Any]]] = {}
    for done in done_by_scenario.values():
        for u in done:
            if u.get("base_model") not in config.TFM_MODELS:
                continue
            ucfg = u.get("config") or {}
            entry = {"weights": ucfg.get("weights"), "library_version": ucfg.get("library_version"),
                     "fake": ucfg.get("fake"), "override": (u.get("info") or {}).get("weights_override"),
                     "device": ucfg.get("device")}
            if entry not in provenance.setdefault(str(u["base_model"]), []):
                provenance[str(u["base_model"])].append(entry)
    results = {
        "analysis": ANALYSIS,
        "feature_set": cfg.feature_set,
        "tag": tag,
        "created": now_iso(),
        "data_fingerprint": fingerprint,
        "version": SMALL_DATA_VERSION,
        "config": {**asdict(cfg), "models_requested": list(cfg.models),
                   "models": [m for m in cfg.models if statuses[m]["available"]], "tag": tag},
        "primary_metric": PRIMARY_METRIC,
        "baseline": BASELINE,
        "alpha": ALPHA,
        "mid": {k: _mid(v[0]) for k, v in COMPARE_METRICS.items()},
        "tfm_out_of_box": TFM_OUT_OF_BOX,
        "lgbm_space": {k: list(v) for k, v in LGBM_SPACE.items()},
        "tfm_status": dict(statuses),
        "tfm_provenance": provenance,
        "skipped_models": skipped_models,
        "skipped_units": list(skipped),
        "n_units": len(units),
        "n_done": n_done,
        "complete": not failures and n_done == len(units),
        "failures": list(failures),
        "scenarios": scenarios,
    }
    results["decision"] = decision_lines(results)
    results["methods_paragraph"] = methods_paragraph(results)
    path = store.write_results(results)
    table = units_table(store, units, expect)
    csv_path = root / "units.csv"
    write_text_atomic(csv_path, table.to_csv(index=False))
    write_text_atomic(root / "report.md", "\n".join(report_lines(results)) + "\n")
    store.write_summary(summary_lines(results))
    log.info(f"Eksperyment A: {n_done}/{len(units)} jednostek → {path}; raport {root / 'report.md'}")
    return results


__all__ = [
    "BASELINE",
    "DEFAULT_REPEATS",
    "DEFAULT_SIZES",
    "PRIMARY_METRIC",
    "SCENARIOS",
    "SmallDataConfig",
    "SourceCounts",
    "analysis_path",
    "boot_summary",
    "build_units",
    "collect_statistics",
    "crossover",
    "crossover_bootstrap",
    "derive_seed",
    "draw_sample",
    "ensure_min_per_class",
    "estimate_unit_s",
    "fit_predict_tfm",
    "holm",
    "inner_folds",
    "parse_sizes_repeats",
    "parse_tfm_models",
    "report_lines",
    "run_small_data",
    "stratified_sample",
    "t_interval",
    "tfm_status",
    "tune_lightgbm_small",
    "two_level_draws",
    "unit_metrics",
    "validate_config",
    "verdict",
]
