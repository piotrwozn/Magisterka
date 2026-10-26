"""
Wspólna baza tabularnych modeli fundamentalnych (TFM): TabPFN-3.5 (`tabpfn`) i TabICLv2 (`tabicl`).

Model TFM nie uczy się wag — "trening" zapisuje KONTEKST (wiersze TRAIN), a predykcja to przebieg
wstępnie wytrenowanej sieci na kontekście + wierszach testowych (in-context learning). Stąd:

    fit(X, y)     — dobór cech (opcjonalnie `max_features`, bez etykiet), worki kontekstu (bags) z TRAIN,
                    zapis kontekstu W MODELU (float32 + etykiety + indeksy worków + priory klas):
                    artefakt jest samowystarczalny (bez danych treningowych przy wczytaniu),
    predict(X)    — dla każdego worka: zbudowanie i dopasowanie estymatora biblioteki na kontekście worka,
                    predykcja wierszy w porcjach (`predict_batch`), mapowanie `classes_` na 5 kolumn ESI,
                    (strategia "balanced") DOKŁADNA korekta priorów p_k ∝ p_k·π_k(fit)/π_k(kontekst),
                    średnia arytmetyczna worków; pamięć GPU zwalniana między workami,
    predict_proba_many([VAL, TEST]) — wszystkie ramki w jednym przebiegu każdego worka (bez ponownego
                    budowania bufora KV dla każdej ramki).

Odporność na brak pamięci GPU (OOM): jedna reguła rozpoznania (`is_tfm_oom`) i drabina awaryjna
biblioteki (adapter): każdy krok logowany (WARNING) i zapisywany w `tfm_events` (→ run_info). Ostatni
szczebel → `TFMOutOfMemoryError` z poleceniem, które parametry zmniejszyć.

CPU: kontekst worka > `config.TFM_CPU_MAX_CONTEXT` bez GPU → błąd NA STARCIE (z instrukcją), nigdy ciche
wielogodzinne liczenie na CPU (także zamiast bazowego powrotu GPU → CPU).

Artefakt (`save`): klucz `model` = `TFMPredictor` (pikowalny: konfiguracja + kontekst + priory; dopasowane
estymatory torch NIGDY nie są zapisywane), więc backend może wołać `bundle["model"].predict_proba(X)`,
gdy torch i biblioteka są zainstalowane. Dodatkowe klucze: `pretrained`, `tfm` (biblioteka, wersja,
identyfikator wag, konfiguracja, worki), `tfm_events`.

torch / tabpfn / tabicl importowane WYŁĄCZNIE leniwie (w metodach adapterów) — `import src.models` działa
bez torch. Haki testowe (nigdy w zwykłym użyciu; aktywne = WARNING + `tfm.weights_override` w run_info):
    SOR_AI_TFM_FAKE=1               — atrapa estymatora (`src/models/tfm_fake.py`), ta sama ścieżka kodu,
    SOR_AI_TABPFN_MODEL_PATH=<plik> — wagi TabPFN z pliku (nazwa z "v3.5"), np. losowe wagi testowe,
    SOR_AI_TABICL_MODEL_PATH=<plik> — checkpoint TabICL z pliku (allow_auto_download=False).
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import logging
import math
import os
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseTriageModel, full_class_proba
from src.utils import config as _config
from src.utils.config import N_CLASSES, RANDOM_SEED
from src.utils.hardware import device_for
from src.utils.logger import get_logger

log = get_logger(__name__)

FAKE_ENV = "SOR_AI_TFM_FAKE"
MIN_ROWS_PER_CLASS = 2                    # każdy worek: ≥ 2 wiersze każdej klasy obecnej w danych fit
MIN_PREDICT_BATCH = 1_024                 # dolna granica porcji testu w drabinie OOM
BAG_MODES = ("partition", "resample")
CONTEXT_STRATEGIES = ("stratified", "balanced")
# Cechy cc_* zawsze chronione przy `max_features` (agregaty skarg); flagi CC_CRITICAL — z engineering
PROTECTED_CC_FEATURES = ("cc_count", "cc_critical")
PROTECTED_CC_PREFIXES = ("cc_group_",)
# Fragmenty komunikatów wyjątków oznaczających brak pamięci (CUDA, TabICL, alokacja bufora)
OOM_MESSAGE_PATTERNS = ("out of memory", "memory allocation failed", "failed to execute even with minimum batch size")
_FALSE_VALUES = ("", "0", "false", "no", "off", "nie")
_override_announced: set[str] = set()     # przekierowania wag już zgłoszone w tym procesie (WARNING raz)


# ─────────────────────────────────────────
# Wyjątki i rozpoznanie OOM
# ─────────────────────────────────────────
class TFMOutOfMemoryError(RuntimeError):
    """Brak pamięci GPU także po wszystkich krokach drabiny awaryjnej."""


class TFMDeviceError(RuntimeError):
    """TFM z dużym kontekstem bez GPU (CPU byłoby wielokrotnie za wolne)."""


class TFMWeightsError(FileNotFoundError):
    """Brak lokalnych wag modelu fundamentalnego (pobranie: scripts/13_tfm_setup.py)."""


def fake_mode() -> bool:
    """Czy aktywna atrapa estymatora (`SOR_AI_TFM_FAKE=1`; czytane na bieżąco — działa w podprocesach)."""
    return os.environ.get(FAKE_ENV, "").strip().lower() not in _FALSE_VALUES


def is_tfm_oom(exc: BaseException) -> bool:
    """
    Jedna reguła rozpoznania braku pamięci (obie biblioteki):
        - `torch.OutOfMemoryError` (= `torch.cuda.OutOfMemoryError`),
        - klasa wyjątku (lub bazowa) z "OutOfMemory" w nazwie (TabPFNCUDAOutOfMemoryError,
          TabPFNOutOfMemoryError — NIE są RuntimeError),
        - komunikat z "out of memory", "memory allocation failed" albo
          "Failed to execute even with minimum batch size" (TabICL).
    """
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return False
    torch = sys.modules.get("torch")          # torch nie zaimportowany → wyjątek nie jest z torch
    if torch is not None:
        for owner in (torch, getattr(torch, "cuda", None)):
            oom_cls = getattr(owner, "OutOfMemoryError", None)
            if isinstance(oom_cls, type) and isinstance(exc, oom_cls):
                return True
    if any("OutOfMemory" in cls.__name__ for cls in type(exc).__mro__):
        return True
    text = str(exc).lower()
    return any(pattern in text for pattern in OOM_MESSAGE_PATTERNS)


def _short(exc: BaseException, limit: int = 300) -> str:
    lines = str(exc).strip().splitlines()
    return (lines[0] if lines else type(exc).__name__)[:limit]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _dist_version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


# ─────────────────────────────────────────
# Pamięć GPU (torch tylko, gdy już zaimportowany przez bibliotekę)
# ─────────────────────────────────────────
def _torch_cuda():
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        return torch if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def free_device_memory() -> None:
    """`gc.collect()` + `torch.cuda.empty_cache()` (bez importu torch, gdy nie był używany)."""
    gc.collect()
    torch = _torch_cuda()
    if torch is not None:
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — sprzątanie nie może przykryć właściwego błędu
            pass


def _peak_reset() -> None:
    torch = _torch_cuda()
    if torch is not None:
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001
            pass


def _peak_mb() -> float | None:
    torch = _torch_cuda()
    if torch is None:
        return None
    try:
        return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────────────────────
# Wagi: identyfikator pliku (nazwa + rozmiar + sha256)
# ─────────────────────────────────────────
@lru_cache(maxsize=16)
def _sha256(path: str, size: int, mtime_ns: int) -> str:
    """sha256 pliku (pamięć w procesie wg ścieżki, rozmiaru i czasu modyfikacji; 835 MB ≈ 2,5 s)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weights_identity(path: Path | str) -> dict[str, Any]:
    """{'path', 'file', 'size', 'sha256'} pliku wag (identyfikator w artefakcie i kluczu wyboru konfiguracji)."""
    p = Path(path)
    st = p.stat()
    return {"path": str(p), "file": p.name, "size": int(st.st_size),
            "sha256": _sha256(str(p), int(st.st_size), int(st.st_mtime_ns))}


def same_weights(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> bool:
    """Ten sam plik wag: sha256, gdy oba znane; inaczej nazwa + rozmiar."""
    if not a or not b:
        return True
    if a.get("sha256") and b.get("sha256"):
        return a["sha256"] == b["sha256"]
    return (a.get("file"), a.get("size")) == (b.get("file"), b.get("size"))


# ─────────────────────────────────────────
# Dobór cech (bez etykiet)
# ─────────────────────────────────────────
def protected_features(names: Sequence[str]) -> list[str]:
    """Cechy nigdy nieusuwane przez `max_features`: wszystkie spoza cc_*, agregaty cc i flagi CC_CRITICAL."""
    from src.features.engineering import CC_CRITICAL

    critical = set(CC_CRITICAL)
    return [n for n in names
            if not n.startswith("cc_") or n in PROTECTED_CC_FEATURES or n.startswith(PROTECTED_CC_PREFIXES)
            or n in critical]


def select_features(names: Sequence[str], X: pd.DataFrame | np.ndarray,
                    max_features: int | None) -> tuple[list[str], dict[str, Any] | None]:
    """
    Redukcja cech dla TFM BEZ etykiet: chronione (`protected_features`) zostają zawsze, z pozostałych flag
    cc_* zostają najczęstsze (częstość w danych fit — TRAIN) aż do limitu. Gdy same chronione przekraczają
    limit — zostają tylko chronione (zapis w opisie). Kolejność cech = kolejność wejścia.

    Returns
    -------
    (zachowane cechy, opis doboru albo None, gdy bez redukcji)
    """
    names = list(names)
    if max_features is None or int(max_features) >= len(names):
        return names, None
    limit = int(max_features)
    if limit < 1:
        raise ValueError(f"max_features musi być ≥ 1 albo None, podano {max_features}")
    protected = protected_features(names)
    rest = [n for n in names if n not in set(protected)]
    if len(protected) >= limit:
        kept_set = set(protected)
        cutoff = None
        log.warning(f"max_features={limit}: same cechy chronione to {len(protected)} — zostają tylko one "
                    f"(odrzucone wszystkie {len(rest)} flagi skarg cc_*)")
    else:
        if isinstance(X, pd.DataFrame):
            values = X[rest].to_numpy(dtype=np.float32)
        else:
            idx = [names.index(n) for n in rest]
            values = np.asarray(X, dtype=np.float32)[:, idx]
        prevalence = (np.nan_to_num(values, nan=0.0) != 0).mean(axis=0) if len(values) else np.zeros(len(rest))
        order = sorted(range(len(rest)), key=lambda i: (-float(prevalence[i]), i))
        keep_rest = [rest[i] for i in order[: limit - len(protected)]]
        kept_set = set(protected) | set(keep_rest)
        cutoff = float(prevalence[order[limit - len(protected) - 1]]) if keep_rest else None
    kept = [n for n in names if n in kept_set]
    dropped = [n for n in names if n not in kept_set]
    info = {"max_features": limit, "n_in": len(names), "n_kept": len(kept), "n_protected": len(protected),
            "cutoff_prevalence": cutoff, "dropped": dropped}
    return kept, info


# ─────────────────────────────────────────
# Worki kontekstu (bags)
# ─────────────────────────────────────────
def resolve_n_bags(spec: Any, n_rows: int, n_ctx: int, max_bags: int) -> int:
    """n_bags: int ≥ 1 albo "auto" = min(max_bags, ceil(N_fit / n_ctx))."""
    if isinstance(spec, str):
        if spec.strip().lower() != "auto":
            raise ValueError(f"n_bags: liczba ≥ 1 albo 'auto', podano {spec!r}")
        return max(1, min(int(max_bags), math.ceil(max(n_rows, 1) / max(int(n_ctx), 1))))
    value = int(spec)
    if value < 1:
        raise ValueError(f"n_bags musi być ≥ 1, podano {spec}")
    return value


def _proportional(counts: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Przydział m wierszy proporcjonalnie do `counts` (największe reszty; nigdy więcej niż count)."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if m >= total:
        return counts.copy()
    exact = counts * (m / total)
    alloc = np.floor(exact).astype(np.int64)
    rest = int(m - alloc.sum())
    if rest > 0:
        frac = exact - alloc + rng.random(len(counts)) * 1e-9      # losowe rozstrzygnięcie remisów
        frac[alloc >= counts] = -1.0
        alloc[np.argsort(-frac, kind="stable")[:rest]] += 1
    return alloc


def _water_fill(counts: np.ndarray, m: int) -> np.ndarray:
    """Limit na klasę c taki, że Σ min(n_k, c) = m — rzadkie klasy w całości, liczne przycięte."""
    counts = np.asarray(counts, dtype=np.int64)
    if m >= counts.sum():
        return counts.copy()
    alloc = np.zeros_like(counts)
    remaining = int(m)
    order = [int(k) for k in np.argsort(counts, kind="stable") if counts[k] > 0]
    for i, k in enumerate(order):
        share = remaining // (len(order) - i)
        alloc[k] = min(int(counts[k]), share)
        remaining -= int(alloc[k])
    for k in sorted(order, key=lambda c: -counts[c]):          # reszta z dzielenia całkowitego
        if remaining <= 0:
            break
        if alloc[k] < counts[k]:
            alloc[k] += 1
            remaining -= 1
    return alloc


def _take_strata(pos: np.ndarray, labels: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """m pozycji z `pos`, proporcjonalnie do warstw `labels`."""
    if m >= len(pos):
        return np.sort(pos)
    uniq, inv = np.unique(labels, return_inverse=True)
    alloc = _proportional(np.bincount(inv, minlength=len(uniq)), m, rng)
    parts = [rng.choice(pos[inv == s], int(alloc[s]), replace=False) for s in range(len(uniq)) if alloc[s] > 0]
    return np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)


def _select(pos: np.ndarray, labels: np.ndarray, y: np.ndarray, m: int, strategy: str,
            rng: np.random.Generator) -> np.ndarray:
    """m pozycji: "stratified" — proporcjonalnie do warstw; "balanced" — limit na klasę, w klasie warstwy."""
    if m >= len(pos):
        return np.sort(pos)
    if strategy == "stratified":
        return _take_strata(pos, labels[pos], m, rng)
    classes = y[pos]
    alloc = _water_fill(np.bincount(classes, minlength=N_CLASSES), m)
    parts = []
    for k in range(N_CLASSES):
        if alloc[k] > 0:
            members = pos[classes == k]
            parts.append(_take_strata(members, labels[members], int(alloc[k]), rng))
    return np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)


def _deal(pos: np.ndarray, labels: np.ndarray, n_bags: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Rozłączny podział: warstwy po kolei (wiersze przetasowane) rozdawane po kolei do worków."""
    if n_bags == 1:
        return [np.sort(pos)]
    lab = labels[pos]
    uniq, inv = np.unique(lab, return_inverse=True)
    seq = np.concatenate([rng.permutation(pos[inv == s]) for s in rng.permutation(len(uniq))])
    bag_of = (np.arange(len(seq)) + int(rng.integers(n_bags))) % n_bags
    return [np.sort(seq[bag_of == b]) for b in range(n_bags)]


def _ensure_min_per_class(bags: list[np.ndarray], y: np.ndarray, min_per_class: int,
                          rng: np.random.Generator) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """
    Każdy worek: ≥ min(min_per_class, n_k) wierszy każdej klasy obecnej w danych fit. Brakujące wiersze
    rzadkiej klasy dobierane spoza worka (także z innych worków — ponowne użycie przy podziale) w miejsce
    losowych wierszy najliczniejszej klasy worka (rozmiar worka bez zmian).
    """
    counts = np.bincount(y, minlength=N_CLASSES)
    events: list[dict[str, Any]] = []
    out = []
    for b, bag in enumerate(bags):
        bag = np.asarray(bag, dtype=np.int64)
        for k in np.flatnonzero(counts):
            need = min(int(min_per_class), int(counts[k]))
            have = int((y[bag] == k).sum())
            if have >= need:
                continue
            missing = need - have
            candidates = np.setdiff1d(np.flatnonzero(y == k), bag, assume_unique=True)
            add = rng.choice(candidates, missing, replace=False)
            bag_y = y[bag]
            other = [c for c in range(N_CLASSES) if c != k]
            major = max(other, key=lambda c: int((bag_y == c).sum()))
            major_pos = bag[bag_y == major]
            if len(major_pos) - missing >= min(int(min_per_class), int(counts[major])):
                drop = rng.choice(major_pos, missing, replace=False)
                bag = np.setdiff1d(bag, drop, assume_unique=True)
            bag = np.sort(np.concatenate([bag, add]))
            others = [bags[o] for o in range(len(bags)) if o != b]
            reused = int(np.isin(add, np.concatenate(others)).sum()) if others else 0
            events.append({"kind": "rare_class_reuse", "time": _now(), "bag": b, "class": int(k), "esi": int(k) + 1,
                           "added": int(missing), "reused_from_other_bags": reused,
                           "detail": f"worek {b}: dobrano {missing} wiersz(e) ESI {int(k) + 1} "
                                     f"(minimum {need}; {reused} z innych worków)"})
        out.append(bag)
    return out, events


def build_bags(
    y: np.ndarray | pd.Series,
    strata: np.ndarray | None,
    *,
    n_ctx: int,
    n_bags: int,
    bag_mode: str = "partition",
    context_strategy: str = "stratified",
    seed: int = RANDOM_SEED,
    min_per_class: int = MIN_ROWS_PER_CLASS,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """
    Indeksy (pozycje w y) worków kontekstu — deterministycznie z `seed`.

    partition — min(N, n_bags·n_ctx) wierszy wybranych stratyfikowanie (balanced: z limitem na klasę),
                rozdanych rozłącznie do worków (każdy worek ≈ ten sam skład warstw); n_bags·n_ctx ≥ N →
                WSZYSTKIE wiersze,
    resample  — każdy worek niezależnie: n_ctx wierszy (stratified / balanced).
    strata    — etykiety warstw (ESI × oddział) pozycyjnie względem y; None → warstwy = ESI.
    Każdy worek ma ≥ min_per_class wierszy każdej klasy obecnej w y (`rare_class_reuse` w zdarzeniach).
    """
    y = np.asarray(y).astype(np.int64).ravel()
    n = len(y)
    if n == 0:
        raise ValueError("build_bags: brak wierszy kontekstu")
    if bag_mode not in BAG_MODES:
        raise ValueError(f"bag_mode: {' | '.join(BAG_MODES)}, podano {bag_mode!r}")
    if context_strategy not in CONTEXT_STRATEGIES:
        raise ValueError(f"context_strategy: {' | '.join(CONTEXT_STRATEGIES)}, podano {context_strategy!r}")
    n_ctx, n_bags = int(n_ctx), int(n_bags)
    if n_ctx < 1 or n_bags < 1:
        raise ValueError(f"n_ctx i n_bags muszą być ≥ 1 (n_ctx={n_ctx}, n_bags={n_bags})")
    if strata is not None and len(strata) != n:
        raise ValueError(f"context_strata: {len(strata)} etykiet ≠ {n} wierszy kontekstu")
    labels = np.asarray(strata).astype(str) if strata is not None else y.astype(str)
    rng = np.random.default_rng(int(seed))
    positions = np.arange(n, dtype=np.int64)
    if bag_mode == "partition":
        chosen = _select(positions, labels, y, min(n, n_bags * n_ctx), context_strategy, rng)
        bags = _deal(chosen, labels, n_bags, rng)
    else:
        bags = [_select(positions, labels, y, min(n, n_ctx), context_strategy,
                        np.random.default_rng([int(seed), b])) for b in range(n_bags)]
    return _ensure_min_per_class(bags, y, min_per_class, rng)


def class_prior(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.asarray(y).astype(np.int64), minlength=N_CLASSES)[:N_CLASSES].astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def _renormalize(p: np.ndarray) -> np.ndarray:
    np.clip(p, 0.0, None, out=p)
    sums = p.sum(axis=1, keepdims=True)
    return np.divide(p, sums, out=np.full_like(p, 1.0 / N_CLASSES), where=sums > 0)


# ─────────────────────────────────────────
# Adapter biblioteki (TabPFN / TabICL / atrapa)
# ─────────────────────────────────────────
class TFMAdapter:
    """
    Interfejs biblioteki dla `TFMPredictor`: wagi, opcje startowe, budowa i dopasowanie estymatora,
    predykcja, ustawienia procesu (przywracane), drabina awaryjna OOM.

    `fake=True` — budowany jest `tfm_fake.FakeTFMClassifier` z tymi samymi opcjami, a ustawienia procesu
    trafiają do `tfm_fake.FAKE_SETTINGS` — reszta ścieżki identyczna jak z prawdziwą biblioteką.
    """

    library: str = "?"
    display_name: str = "?"
    weights_env: str = ""

    def __init__(self, fake: bool = False):
        self.fake = bool(fake)

    # ── wersje / wagi ──
    def library_version(self) -> str | None:
        return "fake" if self.fake else _dist_version(self.library)

    def weights_override(self) -> str | None:
        """Aktywne przekierowanie wag (opis do run_info) albo None."""
        if self.fake:
            return f"{FAKE_ENV}=1"
        raw = os.environ.get(self.weights_env, "").strip()
        return f"{self.weights_env}={raw}" if raw else None

    def _override_path(self) -> Path | None:
        raw = os.environ.get(self.weights_env, "").strip()
        if not raw:
            return None
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise TFMWeightsError(f"{self.weights_env}={raw}: plik wag nie istnieje")
        if f"{self.weights_env}={path}" not in _override_announced:
            _override_announced.add(f"{self.weights_env}={path}")
            log.warning(f"{self.display_name}: AKTYWNE PRZEKIEROWANIE WAG {self.weights_env}={path} — tylko do "
                        f"testów (losowe wagi dają bezwartościowe predykcje); zapis w run_info: tfm.weights_override")
        return path

    def resolve_weights(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """{'path', 'file', 'size', 'sha256', 'override', …}; brak pliku → TFMWeightsError z poleceniem."""
        if self.fake:
            if FAKE_ENV not in _override_announced:
                _override_announced.add(FAKE_ENV)
                log.warning(f"{FAKE_ENV}=1 — modele TFM używają ATRAPY estymatora (tylko testy; bez torch i wag); "
                            f"zapis w run_info: tfm.weights_override")
            return {"path": None, "file": "fake", "size": 0, "sha256": None, "override": f"{FAKE_ENV}=1"}
        raise NotImplementedError

    # ── opcje startowe i estymator ──
    def initial_runtime(self, params: Mapping[str, Any], *, n_ctx: int, n_features: int,
                        device: str) -> dict[str, Any]:
        """Opcje zmieniane w trakcie (drabina OOM) — NIGDY nie trafiają do `params` (refit dziedziczyłby je)."""
        return {"predict_batch": int(params.get("predict_batch") or 8_192)}

    def estimator_options(self, params: Mapping[str, Any], runtime: Mapping[str, Any], weights: Mapping[str, Any],
                          device: str, seed: int) -> dict[str, Any]:
        raise NotImplementedError

    def build(self, params: Mapping[str, Any], runtime: Mapping[str, Any], weights: Mapping[str, Any],
              device: str, seed: int) -> Any:
        options = self.estimator_options(params, runtime, weights, device, seed)
        if self.fake:
            from src.models.tfm_fake import FakeTFMClassifier

            fake_options = {k: v for k, v in options.items() if k != "model_path"}
            return FakeTFMClassifier(library=self.library, options=fake_options, random_state=int(seed))
        return self._make_estimator(options)

    def _make_estimator(self, options: dict[str, Any]) -> Any:
        raise NotImplementedError

    def fit(self, est: Any, X: np.ndarray, y: np.ndarray) -> None:
        est.fit(X, y)

    def predict(self, est: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(prawdopodobieństwa, klasy kolumn) — kolumny = `est.classes_` (tylko klasy z kontekstu)."""
        return np.asarray(est.predict_proba(X)), np.asarray(est.classes_)

    def describe(self, est: Any) -> dict[str, Any]:
        return {}

    def release(self, est: Any) -> None:
        """Zwolnienie estymatora (bufor KV, kopia modelu)."""

    # ── ustawienia procesu ──
    @contextlib.contextmanager
    def process_settings(self) -> Iterator[None]:
        """Ustawienia biblioteki na całe wywołanie predykcji (wszystkie worki) — przywracane w `finally`."""
        yield

    @contextlib.contextmanager
    def library_settings(self, runtime: Mapping[str, Any]) -> Iterator[None]:
        """Ustawienia zależne od opcji drabiny OOM (jedna próba worka) — przywracane w `finally`."""
        yield

    @contextlib.contextmanager
    def call_session(self, n_jobs: int | None, events: list | None) -> Iterator[None]:
        """Całe wywołanie predykcji: wątki torch, ustawienia procesu, ostrzeżenia biblioteki o OOM."""
        with contextlib.ExitStack() as stack:
            if not self.fake:
                stack.enter_context(_torch_threads(n_jobs))
                stack.enter_context(_capture_library_oom(self.library, events))
            stack.enter_context(self.process_settings())
            yield

    # ── OOM ──
    def next_oom_step(self, runtime: Mapping[str, Any], phase: str) -> tuple[dict[str, Any], str, bool] | None:
        """Następny szczebel drabiny: (nowe opcje, opis, czy przebudować estymator) albo None (koniec)."""
        return None

    def oom_advice(self) -> str:
        return "n_ctx, n_estimators, predict_batch"

    def cache_bytes(self, params: Mapping[str, Any], runtime: Mapping[str, Any], n_ctx: int,
                    n_features: int) -> int | None:
        """Szacunek pamięci bufora jednego worka (bajty; backend: kontrola budżetu VRAM)."""
        return None


@contextlib.contextmanager
def _torch_threads(n_jobs: int | None) -> Iterator[None]:
    """`torch.set_num_threads(n_jobs)` na czas wywołania (TabICL n_jobs działa tylko w predict)."""
    torch = sys.modules.get("torch")
    if torch is None:
        try:
            import torch  # noqa: F811 — biblioteka TFM i tak go importuje
        except Exception:  # noqa: BLE001
            torch = None
    if torch is None or not n_jobs:
        yield
        return
    previous = torch.get_num_threads()
    torch.set_num_threads(max(1, int(n_jobs)))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class _OOMLogCapture(logging.Handler):
    def __init__(self, events: list):
        super().__init__(level=logging.WARNING)
        self.events = events

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        if "oom" in message.lower() or "out of memory" in message.lower():
            self.events.append({"kind": "library_oom_warning", "time": _now(), "logger": record.name,
                                "detail": message[:300]})


@contextlib.contextmanager
def _capture_library_oom(library: str, events: list | None) -> Iterator[None]:
    """Ostrzeżenia biblioteki o OOM (np. TabPFN "OOM: halving row_chunk_size") → tfm_events."""
    if events is None:
        yield
        return
    handler = _OOMLogCapture(events)
    logger = logging.getLogger(library)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)


def adapter_for(library: str, fake: bool | None = None) -> TFMAdapter:
    """Adapter biblioteki po nazwie modelu ('tabpfn' / 'tabicl'); fake=None → `fake_mode()`."""
    use_fake = fake_mode() if fake is None else bool(fake)
    if library == "tabpfn":
        from src.models.tabpfn_model import TabPFNAdapter

        return TabPFNAdapter(fake=use_fake)
    if library == "tabicl":
        from src.models.tabicl_model import TabICLAdapter

        return TabICLAdapter(fake=use_fake)
    raise ValueError(f"Nieznana biblioteka TFM: {library!r}")


# ─────────────────────────────────────────
# Predyktor w artefakcie (bundle["model"])
# ─────────────────────────────────────────
class TFMPredictor:
    """
    Samowystarczalny predyktor TFM: konfiguracja + kontekst (float32) + etykiety + worki + priory.

    `predict_proba(X)` / `predict_proba_many([...])` → (n, 5) float64, kolumny ESI 1–5 (klasy 0..4).
    Estymatory biblioteki budowane leniwie przy predykcji; w pamięci najwyżej `keep_fitted_bags`
    dopasowanych (domyślnie 1; backend: "all" + `warm_up()`); nigdy nie trafiają do pickle.
    """

    def __init__(
        self,
        *,
        library: str,
        params: Mapping[str, Any],
        feature_names: Sequence[str],
        kept_features: Sequence[str],
        X_ctx: np.ndarray,
        y_ctx: np.ndarray,
        bags: Sequence[np.ndarray],
        prior_fit: np.ndarray,
        correction: bool,
        weights: Mapping[str, Any],
        library_version: str | None,
        device: str,
        runtime: Mapping[str, Any],
        fake: bool,
        n_fit: int,
        feature_info: Mapping[str, Any] | None = None,
        events: list | None = None,
    ):
        self.library = library
        self.params = dict(params)
        self.feature_names = list(feature_names)
        self.kept_features = list(kept_features)
        self.X_ctx = np.ascontiguousarray(X_ctx, dtype=np.float32)
        self.y_ctx = np.asarray(y_ctx).astype(np.int8)
        self.bags = [np.asarray(b, dtype=np.int64) for b in bags]
        self.prior_fit = np.asarray(prior_fit, dtype=np.float64)
        self.correction = bool(correction)
        self.weights = dict(weights)
        self.library_version = library_version
        self.device = device
        self.runtime = dict(runtime)
        self.fake = bool(fake)
        self.n_fit = int(n_fit)
        self.feature_info = dict(feature_info) if feature_info else None
        self.events: list = events if events is not None else []
        self.keep_fitted_bags: int | str = 1
        self.stats: dict[str, Any] = {"bags": {}, "calls": 0}
        self._state_init()

    def _state_init(self) -> None:
        self._fitted: dict[int, Any] = {}
        self._adapter: TFMAdapter | None = None
        self._weights_current: dict[str, Any] | None = None
        self._runtime_device: str | None = None
        index = {name: i for i, name in enumerate(self.feature_names)}
        self._kept_idx = np.array([index[n] for n in self.kept_features], dtype=np.int64)

    # ── pickle: bez estymatorów i adaptera ──
    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        for key in ("_fitted", "_adapter", "_weights_current", "_runtime_device", "_kept_idx"):
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._state_init()

    # ── opis ──
    @property
    def classes_(self) -> np.ndarray:
        return np.arange(N_CLASSES)

    @property
    def n_bags(self) -> int:
        return len(self.bags)

    def context_sizes(self) -> list[int]:
        return [int(len(b)) for b in self.bags]

    def bag_class_counts(self) -> list[list[int]]:
        return [np.bincount(self.y_ctx[b].astype(np.int64), minlength=N_CLASSES).tolist() for b in self.bags]

    def bag_prior(self, b: int) -> np.ndarray:
        return class_prior(self.y_ctx[self.bags[b]])

    def max_context(self) -> int:
        return max(self.context_sizes()) if self.bags else 0

    def throughput(self) -> dict[str, Any]:
        """Ostatnie pomiary per worek: fit_s, predict_s, rows, rows_per_s, peak_vram_mb + sumy."""
        bags = self.stats.get("bags") or {}
        if not bags:
            return {}
        values = [bags[b] for b in sorted(bags)]
        predict_s = float(sum(v["predict_s"] for v in values))
        rows = int(sum(v["rows"] for v in values))
        fits = [v["fit_s"] for v in values if v.get("fit_s")]
        peaks = [v["peak_vram_mb"] for v in values if v.get("peak_vram_mb") is not None]
        return {
            "n_bags_measured": len(values),
            "fit_s_per_bag": round(float(np.mean(fits)), 3) if fits else None,
            "predict_s": round(predict_s, 3),
            "rows_predicted": rows,
            "rows_per_s": round(rows / predict_s, 2) if predict_s > 0 else None,
            "peak_vram_mb": max(peaks) if peaks else None,
            "per_bag": {str(b): bags[b] for b in sorted(bags)},
        }

    def estimated_cache_bytes(self) -> int | None:
        """Szacunek sumy buforów wszystkich worków (backend: kontrola budżetu VRAM przed `warm_up`)."""
        adapter = self._get_adapter()
        sizes = [adapter.cache_bytes(self.params, self.runtime, n, len(self.kept_features))
                 for n in self.context_sizes()]
        return None if any(s is None for s in sizes) else int(sum(sizes))

    # ── wejście ──
    def _matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.kept_features if c not in X.columns]
            if missing:
                more = " …" if len(missing) > 10 else ""
                raise KeyError(f"Brak {len(missing)} cech wymaganych przez {self.library}: {missing[:10]}{more}")
            arr = X[self.kept_features].to_numpy(dtype=np.float32, copy=True)
        else:
            arr = np.array(X, dtype=np.float32, copy=True)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] == len(self.feature_names) and len(self.kept_features) != len(self.feature_names):
                arr = arr[:, self._kept_idx]
            elif arr.shape[1] != len(self.kept_features):
                raise ValueError(f"{self.library}: macierz ma {arr.shape[1]} kolumn, oczekiwano "
                                 f"{len(self.feature_names)} (wszystkie cechy) albo {len(self.kept_features)}")
        arr[~np.isfinite(arr)] = np.nan                     # ±inf → NaN (TabICL odrzuca nieskończoności)
        return arr

    # ── adapter, wagi, urządzenie ──
    def _use_fake(self) -> bool:
        return self.fake or fake_mode()

    def _get_adapter(self) -> TFMAdapter:
        fake = self._use_fake()
        if self._adapter is None or self._adapter.fake != fake:
            self._adapter = adapter_for(self.library, fake=fake)
        return self._adapter

    def _weights_for_predict(self, adapter: TFMAdapter) -> dict[str, Any]:
        """Wagi w chwili predykcji (bieżąca maszyna, raz na proces); inny plik niż przy fit → WARNING + zdarzenie."""
        if adapter.fake:
            return dict(self.weights)
        if self._weights_current is None:
            current = adapter.resolve_weights(self.params)
            self._weights_current = current
            if not same_weights(self.weights, current):
                detail = (f"wagi przy predykcji ({current.get('file')}, sha256 {str(current.get('sha256'))[:12]}) "
                          f"≠ wagi przy fit ({self.weights.get('file')}, sha256 {str(self.weights.get('sha256'))[:12]})")
                log.warning(f"{adapter.display_name}: {detail} — predykcje mogą różnić się od zapisanych w runie")
                self.events.append({"kind": "weights_mismatch", "time": _now(), "detail": detail})
        return self._weights_current

    def _device(self, adapter: TFMAdapter) -> str:
        """Urządzenie predykcji: z fit; GPU niedostępne → CPU tylko dla małego kontekstu (inaczej błąd)."""
        if self._runtime_device is not None:
            return self._runtime_device
        device = self.device
        if device.startswith(("cuda", "gpu")) and not adapter.fake:
            from src.utils.hardware import torch_info

            if not torch_info()["cuda_available"]:
                limit = int(_config.TFM_CPU_MAX_CONTEXT)
                if self.max_context() > limit:
                    raise TFMDeviceError(
                        f"{adapter.display_name}: model dopasowano na GPU, a tu CUDA jest niedostępna; kontekst "
                        f"{self.max_context():,} wierszy > TFM_CPU_MAX_CONTEXT={limit:,} — predykcja na CPU "
                        f"trwałaby godziny. Uruchom na maszynie z GPU (torch z CUDA).")
                detail = f"CUDA niedostępna — predykcja na CPU (kontekst {self.max_context():,} ≤ {limit:,})"
                log.warning(f"{adapter.display_name}: {detail}")
                self.events.append({"kind": "device_fallback", "time": _now(), "detail": detail})
                device = "cpu"
        self._runtime_device = device
        return device

    def _seed(self, b: int) -> int:
        base = self.params.get("random_state")
        base = RANDOM_SEED if base is None else int(base)
        return int((base + b) % (2**31 - 1))

    def _n_jobs(self) -> int | None:
        value = self.params.get("n_jobs")
        return int(value) if value else int(_config.N_JOBS)

    def _keep_limit(self) -> float:
        keep = self.keep_fitted_bags
        if isinstance(keep, str):
            if keep.strip().lower() != "all":
                raise ValueError(f"keep_fitted_bags: liczba ≥ 0 albo 'all', podano {keep!r}")
            return math.inf
        return max(0, int(keep))

    # ── estymatory ──
    def _release_est(self, adapter: TFMAdapter, est: Any) -> None:
        if est is not None:
            try:
                adapter.release(est)
            except Exception:  # noqa: BLE001
                pass

    def _make_room(self, adapter: TFMAdapter) -> None:
        """Przed budową worka: w pamięci łącznie najwyżej max(keep_fitted_bags, 1) estymatorów."""
        capacity = max(self._keep_limit(), 1)
        while self._fitted and len(self._fitted) + 1 > capacity:
            oldest = next(iter(self._fitted))
            self._release_est(adapter, self._fitted.pop(oldest))
            free_device_memory()

    def _keep_or_release(self, adapter: TFMAdapter, b: int, est: Any) -> None:
        if len(self._fitted) < self._keep_limit():
            self._fitted[b] = est
        else:
            self._release_est(adapter, est)
            del est
            free_device_memory()

    def release(self) -> None:
        """Zwalnia wszystkie dopasowane estymatory (pamięć GPU)."""
        adapter = self._adapter
        for b in list(self._fitted):
            est = self._fitted.pop(b)
            if adapter is not None:
                self._release_est(adapter, est)
        free_device_memory()

    def _to_full(self, proba: np.ndarray, classes: np.ndarray, b: int) -> np.ndarray:
        """Kolumny biblioteki (klasy z kontekstu) → 5 kolumn ESI; korekta priorów worka ("balanced")."""
        proba = np.asarray(proba, dtype=np.float64)
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)
        cols = np.asarray(classes).astype(np.int64).ravel()
        if len(cols) != proba.shape[1]:
            raise RuntimeError(f"{self.library}: {proba.shape[1]} kolumn prawdopodobieństw, klasy {cols.tolist()}")
        full = np.zeros((proba.shape[0], N_CLASSES), dtype=np.float64)
        full[:, cols] = proba
        if self.correction:
            context = self.bag_prior(b)
            factor = np.divide(self.prior_fit, context, out=np.zeros(N_CLASSES), where=context > 0)
            full *= factor
        return _renormalize(full)

    # ── predykcja ──
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.predict_proba_many([X])[0]

    def warm_up(self) -> None:
        """Dopasowuje wszystkie worki z góry (przy keep_fitted_bags="all" — backend przy starcie)."""
        empty = np.zeros((0, len(self.kept_features)), dtype=np.float32)
        self.predict_proba_many([empty])

    def predict_proba_many(self, frames: Sequence[pd.DataFrame | np.ndarray]) -> list[np.ndarray]:
        """Każdy worek dopasowany raz, wszystkie ramki w jednym przebiegu; średnia worków."""
        if not self.bags:
            raise RuntimeError(f"{self.library}: predyktor bez worków kontekstu")
        mats = [self._matrix(f) for f in frames]
        adapter = self._get_adapter()
        device = self._device(adapter)
        weights = self._weights_for_predict(adapter)
        order = [b for b in self._fitted if b < self.n_bags] + [b for b in range(self.n_bags) if b not in self._fitted]
        per_bag: dict[int, list[np.ndarray]] = {}
        with adapter.call_session(self._n_jobs(), self.events):
            for b in order:
                per_bag[b] = self._predict_bag(adapter, b, mats, device, weights)
        self.stats["calls"] = int(self.stats.get("calls", 0)) + 1
        outputs = []
        for i, mat in enumerate(mats):
            acc = np.zeros((len(mat), N_CLASSES), dtype=np.float64)
            for b in range(self.n_bags):                    # stała kolejność sumowania (deterministycznie)
                acc += per_bag[b][i]
            outputs.append(_renormalize(acc / self.n_bags))
        return outputs

    def _predict_rows(self, adapter: TFMAdapter, est: Any, mat: np.ndarray, b: int) -> np.ndarray:
        batch = max(1, int(self.runtime.get("predict_batch") or len(mat) or 1))
        parts = []
        for start in range(0, len(mat), batch):
            proba, classes = adapter.predict(est, mat[start:start + batch])
            parts.append(self._to_full(proba, classes, b))
        return np.vstack(parts) if parts else np.zeros((0, N_CLASSES), dtype=np.float64)

    def _predict_bag(self, adapter: TFMAdapter, b: int, mats: list[np.ndarray], device: str,
                     weights: Mapping[str, Any]) -> list[np.ndarray]:
        """Predykcja jednego worka z drabiną awaryjną OOM (każdy krok → WARNING + tfm_events)."""
        est = self._fitted.pop(b, None)
        while True:
            phase = "fit"
            failure: str | None = None
            outs: list[np.ndarray] = []
            fit_s = predict_s = 0.0
            peak = None
            try:
                with adapter.library_settings(self.runtime):
                    _peak_reset()
                    if est is None:
                        self._make_room(adapter)
                        t0 = time.time()
                        idx = self.bags[b]
                        est = adapter.build(self.params, self.runtime, weights, device, self._seed(b))
                        adapter.fit(est, self.X_ctx[idx], self.y_ctx[idx].astype(np.int64))
                        fit_s = time.time() - t0
                        details = adapter.describe(est)
                        log.info(f"{adapter.display_name}: worek {b + 1}/{self.n_bags} — kontekst {len(idx):,} "
                                 f"wierszy, dopasowanie {fit_s:.1f} s" + (f", {details}" if details else ""))
                    phase = "predict"
                    t0 = time.time()
                    outs = [self._predict_rows(adapter, est, mat, b) for mat in mats]
                    predict_s = time.time() - t0
                    peak = _peak_mb()
            except Exception as exc:
                if not is_tfm_oom(exc):
                    self._release_est(adapter, est)
                    raise
                failure = f"{type(exc).__name__}: {_short(exc)}"
            if failure is None:
                rows = int(sum(len(m) for m in mats))
                previous = self.stats["bags"].get(b, {})
                self.stats["bags"][b] = {
                    "n_ctx": int(len(self.bags[b])),
                    "fit_s": round(fit_s, 3) if fit_s else previous.get("fit_s"),
                    "predict_s": round(predict_s, 3), "rows": rows,
                    "rows_per_s": round(rows / predict_s, 2) if predict_s > 0 else None,
                    "peak_vram_mb": peak,
                }
                self._keep_or_release(adapter, b, est)
                return outs
            # OOM — sprzątanie POZA blokiem except (ślad wyjątku trzyma tensory na karcie)
            step = adapter.next_oom_step(self.runtime, phase)
            if step is None or step[2]:
                self._release_est(adapter, est)
                est = None
            free_device_memory()
            if step is None:
                tried = [e["step"] for e in self.events if e.get("kind") == "oom_downgrade"]
                raise TFMOutOfMemoryError(
                    f"{adapter.display_name}: brak pamięci GPU (worek {b + 1}, faza {phase}) także po krokach "
                    f"awaryjnych: {', '.join(tried) or 'brak dostępnych'}. Zmniejsz: {adapter.oom_advice()} "
                    f"(np. --profile z mniejszym kontekstem). Ostatni błąd: {failure}")
            new_runtime, description, rebuild = step
            log.warning(f"{adapter.display_name}: BRAK PAMIĘCI GPU (worek {b + 1}, faza {phase}: {failure}) — "
                        f"krok awaryjny: {description}" + (" (ponowne dopasowanie worka)" if rebuild else ""))
            self.events.append({"kind": "oom_downgrade", "time": _now(), "bag": b, "phase": phase,
                                "step": description, "rebuild": bool(rebuild), "error": failure,
                                "runtime": dict(new_runtime)})
            self.runtime = dict(new_runtime)

    # ── redukcja worków (budżet finalnej predykcji) ──
    def limit_bags(self, n_bags: int, reason: str) -> None:
        """Zostawia pierwsze `n_bags` worków (kontekst innych usuwany z artefaktu)."""
        n_bags = max(1, int(n_bags))
        if n_bags >= self.n_bags:
            return
        removed = self.n_bags - n_bags
        for b in [k for k in self._fitted if k >= n_bags]:
            self._release_est(self._get_adapter(), self._fitted.pop(b))
        self.bags = self.bags[:n_bags]
        used = np.unique(np.concatenate(self.bags))
        remap = np.full(len(self.y_ctx), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        self.bags = [remap[b] for b in self.bags]
        self.X_ctx = np.ascontiguousarray(self.X_ctx[used])
        self.y_ctx = self.y_ctx[used]
        self.stats["bags"] = {b: v for b, v in self.stats.get("bags", {}).items() if b < n_bags}
        self.events.append({"kind": "bags_reduced", "time": _now(), "n_bags": n_bags, "removed": removed,
                            "detail": reason})


# ─────────────────────────────────────────
# Wrapper modelu (BaseTriageModel)
# ─────────────────────────────────────────
class TFMTriageModel(BaseTriageModel):
    """
    Model fundamentalny w interfejsie `BaseTriageModel` (TabPFN, TabICL — podklasy).

    `context_strata` — opcjonalne etykiety warstw (ESI × oddział, `train._site_strata`) POZYCYJNIE względem
    X przekazanego do `fit`; ustawia je wywołujący przed treningiem (brak → warstwy = ESI).
    `keep_fitted_bags` — ile dopasowanych estymatorów trzymać w pamięci między wywołaniami (1 / "all").
    """

    pretrained = True
    supports_early_stopping = False
    iteration_param = None
    defaults_name: str = ""          # nazwa słownika parametrów domyślnych w config (czytany przy tworzeniu)

    def __init__(self, params: dict[str, Any] | None = None):
        merged = {**dict(getattr(_config, self.defaults_name)), **(params or {})}
        if "device" not in merged:
            merged["device"] = "cuda" if device_for(self.name) == "gpu" else "cpu"
        super().__init__(params=merged)
        self._init_tfm_state()

    def _init_tfm_state(self) -> None:
        self.context_strata: np.ndarray | None = None
        self.tfm_events: list[dict[str, Any]] = []
        self.tfm_predict_s: float | None = None
        self._keep_default: int | str = 1

    # ── urządzenie ──
    def _uses_gpu(self) -> bool:
        return str(self.params.get("device", "cpu")).lower().startswith(("cuda", "gpu"))

    def _switch_to_cpu(self) -> None:
        """Powrót GPU → CPU tylko dla małego kontekstu (≤ TFM_CPU_MAX_CONTEXT); inaczej błąd."""
        limit = int(_config.TFM_CPU_MAX_CONTEXT)
        n_ctx = int(self.params.get("n_ctx") or 0)
        if n_ctx > limit:
            raise TFMDeviceError(
                f"{self.display_name}: błąd GPU — NIE przełączam na CPU: kontekst {n_ctx:,} wierszy > "
                f"TFM_CPU_MAX_CONTEXT={limit:,} (CPU wielokrotnie za wolne). Napraw GPU (zob. błąd powyżej) "
                f"albo zmniejsz n_ctx.")
        self.params["device"] = "cpu"

    def cpu_guard(self, max_context: int) -> None:
        """Brak GPU i kontekst worka > TFM_CPU_MAX_CONTEXT → TFMDeviceError z instrukcją (na starcie)."""
        check_cpu_context(self.name, "gpu" if self._uses_gpu() else "cpu", max_context)

    # ── stan ──
    @property
    def keep_fitted_bags(self) -> int | str:
        if isinstance(self.model, TFMPredictor):
            return self.model.keep_fitted_bags
        return getattr(self, "_keep_default", 1)

    @keep_fitted_bags.setter
    def keep_fitted_bags(self, value: int | str) -> None:
        self._keep_default = value
        if isinstance(self.model, TFMPredictor):
            self.model.keep_fitted_bags = value

    @property
    def n_bags(self) -> int:
        return self.model.n_bags if isinstance(self.model, TFMPredictor) else 0

    def _check_params(self) -> None:
        p = self.params
        if p.get("bag_mode") not in BAG_MODES:
            raise ValueError(f"{self.name}: bag_mode musi być jednym z {BAG_MODES}, podano {p.get('bag_mode')!r}")
        if p.get("context_strategy") not in CONTEXT_STRATEGIES:
            raise ValueError(f"{self.name}: context_strategy musi być jednym z {CONTEXT_STRATEGIES}, "
                             f"podano {p.get('context_strategy')!r}")
        if int(p.get("n_ctx") or 0) < 2 * N_CLASSES:
            raise ValueError(f"{self.name}: n_ctx musi być ≥ {2 * N_CLASSES}, podano {p.get('n_ctx')}")
        if int(p.get("max_bags") or 0) < 1:
            raise ValueError(f"{self.name}: max_bags musi być ≥ 1, podano {p.get('max_bags')}")

    # ── trening = zapis kontekstu ──
    def _fit_impl(self, X, y, X_es, y_es, sample_weight) -> None:
        self._check_params()
        if sample_weight is not None and len(sample_weight) and not np.allclose(sample_weight, sample_weight[0]):
            log.warning(f"{self.name}: wagi sampli są ignorowane (model fundamentalny uczy się z kontekstu)")
        y = np.asarray(y).astype(np.int64).ravel()
        strata = self.context_strata
        if strata is not None:
            strata = np.asarray(strata)
            if len(strata) != len(X):
                raise ValueError(f"{self.name}: context_strata ma {len(strata)} etykiet, a X {len(X)} wierszy "
                                 f"(etykiety są pozycyjne względem X)")
        kept, feature_info = select_features(list(X.columns), X, self.params.get("max_features"))
        if feature_info is not None:
            msg = (f"max_features={feature_info['max_features']}: {feature_info['n_in']} → {feature_info['n_kept']} "
                   f"cech (chronione {feature_info['n_protected']}, odrzucone {len(feature_info['dropped'])})")
            log.info(f"{self.display_name}: {msg}")
            self.tfm_events.append({"kind": "max_features", "time": _now(), "detail": msg})
        n_ctx = int(self.params["n_ctx"])
        n_bags = resolve_n_bags(self.params.get("n_bags", 1), len(X), n_ctx, int(self.params.get("max_bags") or 1))
        bags, bag_events = build_bags(
            y, strata, n_ctx=n_ctx, n_bags=n_bags, bag_mode=self.params["bag_mode"],
            context_strategy=self.params["context_strategy"], seed=self._base_seed())
        for event in bag_events:
            log.warning(f"{self.display_name}: {event['detail']}")
        self.tfm_events.extend(bag_events)
        max_ctx = max(len(b) for b in bags)
        self.cpu_guard(max_ctx)

        adapter = adapter_for(self.name)
        weights = adapter.resolve_weights(self.params)
        weights["override"] = adapter.weights_override()
        if weights["override"]:
            self.tfm_events.append({"kind": "weights_override", "time": _now(), "detail": weights["override"]})
        device = "cuda" if self._uses_gpu() else "cpu"
        runtime = adapter.initial_runtime(self.params, n_ctx=max_ctx, n_features=len(kept), device=device)

        used = np.unique(np.concatenate(bags))
        remap = np.full(len(y), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        # Tylko wiersze worków i zachowane cechy (jedna kopia; pełny TRAIN × 271 cech to ~0,5 GB)
        X_ctx = X.iloc[used, [X.columns.get_loc(c) for c in kept]].to_numpy(dtype=np.float32)
        self.model = TFMPredictor(
            library=self.name, params=self.params, feature_names=list(X.columns), kept_features=kept,
            X_ctx=X_ctx, y_ctx=y[used], bags=[remap[b] for b in bags], prior_fit=class_prior(y),
            correction=self.params["context_strategy"] == "balanced", weights=weights,
            library_version=adapter.library_version(), device=device, runtime=runtime, fake=adapter.fake,
            n_fit=len(y), feature_info=feature_info, events=self.tfm_events,
        )
        self.model.keep_fitted_bags = getattr(self, "_keep_default", 1)
        log.info(f"{self.display_name}: {n_bags} worek/worków ({self.params['bag_mode']}, "
                 f"{self.params['context_strategy']}), kontekst {self.model.context_sizes()} wierszy z {len(y):,}, "
                 f"{len(kept)} cech; wagi {weights.get('file')}"
                 + (f" [{weights['override']}]" if weights.get("override") else ""))

    def _base_seed(self) -> int:
        value = self.params.get("random_state")
        return RANDOM_SEED if value is None else int(value)

    def _after_fit(self, has_es: bool) -> None:
        self.classes_ = np.arange(N_CLASSES)
        self.actual_device = "cuda" if self._uses_gpu() else "cpu"
        self.best_iteration = None
        self.iteration_cap = None

    # ── predykcja ──
    def _predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_proba_many(self, frames: Sequence[pd.DataFrame]) -> list[np.ndarray]:
        """Wszystkie ramki w jednym przebiegu każdego worka (VAL + TEST bez ponownego budowania buforów)."""
        if not self.is_fitted:
            raise RuntimeError("Model nie wytrenowany.")
        t0 = time.time()
        out = self.model.predict_proba_many([self._align_features(f) for f in frames])
        self.tfm_predict_s = round(time.time() - t0, 3)
        return [full_class_proba(p, self.classes_) for p in out]

    def limit_bags(self, n_bags: int, reason: str) -> None:
        if isinstance(self.model, TFMPredictor):
            self.model.limit_bags(n_bags, reason)

    def release(self) -> None:
        if isinstance(self.model, TFMPredictor):
            self.model.release()

    def feature_importances(self) -> pd.DataFrame | None:
        return None

    # ── opis / logi ──
    def tfm_info(self) -> dict[str, Any]:
        """Blok `tfm` do run_info / DONE.json / artefaktu."""
        p = self.model if isinstance(self.model, TFMPredictor) else None
        if p is None:
            return {"library": self.name, "display_name": self.display_name}
        info = {
            "library": self.name,
            "display_name": self.display_name,
            "library_version": p.library_version,
            "torch_version": None if p.fake else _dist_version("torch"),
            "weights": {k: v for k, v in p.weights.items() if k != "override"},
            "weights_override": p.weights.get("override"),
            "licence": _config.TFM_LICENSES.get(self.name),
            "fake": p.fake,
            "device": p.device,
            "config": {k: v for k, v in self.params.items() if k != "device"},
            "n_bags": p.n_bags,
            "bag_mode": self.params.get("bag_mode"),
            "context_strategy": self.params.get("context_strategy"),
            "prior_correction": p.correction,
            "context_sizes": p.context_sizes(),
            "class_counts_per_bag": p.bag_class_counts(),
            "n_fit_rows": p.n_fit,
            "n_context_rows_stored": int(len(p.y_ctx)),
            "context_mb": round(p.X_ctx.nbytes / 1e6, 1),
            "n_features_in": len(p.feature_names),
            "n_features_used": len(p.kept_features),
            "feature_selection": p.feature_info,
            "runtime": dict(p.runtime),
            "predict_s": self.tfm_predict_s,
            "throughput": p.throughput(),
            "peak_vram_mb": (p.throughput() or {}).get("peak_vram_mb"),
            "events": list(self.tfm_events),
        }
        return info

    def _log_extra_details(self) -> None:
        p = self.model if isinstance(self.model, TFMPredictor) else None
        if p is None or self.train_logger is None:
            return
        self.train_logger.info(f"Biblioteka: {self.name} {p.library_version} | wagi: {p.weights.get('file')}"
                               + (f" | PRZEKIEROWANIE: {p.weights['override']}" if p.weights.get("override") else ""))
        self.train_logger.info(f"Worki: {p.n_bags} ({self.params.get('bag_mode')}, {self.params.get('context_strategy')}"
                               f"{', korekta priorów' if p.correction else ''}), kontekst: {p.context_sizes()}")
        for b, counts in enumerate(p.bag_class_counts()):
            self.train_logger.info(f"  worek {b}: klasy ESI 1–5 = {counts}")
        self.train_logger.info(f"Cechy: {len(p.kept_features)} z {len(p.feature_names)}; kontekst w artefakcie "
                               f"{len(p.y_ctx):,} wierszy ≈ {p.X_ctx.nbytes / 1e6:.1f} MB (przed kompresją)")
        if self.tracker is not None:
            self.tracker.add_note(f"TFM {self.name}: {p.n_bags} worków, kontekst {p.context_sizes()}, "
                                  f"wagi {p.weights.get('file')}")
            self.tracker.log_metrics({"tfm_n_bags": p.n_bags, "tfm_context_rows": int(len(p.y_ctx))})

    # ── persystencja ──
    def _artifact_extra(self) -> dict[str, Any]:
        return {"pretrained": True, "tfm": self.tfm_info(), "tfm_events": list(self.tfm_events)}

    def _restore_extra(self, artifact: dict[str, Any]) -> None:
        self._init_tfm_state()
        self.tfm_events = list(artifact.get("tfm_events") or [])
        if isinstance(self.model, TFMPredictor):
            self.model.events = self.tfm_events
            self._keep_default = self.model.keep_fitted_bags


# ─────────────────────────────────────────
# Pomocnicze dla train.py / orkiestratora
# ─────────────────────────────────────────
def check_cpu_context(model_name: str, device: str, max_context: int) -> None:
    """Brak GPU (device 'cpu') i kontekst > config.TFM_CPU_MAX_CONTEXT → TFMDeviceError z instrukcją."""
    limit = int(_config.TFM_CPU_MAX_CONTEXT)
    if str(device).lower().startswith(("cuda", "gpu")) or int(max_context) <= limit:
        return
    from src.utils.hardware import TORCH_CUDA_INDEX_URL

    raise TFMDeviceError(
        f"{model_name}: brak GPU (urządzenie 'cpu'), a kontekst worka to {int(max_context):,} wierszy > "
        f"TFM_CPU_MAX_CONTEXT={limit:,}. Na CPU predykcja trwałaby wiele godzin — przerywam na starcie. "
        f"Rozwiązania: (1) GPU NVIDIA z torch z CUDA (Windows: `pip install torch --index-url "
        f"{TORCH_CUDA_INDEX_URL}`, potem `python scripts/13_tfm_setup.py`); (2) profil smoke "
        f"(`--profile smoke`, kontekst 1000); (3) mniejszy kontekst (n_ctx ≤ {limit:,}). "
        f"Wymuszenie urządzenia: SOR_AI_DEVICE_{model_name.upper()}=gpu.")


def planned_max_context(model_name: str, budget: Mapping[str, Any], params: Mapping[str, Any] | None,
                        tune: bool) -> int:
    """
    Największy kontekst worka planowany przez run (przed wczytaniem danych): kandydaci wyboru konfiguracji
    (przy tune) albo n_ctx z parametrów, przycięte do train_sample profilu, gdy ten jest ustalony.
    """
    defaults = dict(getattr(_config, f"{model_name.upper()}_DEFAULT_PARAMS", {}))
    merged = {**defaults, **(params or {})}
    sizes = [int(merged.get("n_ctx") or 0)]
    if tune:
        n_trials = int(budget.get("n_trials") or 0) or None
        candidates = list(budget.get("candidates") or [])
        sizes = [int({**merged, **c}.get("n_ctx") or 0) for c in candidates[:n_trials]] or sizes
    largest = max(sizes)
    sample = budget.get("train_sample")
    return int(min(largest, int(sample))) if sample else int(largest)


def weights_override_active(model_name: str) -> str | None:
    """Aktywny hak testowy wag ('SOR_AI_TFM_FAKE=1' / 'SOR_AI_TABPFN_MODEL_PATH=…') albo None (preflight)."""
    if fake_mode():
        return f"{FAKE_ENV}=1"
    var = {"tabpfn": "SOR_AI_TABPFN_MODEL_PATH", "tabicl": "SOR_AI_TABICL_MODEL_PATH"}.get(model_name)
    raw = os.environ.get(var or "", "").strip() if var else ""
    return f"{var}={raw}" if raw else None


__all__ = [
    "FAKE_ENV",
    "TFMAdapter",
    "TFMDeviceError",
    "TFMOutOfMemoryError",
    "TFMPredictor",
    "TFMTriageModel",
    "TFMWeightsError",
    "adapter_for",
    "build_bags",
    "check_cpu_context",
    "fake_mode",
    "free_device_memory",
    "is_tfm_oom",
    "planned_max_context",
    "resolve_n_bags",
    "select_features",
    "weights_identity",
    "weights_override_active",
]
