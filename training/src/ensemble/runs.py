"""Wczytywanie runów i operacje na prawdopodobieństwach ESI (łączenie, kalibracja, różnorodność)."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import log_loss

from src.data.dataset import FEATURE_SET_ALIASES
from src.evaluation.metrics import decide, full_evaluation
from src.utils import config
from src.utils.config import CLASS_NAMES, N_CLASSES
from src.utils.logger import get_logger

log = get_logger(__name__)

PROBA_COLS = [f"p_{name.lower()}" for name in CLASS_NAMES]
VITALS_COMPLETE_COL = "vitals_complete"     # = src.data.prepare.VITALS_COMPLETE_COL
# Kolumny meta w predykcjach (bez prawdopodobieństw), które `aligned_meta` zwraca, jeśli są
META_COLUMNS = ("site", "esi", "target", VITALS_COMPLETE_COL, "admitted")
DONE_JSON = "DONE.json"                     # = src.models.train.DONE_JSON (run ukończony)
_EPS = 1e-12


def _runs_dir(runs_dir: Path | str | None) -> Path:
    """Katalog runów rozwiązywany przy WYWOŁANIU (config.RUNS_DIR może być nadpisany zmienną/testem)."""
    return Path(runs_dir) if runs_dir is not None else Path(config.RUNS_DIR)


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_feature_set(name: str) -> str:
    """Stare nazwy zestawów cech ('triage_only', 'top') → 'triage'."""
    return FEATURE_SET_ALIASES.get(name, name)


def _is_weighted(weights: Any) -> bool:
    """Run trenowany z wagami klas (prawdopodobieństwa nieskalibrowane — nie do domyślnego łączenia)."""
    return weights not in (None, "none") and not (isinstance(weights, float) and np.isnan(weights))


def _is_tuned(tuned: Any) -> bool:
    """Run po tuningu Optuną (brak informacji w run_info.json = bez tuningu)."""
    return isinstance(tuned, (bool, np.bool_)) and bool(tuned)


@dataclass
class Run:
    name: str
    path: Path
    info: dict

    @property
    def model(self) -> str:
        return self.info["model"]

    @property
    def done(self) -> bool:
        return (self.path / DONE_JSON).exists()

    @property
    def has_model(self) -> bool:
        return (self.path / "model.joblib").exists()

    def predictions(self, split: str) -> pd.DataFrame:
        path = self.path / f"predictions_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Brak {path}")
        return pd.read_parquet(path)


# ─────────────────────────────────────────
# Runy
# ─────────────────────────────────────────
def load_run(run: str | Path, runs_dir: Path | str | None = None) -> Run:
    """Run po nazwie katalogu w runs/ albo po ścieżce."""
    path = Path(run)
    if not path.is_absolute() and not path.exists():
        path = _runs_dir(runs_dir) / str(run)
    info_path = path / "run_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{path} nie jest katalogiem runa (brak run_info.json)")
    return Run(name=path.name, path=path, info=_read_json(info_path))


def list_runs(runs_dir: Path | str | None = None) -> pd.DataFrame:
    """Wszystkie runy z podstawowymi informacjami (najnowsze na końcu)."""
    rows = []
    for info_path in sorted(_runs_dir(runs_dir).glob("*/run_info.json")):
        try:
            info = _read_json(info_path)
        except (OSError, ValueError) as exc:
            log.warning(f"Pomijam uszkodzony {info_path}: {exc}")
            continue
        run_dir = info_path.parent
        rows.append({
            "run": run_dir.name,
            "model": info.get("model"),
            "feature_set": resolve_feature_set(str(info.get("feature_set"))),
            "tag": info.get("tag"),
            "profile": info.get("profile"),
            "tuned": info.get("tuned"),
            "weights": info.get("weights_strategy") or info.get("sample_weight_strategy"),
            "data_version": info.get("data_version"),
            "data_fingerprint": info.get("data_fingerprint"),
            "n_fit": info.get("n_fit"),
            "n_val": info.get("n_val"),
            "train_sample": info.get("train_sample"),
            "eval_sample": info.get("eval_sample"),
            "done": (run_dir / DONE_JSON).exists(),
            "has_model": (run_dir / "model.joblib").exists(),
        })
    return pd.DataFrame(rows)


def latest_runs(
    feature_set: str = "triage",
    models: list[str] | None = None,
    include_samples: bool = False,
    runs_dir: Path | str | None = None,
    *,
    tag: str | None = None,
    require_model: bool = True,
    require_done: bool = True,
    include_weighted: bool = False,
    data_version: str | None = "current",
    require_tuned: bool = True,
) -> list[Run]:
    """
    Najnowszy run każdego modelu dla zestawu cech — kandydaci do porównania i łączenia.

    Domyślnie pomijane są runy:
        - na próbkach (--sample / --eval-sample)       → include_samples=True,
        - nieukończone (bez DONE.json)                  → require_done=False,
        - bez pliku modelu (--no-save)                  → require_model=False,
        - z wagami klas (nieskalibrowane prawdopodobieństwa; z notą w logu) → include_weighted=True,
        - bez tuningu (parametry domyślne, np. tag '<profil>-notune'; z notą w logu)
                                                        → require_tuned=False,
        - z innej wersji danych niż bieżąca DATA_VERSION → data_version=None (dowolna) albo nazwa.
    tag : tylko runy z tym tagiem (np. nazwa profilu 'standard').
    Jawna lista runów (`load_run`, `07_compare_runs.py --runs`) nie podlega tym filtrom.
    """
    table = list_runs(runs_dir)
    if table.empty:
        return []
    table = table[table["feature_set"] == resolve_feature_set(feature_set)]
    if not include_samples:
        table = table[table["train_sample"].isna() & table["eval_sample"].isna()]
    if require_done:
        table = table[table["done"]]
    if require_model:
        table = table[table["has_model"]]
    if tag is not None:
        table = table[table["tag"] == tag]
    if data_version is not None:
        wanted = config.DATA_VERSION if data_version == "current" else data_version
        table = table[table["data_version"] == wanted]
    if models is not None:
        table = table[table["model"].isin(models)]
    if not include_weighted and not table.empty:
        weighted = table[table["weights"].map(_is_weighted)]
        if not weighted.empty:
            log.info("Pominięto runy z wagami klas (prawdopodobieństwa nieskalibrowane — nie do domyślnego "
                     "łączenia; include_weighted=True je uwzględni): " + ", ".join(weighted["run"]))
            table = table[~table["run"].isin(weighted["run"])]
    if require_tuned and not table.empty:
        untuned = table[~table["tuned"].map(_is_tuned)]
        if not untuned.empty:
            log.info("Pominięto runy bez tuningu (parametry domyślne; require_tuned=False / "
                     "--include-untuned je uwzględni): " + ", ".join(untuned["run"]))
            table = table[~table["run"].isin(untuned["run"])]
    latest = table.groupby("model", sort=False).tail(1)
    return [load_run(r, runs_dir) for r in latest["run"]]


def aligned_probas(
    runs: list[Run],
    split: str = "val",
    return_vc: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]:
    """
    Prawdopodobieństwa wszystkich runów na tym samym zbiorze wierszy (po `row_id`).

    Sprawdza: ten sam zbiór wierszy, identyczną etykietę dla każdego `row_id`, tę samą wersję
    danych we wszystkich runach (inaczej ValueError — łącz tylko runy z jednego przygotowania)
    i poprawność prawdopodobieństw (skończone, nieujemne, dodatnia suma wiersza — inaczej
    ValueError z nazwą runa i liczbą złych wierszy).

    Returns
    -------
    y : np.ndarray (n,) — prawdziwe klasy 0–4
    probas : dict nazwa → np.ndarray (n, 5); nazwa = model (albo nazwa runa przy duplikatach)
    vc_mask : (tylko return_vc=True) maska `vitals_complete == 1` albo None, gdy predykcje nie
              mają tej kolumny (stare runy)
    """
    if not runs:
        raise ValueError("Brak runów do porównania")
    versions = {r.info.get("data_version") for r in runs if r.info.get("data_version")}
    if len(versions) > 1:
        raise ValueError(f"Runy z różnych wersji danych ({sorted(versions)}) — łącz tylko runy z tego samego "
                         f"przygotowania danych")
    fingerprints = {r.info.get("data_fingerprint") for r in runs if r.info.get("data_fingerprint")}
    if len(fingerprints) > 1:
        log.warning(f"Runy z różnych przygotowań danych (odciski {sorted(fingerprints)}) — sprawdzam etykiety "
                    f"wiersz po wierszu")

    frames: dict[str, pd.DataFrame] = {}
    for run in runs:
        key = run.model if sum(r.model == run.model for r in runs) == 1 else run.name
        frames[key] = run.predictions(split).set_index("row_id").sort_index()
    ids = [set(df.index) for df in frames.values()]
    common = set.intersection(*ids)
    if any(len(i) != len(common) for i in ids):
        raise ValueError(
            "Runy mają różne zbiory wierszy (np. inne --eval-sample albo inny podział danych). "
            "Łącz tylko runy z tego samego przygotowania danych."
        )
    first_key, first = next(iter(frames.items()))
    y = first["target"].to_numpy()
    for key, df in frames.items():
        if not np.array_equal(df["target"].to_numpy(), y):
            raise ValueError(f"Run '{key}' ma inne etykiety niż '{first_key}' dla tych samych row_id — "
                             f"runy pochodzą z różnych przygotowań danych")
    probas = {k: _normalize(_checked_proba(df, k, split)) for k, df in frames.items()}
    if not return_vc:
        return y, probas
    vc_mask = None
    if VITALS_COMPLETE_COL in first.columns:
        values = pd.to_numeric(first[VITALS_COMPLETE_COL], errors="coerce")
        if values.notna().any():
            vc_mask = values.fillna(0).to_numpy() == 1
    return y, probas, vc_mask


def aligned_meta(runs: list[Run], split: str = "val") -> pd.DataFrame:
    """
    Kolumny meta splitu (oddział, ESI, target, vitals_complete, admitted — te, które są
    w predykcjach) z indeksem `row_id` w KOLEJNOŚCI `aligned_probas` (rosnące row_id).

    Wywołuj po `aligned_probas` (ta sprawdza, że wszystkie runy mają te same wiersze i etykiety).
    """
    if not runs:
        raise ValueError("Brak runów")
    df = runs[0].predictions(split).set_index("row_id").sort_index()
    return df[[c for c in META_COLUMNS if c in df.columns]].copy()


def _checked_proba(df: pd.DataFrame, key: str, split: str) -> np.ndarray:
    """p_esi1..p_esi5 runa; NaN / nieskończoność / wartości ujemne / zerowa suma wiersza → ValueError."""
    proba = df[PROBA_COLS].to_numpy(dtype=float)
    bad = ~np.isfinite(proba).all(axis=1) | (proba < 0).any(axis=1)
    bad |= ~(np.where(bad, 1.0, proba.sum(axis=1)) > 0)
    if bad.any():
        examples = ", ".join(str(i) for i in df.index[bad][:5])
        raise ValueError(f"Run '{key}' ({split}): {int(bad.sum()):,} z {len(proba):,} wierszy ma prawdopodobieństwa "
                         f"NaN / nieskończone / ujemne albo o zerowej sumie (row_id: {examples}) — popraw predykcje")
    return proba


def _normalize(proba: np.ndarray) -> np.ndarray:
    """Predykcje zapisane jako float32 — przywróć dokładną sumę 1 w każdym wierszu."""
    return proba / proba.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────
# Łączenie i ocena
# ─────────────────────────────────────────
def blend(
    probas: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
    method: str = "mean",
) -> np.ndarray:
    """
    Ważone łączenie prawdopodobieństw.

    method='mean'      — średnia arytmetyczna (domyślnie)
    method='geometric' — średnia geometryczna (log-prob), potem normalizacja
    Brak wag = równe wagi. Wagi są normalizowane do sumy 1.
    """
    names = list(probas)
    w = np.array([1.0 if weights is None else float(weights.get(n, 0.0)) for n in names])
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("Wagi muszą być nieujemne i niezerowe")
    w = w / w.sum()
    stack = np.stack([probas[n] for n in names])            # (m, n, 5)
    if method == "mean":
        out = np.tensordot(w, stack, axes=1)
    elif method == "geometric":
        out = np.exp(np.tensordot(w, np.log(np.clip(stack, _EPS, 1.0)), axes=1))
    else:
        raise ValueError(f"Nieznana metoda łączenia '{method}' (mean | geometric)")
    return out / out.sum(axis=1, keepdims=True)


def evaluate(y: np.ndarray, proba: np.ndarray, rule: str = "argmax") -> dict:
    """Pełne metryki (bez wypisywania) dla danej reguły decyzyjnej."""
    return full_evaluation(y, decide(proba, rule), proba, print_report=False)


def optimize_weights(
    probas: dict[str, np.ndarray],
    y: np.ndarray,
    method: str = "mean",
) -> dict[str, float]:
    """
    Wagi minimalizujące log loss na podanym zbiorze (używaj WYŁĄCZNIE val).

    To podpowiedź, nie wyrocznia — porównaj z wagami ręcznymi i równymi.
    """
    names = list(probas)

    def loss(raw: np.ndarray) -> float:
        w = np.abs(raw) / np.abs(raw).sum()
        return log_loss(y, blend(probas, dict(zip(names, w)), method), labels=list(range(N_CLASSES)))

    x0 = np.full(len(names), 1.0 / len(names))
    res = minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 400 * len(names), "xatol": 1e-4})
    w = np.abs(res.x) / np.abs(res.x).sum()
    return {n: round(float(v), 4) for n, v in zip(names, w)}


# ─────────────────────────────────────────
# Kalibracja (temperature scaling)
# ─────────────────────────────────────────
def fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    """
    Temperatura T minimalizująca log loss: p ∝ exp(log p / T).

    T > 1 spłaszcza zbyt pewne prawdopodobieństwa, T < 1 je wyostrza.
    Dopasuj na val, zastosuj tę samą T na test.
    """
    logp = np.log(np.clip(proba, _EPS, 1.0))

    def loss(log_t: float) -> float:
        return log_loss(y, apply_temperature(proba, float(np.exp(log_t)), logp=logp),
                        labels=list(range(N_CLASSES)))

    res = minimize_scalar(loss, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(proba: np.ndarray, temperature: float, logp: np.ndarray | None = None) -> np.ndarray:
    logp = np.log(np.clip(proba, _EPS, 1.0)) if logp is None else logp
    z = logp / temperature
    z = z - z.max(axis=1, keepdims=True)
    out = np.exp(z)
    return out / out.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────
# Różnorodność modeli
# ─────────────────────────────────────────
def diversity(probas: dict[str, np.ndarray], y: np.ndarray) -> pd.DataFrame:
    """
    Para po parze: jak różne są modele (im bardziej różne przy podobnej jakości,
    tym więcej daje ich połączenie).

    disagreement  — odsetek wierszy z różną predykcją argmax
    corr_expected — korelacja oczekiwanego poziomu ESI (Σ p·k)
    corr_errors   — korelacja wskaźników błędu argmax (1 = myli się na tych samych wierszach)

    Mniej niż 2 modele → pusta tabela z tymi samymi kolumnami.
    """
    columns = ["model_a", "model_b", "disagreement", "corr_expected", "corr_errors"]
    levels = np.arange(N_CLASSES)
    preds = {k: p.argmax(axis=1) for k, p in probas.items()}
    expected = {k: p @ levels for k, p in probas.items()}
    errors = {k: (preds[k] != y).astype(float) for k in probas}
    rows = []
    for a, b in itertools.combinations(probas, 2):
        rows.append({
            "model_a": a,
            "model_b": b,
            "disagreement": float((preds[a] != preds[b]).mean()),
            "corr_expected": float(np.corrcoef(expected[a], expected[b])[0, 1]),
            "corr_errors": float(np.corrcoef(errors[a], errors[b])[0, 1]),
        })
    return pd.DataFrame(rows, columns=columns)
