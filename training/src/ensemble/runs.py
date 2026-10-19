"""Wczytywanie runów i operacje na prawdopodobieństwach ESI (łączenie, kalibracja, różnorodność)."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import log_loss

from src.evaluation.metrics import decide, full_evaluation
from src.utils.config import CLASS_NAMES, N_CLASSES, RUNS_DIR

PROBA_COLS = [f"p_{name.lower()}" for name in CLASS_NAMES]
_EPS = 1e-12


@dataclass
class Run:
    name: str
    path: Path
    info: dict

    @property
    def model(self) -> str:
        return self.info["model"]

    def predictions(self, split: str) -> pd.DataFrame:
        path = self.path / f"predictions_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Brak {path}")
        return pd.read_parquet(path)


# ─────────────────────────────────────────
# Runy
# ─────────────────────────────────────────
def load_run(run: str | Path, runs_dir: Path = RUNS_DIR) -> Run:
    path = Path(run)
    if not path.is_absolute() and not path.exists():
        path = runs_dir / str(run)
    info_path = path / "run_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{path} nie jest katalogiem runa (brak run_info.json)")
    return Run(name=path.name, path=path, info=json.loads(info_path.read_text()))


def list_runs(runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    """Wszystkie runy z podstawowymi informacjami (najnowsze na końcu)."""
    rows = []
    for info_path in sorted(Path(runs_dir).glob("*/run_info.json")):
        info = json.loads(info_path.read_text())
        rows.append({
            "run": info_path.parent.name,
            "model": info.get("model"),
            "feature_set": info.get("feature_set"),
            "tuned": info.get("tuned"),
            "weights": info.get("sample_weight_strategy"),
            "n_fit": info.get("n_fit"),
            "n_val": info.get("n_val"),
            "train_sample": info.get("train_sample"),
            "eval_sample": info.get("eval_sample"),
        })
    return pd.DataFrame(rows)


def latest_runs(
    feature_set: str = "triage",
    models: list[str] | None = None,
    include_samples: bool = False,
    runs_dir: Path = RUNS_DIR,
) -> list[Run]:
    """
    Najnowszy run każdego modelu dla danego zestawu cech.

    Runy na próbkach (--sample / --eval-sample) są pomijane, chyba że include_samples=True.
    """
    table = list_runs(runs_dir)
    if table.empty:
        return []
    table = table[table["feature_set"] == feature_set]
    if not include_samples:
        table = table[table["train_sample"].isna() & table["eval_sample"].isna()]
    if models is not None:
        table = table[table["model"].isin(models)]
    latest = table.groupby("model", sort=False).tail(1)
    return [load_run(r, runs_dir) for r in latest["run"]]


def aligned_probas(runs: list[Run], split: str = "val") -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Prawdopodobieństwa wszystkich runów na tym samym zbiorze wierszy.

    Returns
    -------
    y : np.ndarray (n,) — prawdziwe klasy 0–4
    probas : dict nazwa → np.ndarray (n, 5); nazwa = model (albo nazwa runa przy duplikatach)
    """
    if not runs:
        raise ValueError("Brak runów do porównania")
    frames = {}
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
    first = next(iter(frames.values()))
    y = first["target"].to_numpy()
    probas = {k: _normalize(df[PROBA_COLS].to_numpy(dtype=float)) for k, df in frames.items()}
    return y, probas


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
    """
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
    return pd.DataFrame(rows)
