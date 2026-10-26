"""
C9 — krzywa uczenia: refit z najlepszymi parametrami na ułamkach zbioru dopasowania X_fit
(domyślnie 5, 10, 25, 50 i 100%) → metryki na val + dopasowanie potęgowe (uzasadnienie liczebności,
TRIPOD+AI pkt 10).

Jednostka = (model, ułamek): refit `runs/<czas>_<model>_<cechy>_<tag>-lc<ułamek>/` — stratyfikowany
podzbiór X_fit (po wydzieleniu 10% TRAIN na early stopping; zbiór ES bez zmian), parametry
liczności (min. wielkość liścia) skalowane do nowej liczby wierszy jak w treningu finalnym, seed
modelu jak w runie źródłowym (punkt 100% odtwarza run źródłowy). Predykcje tylko na val.

Dopasowanie: metryka(n) = a + b · n^(−c) (c > 0; scipy.optimize.curve_fit) na punktach krzywej;
ekstrapolacja do X_fit całego TRAIN (bez próbki) i do 2× tej liczby — ekstrapolacja poza zakres
danych, opisuj ostrożnie. Wykres: learning_curve.png.

Wynik: runs/analyses/learning_curve__<cechy>__<tag>/results.json
    {"models": {model: {"points": [{"fraction", "n_fit", "run", "val": {...}}], "power_law": {...}}},
     "fractions": [...], "metrics": [...], ...}
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.common import (
    METRICS_VERSION,
    UnitStore,
    admitted_for,
    analysis_dir,
    current_source_names,
    default_tag,
    evaluate,
    execute_units,
    find_refit_run,
    load_predictions,
    merge_units,
    now_iso,
    read_json,
    require_source_runs,
    source_seed,
    vc_mask,
)
from src.utils import config
from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_FRACTIONS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0)
DEFAULT_MODELS: tuple[str, ...] = ("lightgbm", "xgboost")
RULES: tuple[str, ...] = ("argmax", "cost")
_BASE = ("qwk", "undertriage", "critical_miss", "recall_esi1")
METRICS: list[str] = [f"{m}_{r}" for r in RULES for m in _BASE] + ["log_loss", "auc_macro", "admission_auc"]
METRICS = METRICS + [f"vc_{m}" for m in METRICS]
# Metryki z dopasowaniem potęgowym (log loss maleje, QWK / AUC rosną z liczbą wierszy)
FIT_METRICS: tuple[str, ...] = ("log_loss", "qwk_argmax", "qwk_cost", "auc_macro", "undertriage_cost")
# Wykładnik c: typowe krzywe uczenia mają c ≈ 0,1–1; c na granicy przedziału = dopasowanie niewiarygodne
C_BOUNDS: tuple[float, float] = (0.05, 2.0)
C_FIXED: float = 0.5           # przy 3 punktach (3 parametry = brak stopni swobody) c ustalone
# Kierunek metryk dopasowania: True = im więcej, tym lepiej (QWK, AUC); False = im mniej (log loss, undertriage)
HIGHER_IS_BETTER: dict[str, bool] = {"log_loss": False, "qwk_argmax": True, "qwk_cost": True, "auc_macro": True,
                                     "undertriage_cost": False}
# Kontrola wiarygodności ekstrapolacji (poza c ustalonym / na granicy):
MAX_EXTRAPOLATION_FACTOR: float = 10.0   # cel > 10 × największe n z krzywej → za daleko poza danymi
MAX_RMSE_FRACTION: float = 0.25          # RMSE > 25% zakresu obserwowanych wartości → złe dopasowanie


def fraction_label(fraction: float) -> str:
    """0.05 → '0.05', 1.0 → '1' (tag refitu '<tag>-lc0.05', jednostka 'lightgbm/f0.05')."""
    return f"{float(fraction):g}"


def parse_fractions(spec: str | Sequence[float]) -> list[float]:
    """'0.05,0.1,1' albo lista → posortowane, unikalne ułamki z (0, 1]."""
    values = [float(v) for v in (spec.split(",") if isinstance(spec, str) else spec) if str(v).strip()]
    bad = [v for v in values if not 0 < v <= 1]
    if bad or not values:
        raise ValueError(f"Ułamki muszą należeć do (0, 1], podano: {spec}")
    return sorted(set(values))


# ─────────────────────────────────────────
# Dopasowanie potęgowe
# ─────────────────────────────────────────
def _power(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a + b * np.power(n, -c)


def fit_power_law(n: Sequence[float], values: Sequence[float | None],
                  extrapolate_to: Sequence[float] = (), higher_is_better: bool | None = None) -> dict[str, Any] | None:
    """
    metryka(n) = a + b·n^(−c). ≥ 4 punkty: a, b, c dopasowane (c ∈ C_BOUNDS); 3 punkty: c = C_FIXED,
    dopasowane tylko a, b. None, gdy < 3 skończonych punktów albo dopasowanie się nie zbiega.
    `a` = asymptota (nieskończenie wiele danych); `predicted` — wartości w `extrapolate_to`.
    `reliable` = False (ekstrapolacji nie raportuj), a `reasons` mówi dlaczego, gdy:
        c ustalone (3 punkty) albo na granicy przedziału; krzywa idzie w złą stronę (wg
        `higher_is_better`: b < 0 dla QWK/AUC, b > 0 dla log loss); asymptota albo ekstrapolacja
        gorsza od OSTATNIEGO obserwowanego punktu o więcej niż RMSE; RMSE > MAX_RMSE_FRACTION zakresu
        obserwacji; cel ekstrapolacji > MAX_EXTRAPOLATION_FACTOR × największe n z krzywej.
    """
    from scipy.optimize import curve_fit

    pts = sorted((float(x), float(v)) for x, v in zip(n, values, strict=True)
                 if v is not None and math.isfinite(float(v)) and x and x > 0)
    if len(pts) < 3 or len({x for x, _ in pts}) < 3:
        return None
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    free_c = len({x for x, _ in pts}) >= 4
    # Start / wersja z ustalonym c: a, b z regresji liniowej y na n^(−c)
    b0, a0 = np.polyfit(x ** -C_FIXED, y, 1)
    a, b, c = float(a0), float(b0), C_FIXED
    if free_c:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                (a, b, c), _ = curve_fit(_power, x, y, p0=(a0, b0, C_FIXED),
                                         bounds=([-np.inf, -np.inf, C_BOUNDS[0]], [np.inf, np.inf, C_BOUNDS[1]]),
                                         maxfev=20000)
        except (RuntimeError, ValueError) as exc:
            log.warning(f"Dopasowanie potęgowe nieudane: {exc}")
            return None
    at_bound = free_c and (c <= C_BOUNDS[0] * 1.01 or c >= C_BOUNDS[1] * 0.99)
    fitted = _power(x, a, b, c)
    rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
    predicted = {str(int(t)): float(_power(np.array([t], dtype=float), a, b, c)[0]) for t in extrapolate_to}
    reasons: list[str] = []
    if not free_c:
        reasons.append("3 punkty, c ustalone")
    if at_bound:
        reasons.append("c na granicy przedziału")
    if higher_is_better is not None:
        if (b >= 0) if higher_is_better else (b <= 0):
            reasons.append("krzywa pogarsza się z liczbą danych")
        last, tol = float(y[-1]), max(rmse, 1e-12)

        def worse(v: float) -> bool:
            return v < last - tol if higher_is_better else v > last + tol

        beyond = [v for t, v in zip(extrapolate_to, predicted.values(), strict=True) if float(t) >= x.max()]
        if worse(float(a)) or any(worse(v) for v in beyond):
            reasons.append("asymptota / ekstrapolacja gorsza od ostatniego punktu")
    spread = float(y.max() - y.min())
    if spread > 0 and rmse > MAX_RMSE_FRACTION * spread:
        reasons.append(f"RMSE > {MAX_RMSE_FRACTION:.0%} zakresu punktów")
    if extrapolate_to and max(float(t) for t in extrapolate_to) > MAX_EXTRAPOLATION_FACTOR * float(x.max()):
        reasons.append(f"ekstrapolacja > {MAX_EXTRAPOLATION_FACTOR:g}× największego n krzywej")
    return {
        "a": float(a), "b": float(b), "c": float(c),
        "c_fixed": not free_c,
        "at_bound": bool(at_bound),
        "reliable": not reasons,
        "reasons": reasons,
        "rmse": rmse,
        "n_points": len(pts),
        "n_range": [float(x.min()), float(x.max())],
        "predicted": predicted,
    }


def full_fit_rows(uses_es: bool) -> int | None:
    """Liczba wierszy X_fit przy CAŁYM TRAIN (manifest; z modelem z early stopping — 90%)."""
    from src.data.dataset import manifest

    try:
        n_train = int((manifest().get("n_rows") or {}).get("train") or 0)
    except (OSError, ValueError, TypeError):
        return None
    if n_train <= 0:
        return None
    return int(round(n_train * (1 - config.EARLY_STOPPING_FRACTION))) if uses_es else n_train


# ─────────────────────────────────────────
# Jednostki
# ─────────────────────────────────────────
def unit_metrics(run_dir: Path | str) -> dict[str, Any]:
    from src.evaluation.metrics import decide

    run_dir = Path(run_dir)
    info = read_json(run_dir / "run_info.json") or {}
    y, proba, frame = load_predictions(run_dir, "val")
    preds = {rule: decide(proba, rule) for rule in RULES}
    return {
        "n_fit": info.get("n_fit"),
        "best_iteration": info.get("best_iteration"),
        "training_duration_s": info.get("training_duration_s"),
        "val": evaluate(y, proba, preds, vc_mask(frame), admitted_for(frame["row_id"].to_numpy(), ("val",))),
    }


def plot_learning_curve(results: Mapping[str, Any], path: Path) -> Path | None:
    """Log loss i QWK argmax (val) vs liczba wierszy dopasowania, z krzywą potęgową."""
    from src.evaluation import visualizations  # noqa: F401 — backend Agg poza notebookiem
    import matplotlib.pyplot as plt

    models = results.get("models") or {}
    if not models:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in ((axes[0], "log_loss", "log loss (val) ↓"), (axes[1], "qwk_argmax", "QWK argmax (val) ↑")):
        for model, entry in models.items():
            pts = [p for p in entry.get("points", []) if (p.get("val") or {}).get(metric) is not None]
            if not pts:
                continue
            x = np.array([p["n_fit"] for p in pts], dtype=float)
            y = np.array([p["val"][metric] for p in pts], dtype=float)
            line = ax.plot(x, y, "o-", label=model)[0]
            fit = (entry.get("power_law") or {}).get(metric)
            if fit:
                hi = max([x.max(), *[float(k) for k in fit.get("predicted", {})]])
                grid = np.geomspace(x.min(), hi, 100)
                ax.plot(grid, _power(grid, fit["a"], fit["b"], fit["c"]), "--", color=line.get_color(), alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("wiersze dopasowania (X_fit)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"Krzywa uczenia — {results.get('feature_set')} · tag {results.get('tag')} "
                 f"(przerywana: a + b·n^(−c))", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def summary_lines(results: Mapping[str, Any]) -> list[str]:
    lines = [f"# Krzywa uczenia (C9) — {results['feature_set']} · tag {results['tag']}", "",
             f"- ułamki X_fit: {results['fractions']}; jednostki {results['n_done']}/{results['n_units']}"
             + (f", błędy: {len(results['failures'])}" if results["failures"] else ""), ""]
    for model, entry in results["models"].items():
        lines += [f"## {model}", "", "| ułamek | n_fit | log loss | QWK argmax | QWK koszt | undertriage koszt | AUC |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        for p in entry["points"]:
            v = p.get("val") or {}

            def f(key: str, val: dict = v) -> str:
                return "—" if val.get(key) is None else f"{val[key]:.4f}"

            lines.append(f"| {p['fraction']:g} | {p['n_fit']:,} | {f('log_loss')} | {f('qwk_argmax')} | "
                         f"{f('qwk_cost')} | {f('undertriage_cost')} | {f('auc_macro')} |")
        for metric, fit in (entry.get("power_law") or {}).items():
            if fit:
                pred = ", ".join(f"n={int(k):,}: {val:.4f}" for k, val in fit["predicted"].items())
                note = "" if fit.get("reliable") else (
                    " — **niewiarygodne** (" + "; ".join(fit.get("reasons") or ["c ustalone albo na granicy"])
                    + "; nie raportuj ekstrapolacji)")
                lines.append(f"\n- {metric}: a + b·n^(−c), a = {fit['a']:.4f}, c = {fit['c']:.3f} "
                             f"(RMSE {fit['rmse']:.4f}); ekstrapolacja: {pred}{note}")
        lines.append("")
    return lines


def run_learning_curve(
    feature_set: str = "triage",
    tag: str | None = None,
    models: Sequence[str] | None = DEFAULT_MODELS,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    seed: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Refity na ułamkach X_fit (wznawialne) → results.json, summary.md, learning_curve.png."""
    from src.data.dataset import data_fingerprint
    from src.models import MODEL_REGISTRY
    from src.models.train import REFIT_VERSION, refit_from_run, run_duration_s

    tag = tag or default_tag()
    fractions = parse_fractions(fractions)
    runs = require_source_runs(feature_set, tag, models)
    store = UnitStore(analysis_dir("learning_curve", feature_set, tag))
    fingerprint = data_fingerprint()
    units = [{"unit": f"{model}/f{fraction_label(fr)}", "model": model, "fraction": fr, "source_run": run.name,
              "seed": int(seed) if seed is not None else source_seed(run.info)}
             for model, run in runs.items() for fr in fractions]
    # Wyniki kumulatywne: modele policzone wcześniej (ten sam run źródłowy) zostają w results.json
    all_units = merge_units(store.planned_units(), units, lambda u: u["model"],
                            current_source_names(feature_set, tag))
    store.write_plan({"analysis": "learning_curve", "feature_set": feature_set, "tag": tag,
                      "fractions": fractions, "units": all_units})

    def expect(u: Mapping[str, Any]) -> dict[str, Any]:
        return {"source_run": u["source_run"], "data_fingerprint": fingerprint, "metrics_version": METRICS_VERSION,
                "refit_version": REFIT_VERSION, "seed": u["seed"]}

    def compute(u: Mapping[str, Any]) -> dict[str, Any]:
        run = runs[u["model"]]
        refit_tag = f"{tag}-lc{fraction_label(u['fraction'])}"
        run_dir = None if force else find_refit_run(run.name, refit_tag, str(u["unit"]))
        if run_dir is not None and (read_json(run_dir / "DONE.json") or {}).get("seed") != u["seed"]:
            run_dir = None                              # gotowy refit z innym seedem
        if run_dir is not None:
            log.info(f"{u['unit']}: gotowy refit {run_dir.name} — liczę tylko metryki")
        else:
            result = refit_from_run(run.path, seed=int(u["seed"]), tag=refit_tag, fit_fraction=float(u["fraction"]),
                                    eval_splits=("val",), analysis="learning_curve", unit=str(u["unit"]))
            run_dir = Path(result["run_dir"])
        return {"model": u["model"], "fraction": u["fraction"], "run": run_dir.name, **unit_metrics(run_dir)}

    def estimate(u: Mapping[str, Any]) -> float:
        return run_duration_s(runs[u["model"]].info) * float(u["fraction"]) + 5.0

    failures = execute_units(store, units, expect, compute, "learning_curve", force=force, estimate_s=estimate)

    models_out: dict[str, Any] = {}
    n_done = 0
    for model in dict.fromkeys(str(u["model"]) for u in all_units):
        model_units = [u for u in all_units if u["model"] == model]
        points = []
        for u in model_units:
            payload = store.valid(str(u["unit"]), expect(u))
            if payload is None:
                continue
            n_done += 1
            points.append({"fraction": u["fraction"], "n_fit": payload.get("n_fit"), "run": payload.get("run"),
                           "best_iteration": payload.get("best_iteration"),
                           "training_duration_s": payload.get("training_duration_s"), "val": payload.get("val")})
        points.sort(key=lambda p: p["fraction"])
        uses_es = MODEL_REGISTRY[model].supports_early_stopping
        n_full = full_fit_rows(uses_es)
        targets = [t for t in (n_full, 2 * n_full if n_full else None) if t]
        power = {metric: fit_power_law([p["n_fit"] for p in points], [(p["val"] or {}).get(metric) for p in points],
                                       targets, HIGHER_IS_BETTER.get(metric)) for metric in FIT_METRICS}
        models_out[model] = {"source_run": model_units[0]["source_run"], "points": points,
                             "n_fit_full_train": n_full, "power_law": power}
    results = {
        "analysis": "learning_curve",
        "feature_set": feature_set,
        "tag": tag,
        "created": now_iso(),
        "data_fingerprint": fingerprint,
        "metrics_version": METRICS_VERSION,
        "source_runs": {u["model"]: u["source_run"] for u in all_units},
        "fractions": sorted({float(u["fraction"]) for u in all_units}),
        "metrics": METRICS,
        "fit_metrics": list(FIT_METRICS),
        "power_law_formula": ("metric(n) = a + b * n^(-c); n = wiersze X_fit; predicted: n X_fit całego TRAIN i 2×; "
                              f"c w {list(C_BOUNDS)} (≥ 4 punkty), c = {C_FIXED} przy 3 punktach; "
                              "reliable = brak powodów w `reasons` (c ustalone / na granicy, zły kierunek, "
                              "ekstrapolacja gorsza od ostatniego punktu, duże RMSE, cel > "
                              f"{MAX_EXTRAPOLATION_FACTOR:g}× n krzywej)"),
        "n_units": len(all_units),
        "n_done": n_done,
        "complete": not failures and n_done == len(all_units),
        "failures": failures,
        "models": models_out,
    }
    try:
        plot = plot_learning_curve(results, store.root / "learning_curve.png")
        results["plot"] = str(plot) if plot else None
    except Exception as exc:  # noqa: BLE001 — wykres nie może unieważnić wyników
        log.warning(f"Wykres krzywej uczenia nieudany: {exc}")
        results["plot"] = None
    path = store.write_results(results)
    store.write_summary(summary_lines(results))
    log.info(f"Krzywa uczenia: {n_done}/{len(all_units)} jednostek (kumulatywnie) → {path}")
    return results


__all__ = ["DEFAULT_FRACTIONS", "DEFAULT_MODELS", "HIGHER_IS_BETTER", "fit_power_law", "parse_fractions",
           "run_learning_curve"]
