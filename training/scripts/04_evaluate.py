"""
Krok 4: Pełna ewaluacja zapisanego modelu na VAL (domyślnie) albo TEST.

Wykonuje:
    1. Załadowanie modelu z runa (runs/<run>/model.joblib) — zestaw cech z artefaktu.
    2. Predykcję na wskazanym splicie (val | test).
    3. Pełną ewaluację dla obu reguł decyzyjnych: argmax i minimalny koszt oczekiwany
       (QWK, AUC, undertriage, mistriage, confusion matrix …).
    4. Podgrupy: pacjenci z kompletem parametrów życiowych (`vitals_complete == 1`:
       HR, SBP i RR zmierzone — populacja jak w TOPSOR, gdzie pomiar jest obowiązkowy;
       klucze `{split}_vc_{reguła}_{metryka}`) oraz każdy oddział Yale osobno.
    5. Kalibrację (zawsze): ECE top-label i per poziom ESI, MCE, pewność − trafność
       (całość i podgrupa vc) + wykres niezawodności `reliability_<split>.png`.
    6. Przedziały ufności 95% (bootstrap percentylowy stratyfikowany po ESI, `--bootstrap N`,
       domyślnie 1000, 0 = wyłączone; odsetki pacjentów — dokładny przedział Clopper–Pearsona
       z liczebności k/n) dla obu reguł — całość i podgrupa vc.
    7. Wykresy: macierze pomyłek (całość i podgrupa vc), krzywe ROC, kalibracja, ważność cech.
    8. Zapis do runs/<run>/figures/ (wykresy + metrics_<split>.json z kluczami "metrics",
       "subgroups", "calibration" i "ci"); dla modelu spoza runs/ — do results/figures/<model>_eval_<czas>/.

Protokół: rozwój i porównania na VAL; `--split test` tylko do raportu końcowego.

Modele fundamentalne (TFM: tabpfn, tabicl): dla runa (`--run` / `--model-name`) ewaluacja na predykcjach
zapisanych w treningu (`predictions_<split>.parquet`) zamiast ponownej, wielogodzinnej inferencji na GPU
(run_all wywołuje ten skrypt po każdym treningu); obejmuje wiersze zapisane w runie (próbka eval_sample
runa → WARNING). `--sample` albo `--model <plik>` liczą predykcje od nowa (wymaga torch i biblioteki).

Użycie:
    python scripts/04_evaluate.py --run 20261019_115324_xgboost_triage_smoke
    python scripts/04_evaluate.py --model-name xgboost              # najnowszy run modelu
    python scripts/04_evaluate.py --model-name lgbm --split test    # raport końcowy
    python scripts/04_evaluate.py --model path/do/model.joblib
    python scripts/04_evaluate.py --run <run> --sample 3000         # szybki test
    python scripts/04_evaluate.py --run <run> --bootstrap 0         # bez przedziałów ufności
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Dodaj root projektu do PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_xy, load_splits, sample_frame  # noqa: E402
from src.data.prepare import SITE_COL, VITALS_COMPLETE_COL  # noqa: E402
from src.ensemble import list_runs, load_run  # noqa: E402
from src.evaluation.cross_validation import DECISION_RULES  # noqa: E402
from src.evaluation.metrics import decide, full_evaluation  # noqa: E402
from src.evaluation.statistics import (  # noqa: E402
    DEFAULT_N_BINS,
    PROBA_METRICS,
    bootstrap_samples,
    calibration_summary,
    ci_to_dict,
    format_ci,
)
from src.evaluation.visualizations import (  # noqa: E402
    plot_calibration,
    plot_confusion_matrix,
    plot_feature_importances,
    plot_reliability,
    plot_roc_curves,
)
from src.models import MODEL_ALIASES, MODEL_REGISTRY, load_model, resolve_model_name  # noqa: E402
from src.utils.config import CLASS_NAMES, DATA_VERSION, FIGURES_DIR, RUNS_DIR, TARGET_COL  # noqa: E402
from src.utils.fileio import write_json_atomic  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

# Metryki w podsumowaniu na konsoli (pełny komplet w metrics_<split>.json)
SUMMARY_METRICS = [
    "quadratic_weighted_kappa", "f1_macro", "auc_macro", "log_loss", "brier",
    "undertriage_rate", "critical_miss_rate", "overtriage_rate",
    "mistriage_under", "mistriage_over", "within_one_accuracy", "expected_cost",
]
# Kolumny prawdopodobieństw w runs/<run>/predictions_<split>.parquet (p_esi1 … p_esi5)
PROBA_COLS = [f"p_{name.lower()}" for name in CLASS_NAMES]

# Przedziały ufności: metryki w tabeli na konsoli (pełny komplet w metrics_<split>.json → "ci")
CI_SUMMARY_METRICS = [
    "quadratic_weighted_kappa", "undertriage_rate", "critical_miss_rate", "overtriage_rate",
    "recall_esi1", "expected_cost", "auc_macro", "log_loss", "ece",
]
# Odsetki rzadkich zdarzeń — w tabeli także liczebność k/n
COUNT_METRICS = frozenset({"critical_miss_rate", "recall_esi1"})
BOOTSTRAP_SEED = 42
BOOTSTRAP_ALPHA = 0.05
# Losowania „idealnie skalibrowanego” modelu — wartość odniesienia ECE (szum przy tej liczności)
ECE_NULL_SIMS = 200

# Metryki w tabeli per oddział (metryka → krótka etykieta kolumny)
SITE_SUMMARY_METRICS: dict[str, str] = {
    "quadratic_weighted_kappa": "QWK",
    "undertriage_rate": "under",
    "critical_miss_rate": "crit_miss",
    "overtriage_rate": "over",
}


# ─────────────────────────────────────────
# Wybór modelu
# ─────────────────────────────────────────
def find_latest_run(
    model_name: str,
    feature_set: str,
    include_samples: bool,
    runs_dir: Path = RUNS_DIR,
):
    """Najnowszy run danego modelu (z zapisanym model.joblib) dla zestawu cech."""
    table = list_runs(runs_dir)
    candidates: list[str] = []
    if not table.empty:
        table = table[(table["model"] == model_name) & (table["feature_set"] == feature_set)]
        if not include_samples:
            table = table[table["train_sample"].isna() & table["eval_sample"].isna()]
        candidates = table["run"].tolist()
    for run_name in reversed(candidates):   # list_runs: najnowsze na końcu
        if (Path(runs_dir) / run_name / "model.joblib").exists():
            log.info(f"Najnowszy run '{model_name}' ({feature_set}): {run_name}")
            return load_run(run_name, runs_dir)
    hint = "" if include_samples else " (runy na próbkach pomijane — dodaj --include-samples)"
    raise FileNotFoundError(
        f"Brak runa modelu '{model_name}' z zestawem '{feature_set}' i plikiem model.joblib "
        f"w {runs_dir}{hint}"
    )


def _run_for_model_path(path: Path):
    """Run, do którego należy plik modelu (jeśli leży w katalogu runa)."""
    if (path.parent / "run_info.json").exists():
        return load_run(path.parent)
    return None


# ─────────────────────────────────────────
# Kontrola spójności z predykcjami z treningu
# ─────────────────────────────────────────
def _check_saved_predictions(run, split: str, row_ids: pd.Series, proba: np.ndarray) -> None:
    """Porównuje predykcje z zapisanymi w runs/<run>/predictions_<split>.parquet (wspólne wiersze)."""
    path = run.path / f"predictions_{split}.parquet"
    if not path.exists():
        return
    saved = pd.read_parquet(path, columns=["row_id"] + PROBA_COLS).set_index("row_id")
    current = pd.DataFrame(proba, columns=PROBA_COLS, index=row_ids.to_numpy())
    common = current.index.intersection(saved.index)
    if len(common) == 0:
        log.info("Kontrola spójności: brak wspólnych wierszy z predykcjami z treningu — pomijam")
        return
    diff = float(np.abs(current.loc[common].to_numpy() - saved.loc[common].to_numpy()).max())
    msg = f"Kontrola spójności z predictions_{split}.parquet: {len(common):,} wierszy, max |Δp| = {diff:.2e}"
    if diff > 1e-4:
        log.warning(msg + " — predykcje różnią się od zapisanych w trakcie treningu!")
    else:
        log.info(msg)


# ─────────────────────────────────────────
# Główna ewaluacja
# ─────────────────────────────────────────
def evaluate_saved_predictions(run, model, split: str) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """
    Metryki z predykcji zapisanych w treningu (runy TFM): wiersze splitu z `predictions_<split>.parquet`,
    prawdopodobieństwa dopasowane po row_id. Zwraca (metryki, ramka splitu, prawdopodobieństwa).
    """
    path = run.path / f"predictions_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run {run.name} nie ma {path.name} — przetrenuj model (scripts/03_train.py)")
    saved = pd.read_parquet(path, columns=["row_id"] + PROBA_COLS).drop_duplicates("row_id").set_index("row_id")
    feature_set = model.feature_set or "triage"
    full = load_splits((split,), feature_set=feature_set)[split]
    df = full[full["row_id"].isin(saved.index)]
    if len(df) != len(saved):
        raise ValueError(f"{path.name}: {len(saved) - len(df)} wierszy spoza bieżącego splitu '{split}' — "
                         f"run z innych danych")
    if len(df) < len(full):
        log.warning(f"{split.upper()}: predykcje runa obejmują {len(df):,} z {len(full):,} wierszy (próbka "
                    f"eval_sample runa) — metryki tylko na tych wierszach, nie do raportowania!")
    raw = saved.loc[df["row_id"].to_numpy(), PROBA_COLS].to_numpy(dtype=np.float64)
    sums = raw.sum(axis=1, keepdims=True)
    proba = np.divide(raw, sums, out=np.full_like(raw, 1.0 / raw.shape[1]), where=sums > 0)
    log.info(f"{split.upper()}: {len(df):,} wierszy — model fundamentalny ({model.name}): metryki z predykcji "
             f"zapisanych w treningu ({path.name}), bez ponownej inferencji")
    y = df[TARGET_COL].to_numpy()
    metrics: dict = {}
    for rule in DECISION_RULES:
        log.info(f"Reguła decyzyjna: {rule}")
        metrics.update(full_evaluation(
            y, decide(proba, rule), proba, prefix=f"{split}_{rule}", print_report=True,
        ))
    return metrics, df, proba


def evaluate_model(model, split: str, sample: int | None) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Predykcje + metryki dla obu reguł. Zwraca (metryki, ramka splitu, prawdopodobieństwa)."""
    feature_set = model.feature_set or "triage"
    df = load_splits((split,), feature_set=feature_set)[split]
    df = sample_frame(df, sample)
    X, y = get_xy(df, feature_set)

    # Kolumny dokładnie jak w treningu (model mógł używać podzbioru cech po filtrze)
    if model.feature_names:
        missing = [c for c in model.feature_names if c not in X.columns]
        if missing:
            raise KeyError(
                f"W danych brakuje {len(missing)} cech modelu (np. {missing[:5]}) — "
                f"model trenowano na innym przygotowaniu danych"
            )
        X = X[model.feature_names]
    log.info(f"{split.upper()}: {X.shape[0]:,} × {X.shape[1]} (zestaw '{feature_set}')"
             + (f" — próbka {sample:,}, nie do raportowania!" if sample else ""))

    proba = model.predict_proba(X)
    metrics: dict = {}
    for rule in DECISION_RULES:
        log.info(f"Reguła decyzyjna: {rule}")
        metrics.update(full_evaluation(
            y, decide(proba, rule), proba, prefix=f"{split}_{rule}", print_report=True,
        ))
    return metrics, df, proba


# ─────────────────────────────────────────
# Podgrupy: komplet parametrów życiowych, oddziały
# ─────────────────────────────────────────
def _rule_metrics(y: np.ndarray, proba: np.ndarray, prefix: str) -> dict:
    """Metryki obu reguł decyzyjnych bez wydruku; klucze `{prefix}_{reguła}_{metryka}`."""
    out: dict = {}
    for rule in DECISION_RULES:
        key = f"{prefix}_{rule}" if prefix else rule
        out.update(full_evaluation(y, decide(proba, rule), proba, prefix=key, print_report=False))
    return out


def vitals_complete_mask(df: pd.DataFrame) -> np.ndarray | None:
    """Maska wierszy z kompletem HR, SBP i RR (None, gdy splitu nie ma kolumny `vitals_complete`)."""
    if VITALS_COMPLETE_COL not in df.columns:
        return None
    return pd.to_numeric(df[VITALS_COMPLETE_COL], errors="coerce").fillna(0).to_numpy() == 1


def evaluate_subgroups(
    df: pd.DataFrame, proba: np.ndarray, split: str
) -> tuple[dict, dict[str, Any]]:
    """
    Metryki w podgrupach.

    Returns
    -------
    vc_metrics : dict
        Metryki na `vitals_complete == 1` z kluczami `{split}_vc_{reguła}_{metryka}`
        (pusty, gdy brak kolumny albo takich wierszy).
    info : dict
        {"vitals_complete": {"n", "n_esi1", "n_incomplete"} | None,
         "sites": {oddział: {"n", "n_esi1", "metrics": {reguła_metryka: …}}}}
    """
    y = df[TARGET_COL].to_numpy()
    proba = np.asarray(proba, dtype=float)
    vc_metrics: dict = {}
    info: dict[str, Any] = {"vitals_complete": None, "sites": {}}

    mask = vitals_complete_mask(df)
    if mask is None:
        log.warning(f"Split '{split}' nie ma kolumny '{VITALS_COMPLETE_COL}' — pomijam podgrupę vc "
                    f"(przygotuj dane ponownie: python scripts/02_preprocess.py)")
    else:
        info["vitals_complete"] = {
            "n": int(mask.sum()),
            "n_esi1": int((y[mask] == 0).sum()),
            "n_incomplete": int((~mask).sum()),
        }
        if mask.any():
            vc_metrics = _rule_metrics(y[mask], proba[mask], prefix=f"{split}_vc")
        log.info(f"Podgrupa vitals_complete == 1: {mask.sum():,} wierszy "
                 f"(ESI 1: {info['vitals_complete']['n_esi1']:,}), bez kompletu: {(~mask).sum():,}")

    if SITE_COL in df.columns:
        sites = df[SITE_COL].astype("string")
        for site in sorted(sites.dropna().unique()):
            site_mask = (sites == site).fillna(False).to_numpy(dtype=bool)
            n_esi1 = int((y[site_mask] == 0).sum())
            info["sites"][str(site)] = {
                "n": int(site_mask.sum()),
                "n_esi1": n_esi1,
                "metrics": _rule_metrics(y[site_mask], proba[site_mask], prefix=""),
            }
            if n_esi1 < 30:
                log.warning(f"Oddział {site}: tylko {n_esi1} wizyt ESI 1 — metryki ESI 1 niestabilne")
    return vc_metrics, info


# ─────────────────────────────────────────
# Kalibracja i przedziały ufności
# ─────────────────────────────────────────
def calibration_section(
    y: np.ndarray, proba: np.ndarray, vc_mask: np.ndarray | None, n_bins: int = DEFAULT_N_BINS
) -> dict[str, Any]:
    """Miary kalibracji (ECE top-label i per poziom ESI, MCE, pewność − trafność) — całość i podgrupa vc."""
    out: dict[str, Any] = {"all": calibration_summary(y, proba, n_bins, n_null=ECE_NULL_SIMS), "vc": None}
    if vc_mask is not None and vc_mask.any():
        out["vc"] = calibration_summary(y[vc_mask], proba[vc_mask], n_bins, n_null=ECE_NULL_SIMS)
    cal = out["all"]
    ref = cal["ece_reference"]
    log.info(f"Kalibracja: ECE top-label = {cal['ece_top_label']:.4f} (idealny model przy tej liczności: "
             f"~{ref['null_mean']:.4f}, p = {ref['p_value']:.3f}), ECE per poziom (średnia) = "
             f"{cal['ece_classwise']['mean']:.4f}, MCE = {cal['mce_top_label']:.4f}, "
             f"pewność − trafność = {cal['overconfidence']:+.4f}")
    return out


def bootstrap_section(
    y: np.ndarray,
    proba: np.ndarray,
    vc_mask: np.ndarray | None,
    n_boot: int,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = BOOTSTRAP_ALPHA,
) -> dict[str, Any] | None:
    """
    95% przedziały ufności (bootstrap percentylowy, losowanie w każdej klasie ESI osobno; odsetki
    pacjentów — dokładny przedział Clopper–Pearsona z k/n) dla obu reguł decyzyjnych — całość
    i podgrupa vc. n_boot ≤ 0 → None.

    Returns
    -------
    {"n_boot", "alpha", "seed", "stratified", "method", "proportions", "duration_s",
     "all": {reguła: {metryka: {value, ci_low, ci_high, se, n_valid, k, n, ci_method}}}, "vc": … | None}
    """
    if n_boot <= 0:
        return None
    start = time.perf_counter()
    out: dict[str, Any] = {"n_boot": int(n_boot), "alpha": alpha, "seed": seed, "stratified": True,
                           "method": "percentile", "proportions": "clopper-pearson (k/n)", "all": None,
                           "vc": None}
    res = bootstrap_samples(y, {"model": proba}, DECISION_RULES, n_boot=n_boot, alpha=alpha, seed=seed)
    out["all"] = {rule: ci_to_dict(res.ci("model", rule)) for rule in DECISION_RULES}
    if vc_mask is not None and vc_mask.any():
        res_vc = bootstrap_samples(y[vc_mask], {"model": proba[vc_mask]}, DECISION_RULES, n_boot=n_boot,
                                   alpha=alpha, seed=seed)
        out["vc"] = {rule: ci_to_dict(res_vc.ci("model", rule)) for rule in DECISION_RULES}
    out["duration_s"] = round(time.perf_counter() - start, 2)
    log.info(f"Bootstrap {n_boot}× (całość{' + vc' if out['vc'] else ''}): {out['duration_s']:.1f} s")
    return out


def _ci_cell(ci: dict[str, Any] | None, rule: str, metric: str) -> str:
    row = ((ci or {}).get(rule) or {}).get(metric)
    if not row:
        return "—"
    text = format_ci(row["value"], row["ci_low"], row["ci_high"], digits=4)
    if metric in COUNT_METRICS and row.get("k") is not None and row.get("n") is not None:
        text += f" ({row['k']}/{row['n']})"
    return text


def print_ci_table(ci: dict[str, Any] | None, calibration: dict[str, Any] | None, split: str) -> None:
    """Zwięzła tabela: wartość [95% CI] dla obu reguł, całość i podgrupa vc; kalibracja w skrócie."""
    if calibration:
        for name, cal in (("całość", calibration.get("all")), ("vc", calibration.get("vc"))):
            if cal:
                ref = cal.get("ece_reference") or {}
                noise = (f" (szum idealnego modelu ~{ref['null_mean']:.4f}, p = {ref['p_value']:.3f})"
                         if ref else "")
                print(f"Kalibracja ({name}): ECE top-label {cal['ece_top_label']:.4f}{noise} | ECE per poziom "
                      f"{cal['ece_classwise']['mean']:.4f} | MCE {cal['mce_top_label']:.4f} | "
                      f"pewność − trafność {cal['overconfidence']:+.4f}")
    if not ci:
        return
    groups = [("all", "")] + ([("vc", "vc ")] if ci.get("vc") else [])
    width = 34
    print(f"\n=== {split.upper()}: 95% przedziały ufności (bootstrap {ci['n_boot']}×, stratyfikowany po ESI; "
          f"odsetki: Clopper–Pearson, (k/n) = zdarzenia / pacjenci) ===")
    decision = [m for m in CI_SUMMARY_METRICS if m not in PROBA_METRICS]
    header = " ".join(f"{prefix + rule:>{width}s}" for _, prefix in groups for rule in DECISION_RULES)
    print(f"{'metryka':26s} {header}")
    for m in decision:
        cells = [_ci_cell(ci[g], rule, m) for g, _ in groups for rule in DECISION_RULES]
        print(f"{m:26s} " + " ".join(f"{c:>{width}s}" for c in cells))
    print(f"{'(prawdopodobieństwa)':26s} " + " ".join(f"{prefix + 'bez reguły':>{width}s}" for _, prefix in groups))
    for m in [m for m in CI_SUMMARY_METRICS if m in PROBA_METRICS]:
        cells = [_ci_cell(ci[g], DECISION_RULES[0], m) for g, _ in groups]
        print(f"{m:26s} " + " ".join(f"{c:>{width}s}" for c in cells))


def make_plots(
    model_name: str,
    split: str,
    y: np.ndarray,
    proba: np.ndarray,
    importances: pd.DataFrame | None,
    out_dir: Path,
    suffix: str = "",
    vc_mask: np.ndarray | None = None,
) -> list[Path]:
    """Macierze pomyłek (obie reguły; także podgrupa vc), ROC, kalibracja i ważność cech."""
    paths = []
    for rule in DECISION_RULES:
        paths.append(plot_confusion_matrix(
            y, decide(proba, rule),
            save_path=out_dir / f"confusion_matrix_{split}_{rule}{suffix}.png",
            title=f"Macierz pomyłek — {model_name} ({split}, reguła {rule})",
        ))
        if vc_mask is not None and vc_mask.any():
            paths.append(plot_confusion_matrix(
                y[vc_mask], decide(proba[vc_mask], rule),
                save_path=out_dir / f"confusion_matrix_{split}_vc_{rule}{suffix}.png",
                title=f"Macierz pomyłek — {model_name} ({split}, komplet parametrów, reguła {rule})",
            ))
    paths.append(plot_roc_curves(
        y, proba,
        save_path=out_dir / f"roc_{split}{suffix}.png",
        title=f"Krzywe ROC — {model_name} ({split})",
    ))
    paths.append(plot_calibration(
        y, proba,
        save_path=out_dir / f"calibration_{split}{suffix}.png",
        title=f"Kalibracja — {model_name} ({split})",
    ))
    paths.append(plot_reliability(
        y, proba,
        path=out_dir / f"reliability_{split}{suffix}.png",
        title=f"Wykres niezawodności — {model_name} ({split})",
    ))
    if importances is not None and not importances.empty:
        paths.append(plot_feature_importances(
            importances,
            save_path=out_dir / "feature_importances.png",
            title=f"Top N najważniejszych cech — {model_name}",
        ))
    return paths


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else "—"


def _print_summary(metrics: dict, split: str, subgroups: dict[str, Any] | None = None) -> None:
    """Tabela: wszystkie wiersze i podgrupa vc (obie reguły) + metryki per oddział."""
    print(f"\n{'metryka':28s} {'argmax':>10s} {'koszt':>10s} {'vc argmax':>10s} {'vc koszt':>10s}")
    for m in SUMMARY_METRICS:
        cells = [_fmt(metrics.get(f"{split}_{rule}_{m}")) for rule in DECISION_RULES]
        cells += [_fmt(metrics.get(f"{split}_vc_{rule}_{m}")) for rule in DECISION_RULES]
        print(f"{m:28s} " + " ".join(f"{c:>10s}" for c in cells))
    vc = (subgroups or {}).get("vitals_complete")
    if vc:
        print(f"vc = komplet HR/SBP/RR: n={vc['n']:,} (ESI 1: {vc['n_esi1']:,}), "
              f"bez kompletu: {vc['n_incomplete']:,}")

    sites = (subgroups or {}).get("sites") or {}
    if sites:
        columns = [(m, label, rule) for m, label in SITE_SUMMARY_METRICS.items() for rule in DECISION_RULES]
        header = " ".join(f"{label + ' ' + rule:>16s}" for _, label, rule in columns)
        print(f"\n{'oddział':8s} {'n':>9s} {'ESI 1':>6s} {header}")
        for site, row in sites.items():
            cells = " ".join(f"{_fmt(row['metrics'].get(f'{rule}_{m}')):>16s}" for m, _, rule in columns)
            print(f"{site:8s} {row['n']:>9,} {row['n_esi1']:>6,} {cells}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ewaluacja zapisanego modelu ESI na VAL/TEST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", "-r", default=None,
                        help="Katalog runa: nazwa w runs/ albo ścieżka (zalecane)")
    source.add_argument("--model-name", default=None,
                        choices=list(MODEL_REGISTRY) + list(MODEL_ALIASES),
                        help="Najnowszy run tego modelu z runs/")
    source.add_argument("--model", "-m", default=None,
                        help="Ścieżka do pliku .joblib (artefakt BaseTriageModel.save)")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="Zbiór ewaluacji; 'test' wyłącznie do raportu końcowego")
    parser.add_argument("--feature-set", "-f", default="triage",
                        help="Zestaw cech przy wyborze runa przez --model-name")
    parser.add_argument("--include-samples", action="store_true",
                        help="--model-name: uwzględnij runy trenowane na próbkach")
    parser.add_argument("--sample", type=int, default=None,
                        help="Stratyfikowana próbka splitu — do szybkich testów (nie do raportowania!)")
    parser.add_argument("--no-plots", action="store_true", help="Pomiń wykresy")
    parser.add_argument("--bootstrap", type=int, default=1000, metavar="N",
                        help="Liczba losowań bootstrap do 95%% przedziałów ufności (0 = bez przedziałów)")
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap musi być ≥ 0")

    # 1. Wybór modelu
    run = None
    if args.run:
        run = load_run(args.run)
    elif args.model_name:
        try:
            run = find_latest_run(resolve_model_name(args.model_name), args.feature_set, args.include_samples)
        except FileNotFoundError as e:
            parser.error(str(e))
    else:
        run = _run_for_model_path(Path(args.model))

    if run is not None:
        model_path = run.path / "model.joblib"
        if not model_path.exists():
            parser.error(f"Run {run.name} nie ma pliku model.joblib (trenowany z --no-save?)")
        label = run.name
    else:
        model_path = Path(args.model)
        label = model_path.stem
    model = load_model(model_path)          # klasa wg nazwy modelu w artefakcie
    data_version = (run.info.get("data_version") if run is not None else None) or model.data_version

    log.info("=" * 60)
    log.info(f"EWALUACJA: {label} — split '{args.split}'")
    log.info("=" * 60)
    log.info(f"Model: {model.name} (run_id: {model.run_id}), zestaw cech: {model.feature_set}, "
             f"cech: {len(model.feature_names) if model.feature_names else '?'}")

    if data_version and data_version != DATA_VERSION:
        log.error(f"Model trenowano na danych '{data_version}', bieżące splity to '{DATA_VERSION}' "
                  f"— przetrenuj model (scripts/03_train.py)")
        sys.exit(1)

    # 2–3. Predykcje + metryki (TFM z runa: z predykcji zapisanych w treningu — bez ponownej inferencji)
    from_saved = run is not None and bool(run.info.get("pretrained")) and not args.sample
    if from_saved:
        metrics, df, proba = evaluate_saved_predictions(run, model, args.split)
    else:
        metrics, df, proba = evaluate_model(model, args.split, args.sample)
    y = df[TARGET_COL].to_numpy()
    if run is not None and not from_saved:
        _check_saved_predictions(run, args.split, df["row_id"], proba)

    # 4. Podgrupy: komplet parametrów (populacja TOPSOR) i oddziały
    vc_metrics, subgroups = evaluate_subgroups(df, proba, args.split)
    metrics.update(vc_metrics)
    vc_mask = vitals_complete_mask(df)

    # 5–6. Kalibracja (zawsze) i przedziały ufności (bootstrap)
    calibration = calibration_section(y, proba, vc_mask)
    ci = bootstrap_section(y, proba, vc_mask, args.bootstrap)

    # 5. Wykresy + zapis
    if run is not None:
        out_dir = run.path / "figures"
    else:
        out_dir = FIGURES_DIR / f"{model.name}_eval_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_n{args.sample}" if args.sample else ""

    if not args.no_plots:
        log.info("Generowanie wykresów…")
        make_plots(model.name, args.split, y, proba, model.feature_importances(), out_dir, suffix,
                   vc_mask=vc_mask)

    payload = {
        "run": run.name if run is not None else None,
        "model_path": str(model_path),
        "model": model.name,
        "feature_set": model.feature_set,
        "split": args.split,
        "n": int(len(df)),
        "sample": args.sample,
        **({"from_saved_predictions": True} if from_saved else {}),
        "data_version": DATA_VERSION,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
        "subgroups": subgroups,
        "calibration": calibration,
        "ci": ci,
    }
    metrics_path = out_dir / f"metrics_{args.split}{suffix}.json"
    write_json_atomic(metrics_path, payload, default=_json_default)

    print(f"\n=== {label} — {args.split.upper()} (n={len(df):,}) ===")
    _print_summary(metrics, args.split, subgroups)
    print()
    print_ci_table(ci, calibration, args.split)

    log.info("✓ Ewaluacja zakończona!")
    log.info(f"  Wyniki: {out_dir}")
    log.info(f"  Metryki JSON: {metrics_path}")


if __name__ == "__main__":
    main()
