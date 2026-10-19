"""
Krok 4: Pełna ewaluacja zapisanego modelu na VAL (domyślnie) albo TEST.

Wykonuje:
    1. Załadowanie modelu z runa (runs/<run>/model.joblib) — zestaw cech z artefaktu.
    2. Predykcję na wskazanym splicie (val | test).
    3. Pełną ewaluację dla obu reguł decyzyjnych: argmax i minimalny koszt oczekiwany
       (QWK, AUC, undertriage, mistriage, confusion matrix …).
    4. Wykresy: macierze pomyłek, krzywe ROC, kalibracja, ważność cech.
    5. Zapis do runs/<run>/figures/ (wykresy + metrics_<split>.json);
       dla modelu spoza runs/ — do results/figures/<model>_eval_<czas>/.

Protokół: rozwój i porównania na VAL; `--split test` tylko do raportu końcowego.

Użycie:
    python scripts/04_evaluate.py --run 20261019_115324_xgboost_triage_smoke
    python scripts/04_evaluate.py --model-name xgboost              # najnowszy run modelu
    python scripts/04_evaluate.py --model-name lgbm --split test    # raport końcowy
    python scripts/04_evaluate.py --model path/do/model.joblib
    python scripts/04_evaluate.py --run <run> --sample 3000         # szybki test
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Dodaj root projektu do PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_xy, load_splits, sample_frame  # noqa: E402
from src.ensemble import list_runs, load_run  # noqa: E402
from src.ensemble.runs import PROBA_COLS  # noqa: E402
from src.evaluation.metrics import decide, full_evaluation  # noqa: E402
from src.evaluation.visualizations import (  # noqa: E402
    plot_calibration,
    plot_confusion_matrix,
    plot_feature_importances,
    plot_roc_curves,
)
from src.models import MODEL_ALIASES, MODEL_REGISTRY, resolve_model_name  # noqa: E402
from src.models.train import DECISION_RULES  # noqa: E402
from src.utils.config import DATA_VERSION, FIGURES_DIR, RUNS_DIR  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

# Metryki w podsumowaniu na konsoli (pełny komplet w metrics_<split>.json)
SUMMARY_METRICS = [
    "quadratic_weighted_kappa", "f1_macro", "auc_macro", "log_loss", "brier",
    "undertriage_rate", "critical_miss_rate", "overtriage_rate",
    "mistriage_under", "mistriage_over", "within_one_accuracy", "expected_cost",
]


# ─────────────────────────────────────────
# Wybór modelu
# ─────────────────────────────────────────
def find_latest_run(model_name: str, feature_set: str, include_samples: bool):
    """Najnowszy run danego modelu (z zapisanym model.joblib) dla zestawu cech."""
    table = list_runs()
    candidates: list[str] = []
    if not table.empty:
        table = table[(table["model"] == model_name) & (table["feature_set"] == feature_set)]
        if not include_samples:
            table = table[table["train_sample"].isna() & table["eval_sample"].isna()]
        candidates = table["run"].tolist()
    for run_name in reversed(candidates):   # list_runs: najnowsze na końcu
        if (RUNS_DIR / run_name / "model.joblib").exists():
            log.info(f"Najnowszy run '{model_name}' ({feature_set}): {run_name}")
            return load_run(run_name)
    hint = "" if include_samples else " (runy na próbkach pomijane — dodaj --include-samples)"
    raise FileNotFoundError(
        f"Brak runa modelu '{model_name}' z zestawem '{feature_set}' i plikiem model.joblib "
        f"w {RUNS_DIR}{hint}"
    )


def load_model(path: Path, name: str | None = None) -> tuple:
    """
    Ładuje artefakt modelu właściwą klasą z MODEL_REGISTRY.

    Returns (model, data_version artefaktu albo None, gdy podano `name` z runa).
    """
    data_version = None
    if name is None:   # artefakt spoza runs/ — nazwa i wersja danych tylko z pliku
        artifact = joblib.load(path)
        name, data_version = artifact["name"], artifact.get("data_version")
    return MODEL_REGISTRY[resolve_model_name(name)].load(path), data_version


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


def make_plots(
    model_name: str,
    split: str,
    y: np.ndarray,
    proba: np.ndarray,
    importances: pd.DataFrame | None,
    out_dir: Path,
    suffix: str = "",
) -> list[Path]:
    """Macierze pomyłek (obie reguły), ROC, kalibracja i ważność cech."""
    paths = []
    for rule in DECISION_RULES:
        paths.append(plot_confusion_matrix(
            y, decide(proba, rule),
            save_path=out_dir / f"confusion_matrix_{split}_{rule}{suffix}.png",
            title=f"Macierz pomyłek — {model_name} ({split}, reguła {rule})",
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


def _print_summary(metrics: dict, split: str) -> None:
    print(f"\n{'metryka':28s} {'argmax':>10s} {'koszt':>10s}")
    for m in SUMMARY_METRICS:
        cells = []
        for rule in DECISION_RULES:
            v = metrics.get(f"{split}_{rule}_{m}")
            cells.append(f"{v:.4f}" if isinstance(v, float) else "—")
        print(f"{m:28s} {cells[0]:>10s} {cells[1]:>10s}")


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
    args = parser.parse_args()

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
        model, _ = load_model(model_path, name=run.model)
        data_version = run.info.get("data_version")
        label = run.name
    else:
        model_path = Path(args.model)
        model, data_version = load_model(model_path)
        label = model_path.stem

    log.info("=" * 60)
    log.info(f"EWALUACJA: {label} — split '{args.split}'")
    log.info("=" * 60)
    log.info(f"Model: {model.name} (run_id: {model.run_id}), zestaw cech: {model.feature_set}, "
             f"cech: {len(model.feature_names) if model.feature_names else '?'}")

    if data_version and data_version != DATA_VERSION:
        log.error(f"Model trenowano na danych '{data_version}', bieżące splity to '{DATA_VERSION}' "
                  f"— przetrenuj model (scripts/03_train.py)")
        sys.exit(1)

    # 2–3. Predykcje + metryki
    metrics, df, proba = evaluate_model(model, args.split, args.sample)
    y = df["target"].to_numpy()
    if run is not None:
        _check_saved_predictions(run, args.split, df["row_id"], proba)

    # 4. Wykresy + zapis
    if run is not None:
        out_dir = run.path / "figures"
    else:
        out_dir = FIGURES_DIR / f"{model.name}_eval_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_n{args.sample}" if args.sample else ""

    if not args.no_plots:
        log.info("Generowanie wykresów…")
        make_plots(model.name, args.split, y, proba, model.feature_importances(), out_dir, suffix)

    payload = {
        "run": run.name if run is not None else None,
        "model_path": str(model_path),
        "model": model.name,
        "feature_set": model.feature_set,
        "split": args.split,
        "n": int(len(df)),
        "sample": args.sample,
        "data_version": DATA_VERSION,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    metrics_path = out_dir / f"metrics_{args.split}{suffix}.json"
    metrics_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    print(f"\n=== {label} — {args.split.upper()} (n={len(df):,}) ===")
    _print_summary(metrics, args.split)

    log.info("✓ Ewaluacja zakończona!")
    log.info(f"  Wyniki: {out_dir}")
    log.info(f"  Metryki JSON: {metrics_path}")


if __name__ == "__main__":
    main()
