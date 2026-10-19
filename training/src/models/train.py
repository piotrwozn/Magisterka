"""
Trening modeli ESI — jeden punkt wejścia.

    python scripts/03_train.py --model all                 # wszystkie modele, zestaw 'triage'
    python scripts/03_train.py --model lightgbm            # jeden model
    python scripts/03_train.py --model xgboost --tune -n 50
    python scripts/03_train.py --model all --feature-set full
    python scripts/03_train.py --model all --sample 20000  # szybki test pipeline'u

Przebieg dla każdego modelu:
    1. Dane: splity z `scripts/02_preprocess.py` (przygotowane automatycznie, jeśli brak).
    2. Z TRAIN wydzielane 10% na early stopping — `val` zostaje nietknięty.
    3. (Opcjonalnie) tuning Optuną: CV na części treningowej.
    4. Trening + predykcje na `val` i `test`.
    5. Metryki dla dwóch reguł decyzyjnych: argmax i minimalny koszt oczekiwany.
    6. Zapis do runs/<run>/ + dopisanie wiersza do runs/leaderboard.csv.

Protokół łączenia modeli (ręcznie, zob. src/ensemble):
    wagi / progi / kalibrację dobieraj na `predictions_val.parquet`,
    raportuj na `predictions_test.parquet`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import (
    early_stopping_split,
    ensure_prepared,
    get_xy,
    load_splits,
    manifest,
    resolve_feature_set,
    sample_frame,
)
from src.evaluation.metrics import decide, full_evaluation
from src.models import MODEL_REGISTRY, get_model, resolve_model_name
from src.utils.config import (
    CLASS_NAMES,
    DATA_VERSION,
    DEFAULT_SAMPLE_WEIGHT_STRATEGY,
    OPTUNA_N_TRIALS,
    OPTUNA_N_TRIALS_PER_MODEL,
    OPTUNA_STORAGE_DIR,
    TUNE_CV_FOLDS,
    TUNE_MAX_ROWS,
    RUNS_DIR,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

DECISION_RULES = ("argmax", "cost")
PROBA_COLS = [f"p_{name.lower()}" for name in CLASS_NAMES]   # p_esi1 … p_esi5
LEADERBOARD_CSV = RUNS_DIR / "leaderboard.csv"

# Metryki trafiające do leaderboardu (dla val i test, obie reguły)
LEADERBOARD_METRICS = [
    "quadratic_weighted_kappa", "f1_macro", "auc_macro", "log_loss", "brier",
    "undertriage_rate", "critical_miss_rate", "overtriage_rate",
    "mistriage_under", "mistriage_over", "expected_cost",
]


# ─────────────────────────────────────────
# Pomocnicze
# ─────────────────────────────────────────
def _predictions_frame(meta: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    out = meta[["row_id", "site", "esi", "target"]].reset_index(drop=True).copy()
    proba = np.asarray(proba, dtype=np.float32)
    for i, col in enumerate(PROBA_COLS):
        out[col] = proba[:, i]
    for rule in DECISION_RULES:
        out[f"pred_{rule}"] = decide(proba, rule)
    return out


def _evaluate(y: pd.Series, proba: np.ndarray, split: str, admitted: pd.Series | None = None) -> dict:
    metrics: dict = {}
    for rule in DECISION_RULES:
        y_pred = decide(proba, rule)
        res = full_evaluation(y, y_pred, proba, prefix=f"{split}_{rule}", print_report=(rule == "argmax"))
        metrics.update(res)
    if admitted is not None:
        metrics.update(_admission_validity(y, proba, admitted, split))
    return metrics


def _admission_validity(y: pd.Series, proba: np.ndarray, admitted: pd.Series, split: str) -> dict:
    """
    Trafność kliniczna niezależna od etykiety: czy przewidywana pilność przewiduje
    przyjęcie do szpitala? Porównanie z ESI nadanym przez pielęgniarkę.

    Wynik modelu = oczekiwany poziom ESI (Σ p·k); niższy = pilniej → większa szansa przyjęcia.
    """
    from sklearn.metrics import roc_auc_score

    adm = np.asarray(admitted).astype(int)
    if adm.min() == adm.max():
        return {}
    expected_level = np.asarray(proba) @ np.arange(proba.shape[1])
    return {
        f"{split}_admission_auc_model": float(roc_auc_score(adm, -expected_level)),
        f"{split}_admission_auc_nurse_esi": float(roc_auc_score(adm, -np.asarray(y))),
    }


def _append_leaderboard(row: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if LEADERBOARD_CSV.exists():
        existing = pd.read_csv(LEADERBOARD_CSV)
        df_row = pd.concat([existing, df_row], ignore_index=True)
    df_row.to_csv(LEADERBOARD_CSV, index=False)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ─────────────────────────────────────────
# Trening jednego modelu
# ─────────────────────────────────────────
def run_training(
    model_name: str,
    feature_set: str = "triage",
    tune: bool = False,
    n_trials: int | None = None,
    tune_metric: str = "logloss",
    tune_rows: int | None = TUNE_MAX_ROWS,
    cv_folds: int = TUNE_CV_FOLDS,
    sample_weight_strategy: str = DEFAULT_SAMPLE_WEIGHT_STRATEGY,
    use_mlflow: bool = False,
    save_model: bool = True,
    custom_params: dict | None = None,
    do_filter: bool = False,
    train_sample: int | None = None,
    eval_sample: int | None = None,
    tag: str | None = None,
) -> dict:
    """
    Pełen pipeline treningu jednego modelu.

    Returns
    -------
    dict z metrykami i ścieżką katalogu runa.
    """
    model_name = resolve_model_name(model_name)
    feature_set = resolve_feature_set(feature_set)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_id}_{model_name}_{feature_set}" + (f"_{tag}" if tag else "")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("═" * 64)
    log.info(f" Run: {run_name}")
    log.info("═" * 64)

    # 1. Dane
    ensure_prepared()
    splits = load_splits(feature_set=feature_set)
    df_train = sample_frame(splits["train"], train_sample)
    df_val = sample_frame(splits["val"], eval_sample)
    df_test = sample_frame(splits["test"], eval_sample)

    X_train, y_train = get_xy(df_train, feature_set)
    X_val, y_val = get_xy(df_val, feature_set)
    X_test, y_test = get_xy(df_test, feature_set)
    X_fit, y_fit, X_es, y_es = early_stopping_split(X_train, y_train)
    log.info(f"Train {len(X_fit):,} (+{len(X_es):,} early stopping) | val {len(X_val):,} | "
             f"test {len(X_test):,} | cechy: {X_train.shape[1]} ({feature_set})")

    # 2. (Opcjonalnie) filtr cech + tuning — wyłącznie na części treningowej
    feature_names = list(X_train.columns)
    tuning_info = None
    best_params = dict(custom_params or {})
    class_weights = None
    if tune:
        from src.models.tuning import filter_features_by_importance_1fold, tune_model

        if do_filter:
            kept = filter_features_by_importance_1fold(X_fit, y_fit, feature_names, model_name=model_name)
            if len(kept) < len(feature_names):
                log.info(f"Filtr cech: {len(feature_names)} → {len(kept)}")
                feature_names = kept
        OPTUNA_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        if n_trials is None:
            n_trials = OPTUNA_N_TRIALS_PER_MODEL.get(model_name, OPTUNA_N_TRIALS)
        tuning_info = tune_model(
            model_name=model_name,
            X_train=X_fit[feature_names],
            y_train=y_fit,
            n_trials=n_trials,
            metric=tune_metric,
            n_folds=cv_folds,
            max_rows=tune_rows,
            sample_weight_strategy=sample_weight_strategy,
            storage_path=OPTUNA_STORAGE_DIR / f"optuna_{model_name}.db",
            study_name=f"{model_name}_{feature_set}_{DATA_VERSION}"
                       + (f"_rows{tune_rows}" if tune_rows else ""),
        )
        best_params.update({k: v for k, v in tuning_info["best_params"].items() if not k.startswith("cw_")})
        cw = {k: v for k, v in tuning_info["best_params"].items() if k.startswith("cw_")}
        if cw:
            class_weights = {i: float(cw.get(f"cw_{CLASS_NAMES[i].lower()}", 1.0)) for i in range(len(CLASS_NAMES))}
            log.info(f"Wagi klas z Optuny: {class_weights}")

    # 3. Trening
    model = get_model(model_name, params=best_params or None)
    model.optuna_class_weights = class_weights
    model.fit(
        X_fit[feature_names], y_fit,
        X_val=X_es[feature_names], y_val=y_es,
        sample_weight_strategy=sample_weight_strategy,
        run_id=run_id,
        use_mlflow=use_mlflow,
        feature_set=feature_set,
    )

    # 4. Predykcje + metryki (val i test)
    proba_val = model.predict_proba(X_val[feature_names])
    proba_test = model.predict_proba(X_test[feature_names])
    log.info("Ewaluacja na VAL…")
    metrics = _evaluate(y_val, proba_val, "val", df_val["admitted"])
    log.info("Ewaluacja na TEST…")
    metrics.update(_evaluate(y_test, proba_test, "test", df_test["admitted"]))

    # 5. Zapis artefaktów
    _predictions_frame(df_val, proba_val).to_parquet(run_dir / "predictions_val.parquet", index=False)
    _predictions_frame(df_test, proba_test).to_parquet(run_dir / "predictions_test.parquet", index=False)

    model_path = None
    if save_model:
        model_path = model.save(run_dir / "model.joblib")

    importances = model.feature_importances()
    if importances is not None:
        importances.to_csv(run_dir / "feature_importance.csv", index=False)

    info = {
        "run_name": run_name,
        "model": model_name,
        "feature_set": feature_set,
        "n_features": len(feature_names),
        "sample_weight_strategy": sample_weight_strategy,
        "class_weights": class_weights,
        "tuned": tune,
        "tune_metric": tune_metric if tune else None,
        "params": model.params,
        "data_version": DATA_VERSION,
        "split_strategy": manifest().get("split_strategy"),
        "n_fit": int(len(X_fit)),
        "n_early_stopping": int(len(X_es)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "train_sample": train_sample,
        "eval_sample": eval_sample,
        "training_duration_s": model.training_duration_s,
        "tuning": tuning_info,
        "class_names": CLASS_NAMES,
        "proba_columns": PROBA_COLS,
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=_json_default))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))
    _write_report(run_dir / "report.md", info, metrics, importances)

    row = {
        "run_name": run_name,
        "model": model_name,
        "feature_set": feature_set,
        "weights": sample_weight_strategy,
        "tuned": tune,
        "n_fit": len(X_fit),
        "n_val": len(X_val),
        "duration_s": round(model.training_duration_s or 0.0, 1),
    }
    for split in ("val", "test"):
        for rule in DECISION_RULES:
            for m in LEADERBOARD_METRICS:
                key = f"{split}_{rule}_{m}"
                if key in metrics:
                    row[key] = metrics[key]
        for key in (f"{split}_admission_auc_model", f"{split}_admission_auc_nurse_esi"):
            if key in metrics:
                row[key] = metrics[key]
    _append_leaderboard(row)

    log.info(f"QWK val (argmax/cost): {metrics['val_argmax_quadratic_weighted_kappa']:.4f} / "
             f"{metrics['val_cost_quadratic_weighted_kappa']:.4f} | "
             f"undertriage val (argmax/cost): {metrics['val_argmax_undertriage_rate']:.4f} / "
             f"{metrics['val_cost_undertriage_rate']:.4f}")
    log.info(f"Artefakty: {run_dir}")

    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "model_name": model_name,
        "feature_set": feature_set,
        "metrics": metrics,
        "model_path": str(model_path) if model_path else None,
    }


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _write_report(path: Path, info: dict, metrics: dict, importances: pd.DataFrame | None) -> None:
    lines = [
        f"# {info['run_name']}",
        "",
        f"- model: `{info['model']}`, zestaw cech: `{info['feature_set']}` ({info['n_features']} cech)",
        f"- wagi: `{info['sample_weight_strategy']}`, tuning: {info['tuned']}",
        f"- dane: `{info['data_version']}`, podział `{info['split_strategy']}`; "
        f"fit {info['n_fit']:,} + early stopping {info['n_early_stopping']:,}, "
        f"val {info['n_val']:,}, test {info['n_test']:,}",
        f"- czas treningu: {info['training_duration_s'] or 0:.1f} s",
        "",
        "## Metryki",
        "",
        "| metryka | val · argmax | val · koszt | test · argmax | test · koszt |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in LEADERBOARD_METRICS + ["accuracy", "within_one_accuracy"]:
        cells = [_fmt(metrics.get(f"{s}_{r}_{m}", "—")) for s in ("val", "test") for r in DECISION_RULES]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    if "val_admission_auc_model" in metrics:
        lines += [
            "",
            "Trafność kliniczna (AUC przewidywania przyjęcia do szpitala z pilności):",
            "",
            "| | val | test |",
            "|---|---:|---:|",
            f"| model (oczekiwany poziom ESI) | {_fmt(metrics.get('val_admission_auc_model'))} "
            f"| {_fmt(metrics.get('test_admission_auc_model'))} |",
            f"| ESI pielęgniarki | {_fmt(metrics.get('val_admission_auc_nurse_esi'))} "
            f"| {_fmt(metrics.get('test_admission_auc_nurse_esi'))} |",
        ]
    lines += [
        "",
        "Reguła **koszt**: klasa minimalizująca oczekiwany koszt (niedotriażowanie 3× droższe).",
        "",
        "## Macierz pomyłek (test, argmax; wiersze = prawda ESI 1–5)",
        "",
        "| | " + " | ".join(CLASS_NAMES) + " |",
        "|---|" + "---:|" * len(CLASS_NAMES),
    ]
    cm = metrics.get("test_argmax_confusion_matrix") or []
    for name, row in zip(CLASS_NAMES, cm):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
    if importances is not None:
        lines += ["", "## Top 25 cech", "", "| cecha | ważność |", "|---|---:|"]
        for _, r in importances.head(25).iterrows():
            lines.append(f"| `{r['feature']}` | {r['importance']:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trening modeli klasyfikacji ESI (SOR-AI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m", default="all",
        help=f"'all', jeden model albo lista po przecinku. Dostępne: {', '.join(MODEL_REGISTRY)}",
    )
    parser.add_argument("--feature-set", "-f", default="triage", help="triage | full")
    parser.add_argument("--tune", "-t", action="store_true", help="Tuning Optuną przed treningiem")
    parser.add_argument("--n-trials", "-n", type=int, default=None,
                        help=f"Liczba prób Optuny (domyślnie {OPTUNA_N_TRIALS}; EBM {OPTUNA_N_TRIALS_PER_MODEL.get('ebm')})")
    parser.add_argument("--tune-metric", choices=["logloss", "qwk"], default="logloss",
                        help="Cel tuningu: logloss (jakość prawdopodobieństw) albo qwk (+ wagi klas)")
    parser.add_argument("--tune-rows", type=int, default=TUNE_MAX_ROWS,
                        help="Próbka TRAIN do tuningu (0 = cały TRAIN); finalny trening zawsze na całości")
    parser.add_argument("--cv-folds", type=int, default=TUNE_CV_FOLDS, help="Liczba foldów CV w tuningu")
    parser.add_argument(
        "--filter", action="store_true",
        help="Przed tuningiem odrzuć cechy z zerową ważnością (szybki model na połowie danych)",
    )
    parser.add_argument(
        "--weights", "-w", choices=["none", "balanced", "sqrt_balanced", "custom"],
        default=DEFAULT_SAMPLE_WEIGHT_STRATEGY, help="Wagi klas w treningu",
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Losowa (stratyfikowana) próbka TRAIN — do szybkich testów")
    parser.add_argument("--eval-sample", type=int, default=None,
                        help="Próbka VAL i TEST — do szybkich testów (nie do raportowania!)")
    parser.add_argument("--tag", default=None, help="Dopisek do nazwy runa")
    parser.add_argument("--mlflow", action="store_true", help="Loguj równolegle do MLflow")
    parser.add_argument("--no-save", action="store_true", help="Nie zapisuj pliku modelu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.model.lower() == "all":
        models = list(MODEL_REGISTRY)
    else:
        models = [resolve_model_name(m.strip()) for m in args.model.split(",") if m.strip()]

    results = []
    for name in models:
        results.append(run_training(
            model_name=name,
            feature_set=args.feature_set,
            tune=args.tune,
            n_trials=args.n_trials,
            tune_metric=args.tune_metric,
            tune_rows=args.tune_rows or None,
            cv_folds=args.cv_folds,
            sample_weight_strategy=args.weights,
            use_mlflow=args.mlflow,
            save_model=not args.no_save,
            do_filter=args.filter,
            train_sample=args.sample,
            eval_sample=args.eval_sample,
            tag=args.tag,
        ))

    log.info("╔════════════════════════════════════════════════════════════╗")
    log.info("║                     TRENING ZAKOŃCZONY                     ║")
    log.info("╚════════════════════════════════════════════════════════════╝")
    for r in results:
        m = r["metrics"]
        log.info(
            f"{r['model_name']:14s} QWK val {m['val_argmax_quadratic_weighted_kappa']:.4f} "
            f"| AUC {m.get('val_argmax_auc_macro', float('nan')):.4f} "
            f"| undertriage {m['val_argmax_undertriage_rate']:.4f} → {r['run_dir']}"
        )
    log.info(f"Zestawienie wszystkich runów: {LEADERBOARD_CSV}")
    log.info("Porównanie i łączenie: python scripts/07_compare_runs.py")


if __name__ == "__main__":
    main()
