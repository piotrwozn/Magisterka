"""
Unified training pipeline.

Łączy wszystkie modele w jednym CLI:
    python -m src.models.train --model xgboost --feature-set triage_only --tune

Pełen przebieg:
    1. Załaduj splity (train/val/test).
    2. Wybierz feature set (triage_only/full/top).
    3. Opcjonalnie tuning Optuną.
    4. Trening + automatyczne logowanie.
    5. Ewaluacja na zbiorze test.
    6. Zapis modelu + JSON-a eksperymentu + raportu.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.preprocessing import build_feature_groups, split_features
from src.models import MODEL_REGISTRY, get_model
from src.models.tuning import filter_features_by_importance_1fold, tune_model
from src.utils.config import (
    DATA_PROCESSED_DIR,
    EXPERIMENTS_DIR,
    MODELS_DIR,
    OPTUNA_N_TRIALS,
    REPORTS_DIR,
    TEST_PARQUET,
    TRAIN_PARQUET,
    VAL_PARQUET,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────
# Ładowanie splitów
# ─────────────────────────────────────────
def load_splits(
    train_path: Path = TRAIN_PARQUET,
    val_path: Path = VAL_PARQUET,
    test_path: Path = TEST_PARQUET,
) -> dict[str, pd.DataFrame]:
    """Ładuje train/val/test z parquet."""
    paths = {"train": train_path, "val": val_path, "test": test_path}
    splits = {}

    for name, path in paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Brak pliku {path}. "
                f"Uruchom: `python scripts/02_preprocess.py`"
            )
        log.info(f"Ładowanie {name}: {path}")
        splits[name] = pd.read_parquet(path)

    return splits


# ─────────────────────────────────────────
# Główna funkcja treningu
# ─────────────────────────────────────────
def run_training(
    model_name: str,
    feature_set: str = "triage_only",
    tune: bool = False,
    n_trials: int = OPTUNA_N_TRIALS,
    sample_weight_strategy: str = "custom",
    use_mlflow: bool = False,
    save_model: bool = True,
    custom_params: dict | None = None,
    resume_run_id: str | None = None,
    do_filter: bool = True,
) -> dict:
    """
    Pełen pipeline treningu jednego modelu.

    Parameters
    ----------
    model_name : str
        'xgboost' | 'lightgbm' | 'random_forest' | 'ebm' | 'stacking'
    feature_set : str
        'triage_only' | 'full' | 'top'
    tune : bool
        Czy uruchomić Optunę przed treningiem finalnym.
    n_trials : int
        Liczba prób Optuny.
    sample_weight_strategy : str
        'balanced' | 'custom' | 'none'
    use_mlflow : bool
    save_model : bool

    Returns
    -------
    dict z metadanymi runa.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info(f"╔════════════════════════════════════════════════════════════╗")
    log.info(f"║ Run ID: {run_id} | Model: {model_name:14s} | Features: {feature_set:12s} ║")
    log.info(f"╚════════════════════════════════════════════════════════════╝")

    # 1. Załaduj splity
    splits = load_splits()
    df_train, df_val, df_test = splits["train"], splits["val"], splits["test"]

    # 1b. Feature engineering (dodaje ~43 cechy kliniczne)
    from src.features.engineering import engineer_features
    df_train = engineer_features(df_train)
    df_val = engineer_features(df_val)
    df_test = engineer_features(df_test)
    log.info(f"Dodano cechy inżynieryjne: {sum(1 for c in df_train.columns if c.endswith(('_flag','_score','_index','_proxy','_triad','_product','_ratio','_interaction','_instability','_deviation')))} nowych")

    # 2. Identyfikuj grupy cech (na pełnym DataFrame, by uniknąć missingu kolumn)
    groups = build_feature_groups(df_train)

    # 3. Wybierz feature set
    X_train, y_train, feature_names = split_features(df_train, groups, feature_set=feature_set)
    X_val, y_val, _ = split_features(df_val, groups, feature_set=feature_set)
    X_test, y_test, _ = split_features(df_test, groups, feature_set=feature_set)

    # 4. Opcjonalny 1-fold feature importance filter (przed tuningiem)
    if do_filter and tune and len(feature_names) > 100:
        log.info(f"Feature filter przed tuningiem ({len(feature_names)} cech → ?)")
        filtered = filter_features_by_importance_1fold(
            X_train, y_train, feature_names,
            model_name=model_name,
            min_importance=0.0,
        )
        dropped = [c for c in feature_names if c not in filtered]
        if dropped:
            log.info(f"Feature filter odrzucił {len(dropped)} cech")
            X_train = X_train[filtered]
            X_val = X_val[filtered]
            X_test = X_test[filtered]
            feature_names = filtered

    # 5. (Opcjonalnie) Tuning
    best_params = custom_params or {}
    tuning_info = None
    if tune:
        log.info(f"Optuna tuning ({n_trials} trials)…")
        # Per-model SQLite. /tmp/ na serwerze, /tmp/opencode/ w WSL
        db_dir = Path("/tmp/opencode")
        db_dir.mkdir(parents=True, exist_ok=True)
        storage_path = db_dir / f"optuna_studies_{model_name}.db"
        study_name = f"{model_name}_max_quality"
        tuning_info = tune_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_trials=n_trials,
            sample_weight_strategy=sample_weight_strategy,
            storage_path=storage_path,
            study_name=study_name,
        )
        # Rozdziel parametry modelu od wag klas
        best_params = {
            k: v for k, v in tuning_info["best_params"].items()
            if not k.startswith("cw_")
        }
        class_weight_map = {
            k: v for k, v in tuning_info["best_params"].items()
            if k.startswith("cw_")
        }
        log.info(f"Best params: {best_params}")
        log.info(f"Optuna class weights: {class_weight_map}")

    # 5. Trening finalnego modelu
    model = get_model(model_name, params=best_params if best_params else None)

    # Wstrzyknij wagi klas z Optuny do finalnego treningu
    if tuning_info and class_weight_map:
        parsed = {}
        label_map = {"cw_red": 0, "cw_orange": 1, "cw_yellow": 2, "cw_green": 3, "cw_blue": 4}
        for k, v in class_weight_map.items():
            if k in label_map:
                parsed[label_map[k]] = v
        for cls_id in range(5):
            if cls_id not in parsed:
                parsed[cls_id] = 1.0  # fallback for old trials without cw_yellow
        model.optuna_class_weights = parsed
        log.info(f"Używam Optuna class weights w finalnym treningu: {parsed}")

    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        sample_weight_strategy=sample_weight_strategy,
        run_id=run_id,
        use_mlflow=use_mlflow,
        feature_set=feature_set,
    )

    # 6. Ewaluacja na test set (lazy import — by uniknąć cyklu)
    from src.evaluation.metrics import full_evaluation

    log.info("Ewaluacja na zbiorze test…")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    test_metrics = full_evaluation(y_test, y_pred, y_proba, prefix="test")

    if model.tracker is not None:
        model.tracker.log_metrics(test_metrics, prefix="")
        model.tracker.save()
        log.info(f"Pełne metadane runa: {model.tracker.json_path}")

    # 7. Zapis modelu
    model_path = None
    if save_model:
        model_path = model.save()

    # 8. Raport tekstowy
    report_path = REPORTS_DIR / f"{model_name}_{run_id}_report.txt"
    _write_report(report_path, model, test_metrics, tuning_info, run_id, feature_set)

    return {
        "run_id": run_id,
        "model_name": model_name,
        "feature_set": feature_set,
        "test_metrics": test_metrics,
        "tuning": tuning_info,
        "model_path": str(model_path) if model_path else None,
        "report_path": str(report_path),
        "experiment_json": str(model.tracker.json_path) if model.tracker else None,
    }


def _write_report(
    path: Path,
    model,
    test_metrics: dict,
    tuning_info: dict | None,
    run_id: str,
    feature_set: str,
) -> None:
    """Zapisuje czytelny raport tekstowy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"SOR-AI: Raport treningu modelu\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Run ID:       {run_id}\n")
        f.write(f"Model:        {model.name}\n")
        f.write(f"Feature set:  {feature_set}\n")
        f.write(f"Czas treningu: {model.training_duration_s:.1f} s\n")
        f.write(f"Liczba cech:  {len(model.feature_names)}\n\n")

        f.write("HIPERPARAMETRY:\n")
        f.write(json.dumps(model.params, indent=2, default=str))
        f.write("\n\n")

        if tuning_info:
            f.write("OPTUNA TUNING:\n")
            f.write(f"  Najlepsza wartość QWK: {tuning_info['best_value']:.4f}\n")
            f.write(f"  Liczba prób: {tuning_info['n_trials']}\n")
            f.write(f"  Najlepsze parametry:\n")
            f.write(json.dumps(tuning_info["best_params"], indent=2))
            f.write("\n\n")

        f.write("METRYKI (TEST SET):\n")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                f.write(f"  {k:30s}: {v:.4f}\n")
            else:
                f.write(f"  {k:30s}: {v}\n")
        f.write("\n")

        # Top features
        importances = model.feature_importances()
        if importances is not None:
            f.write("TOP 20 NAJWAŻNIEJSZYCH CECH:\n")
            for _, row in importances.head(20).iterrows():
                f.write(f"  {row['feature']:35s} {row['importance']:.4f}\n")

    log.info(f"Raport zapisany: {path}")


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trening modelu triażowego SOR-AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        choices=sorted(MODEL_REGISTRY.keys()),
        default="xgboost",
        help="Który model trenować",
    )
    parser.add_argument(
        "--feature-set", "-f",
        choices=["triage_only", "full", "top"],
        default="triage_only",
        help="Zestaw cech",
    )
    parser.add_argument(
        "--tune", "-t",
        action="store_true",
        help="Uruchom Optuna tuning przed finalnym treningiem",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        default=True,
        help="1-fold feature importance filter przed tuningiem (usuwa cechy z zerową importance)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_false",
        dest="filter",
        help="Pomiń 1-fold feature importance filter",
    )
    parser.add_argument(
        "--n-trials", "-n",
        type=int,
        default=OPTUNA_N_TRIALS,
        help="Liczba prób Optuny",
    )
    parser.add_argument(
        "--weights", "-w",
        choices=["balanced", "custom", "none"],
        default="custom",
        help="Strategia wag sampli",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Loguj równolegle do MLflow",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Nie zapisuj modelu na dysk",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Wznów tuning z podanego run_id (timestamp z nazwy pliku log)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_training(
        model_name=args.model,
        feature_set=args.feature_set,
        tune=args.tune,
        n_trials=args.n_trials,
        sample_weight_strategy=args.weights,
        use_mlflow=args.mlflow,
        save_model=not args.no_save,
        resume_run_id=args.resume,
        do_filter=args.filter,
    )

    log.info("╔════════════════════════════════════════════════════════════╗")
    log.info("║                     TRENING ZAKOŃCZONY                     ║")
    log.info("╚════════════════════════════════════════════════════════════╝")
    log.info(f"Run ID:           {result['run_id']}")
    log.info(f"Model:            {result['model_name']}")
    log.info(f"Test QWK:         {result['test_metrics'].get('quadratic_weighted_kappa', 'N/A')}")
    log.info(f"Test undertriage: {result['test_metrics'].get('undertriage_rate', 'N/A')}")
    log.info(f"Model:            {result['model_path']}")
    log.info(f"Raport:           {result['report_path']}")
    log.info(f"Eksperyment JSON: {result['experiment_json']}")


if __name__ == "__main__":
    main()
