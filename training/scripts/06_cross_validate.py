"""
Krok 6 (opcjonalny): stratified K-fold cross-validation modelu na zbiorze TRAIN.

Szacuje stabilność modelu (średnia ± odchylenie standardowe metryk po foldach).
`val` i `test` nie są używane. Foldy stratyfikowane po ESI × oddział (jak podział
danych). W każdym foldzie 10% części treningowej idzie na early stopping.
Metryki dla reguł decyzyjnych argmax i koszt.

Zapis:
    results/tables/cv_<model>_<zestaw>[_n<próbka>]_<czas>_per_fold.csv
    results/tables/cv_<model>_<zestaw>[_n<próbka>]_<czas>_summary.csv
    logs/evaluation/cv_<model>/<czas>.log

Użycie:
    python scripts/06_cross_validate.py --model lightgbm --k 5
    python scripts/06_cross_validate.py --model xgboost --feature-set full --weights balanced
    python scripts/06_cross_validate.py --from-run 20261019_115739_xgboost_triage_tunesmoke
    python scripts/06_cross_validate.py --model logreg --k 3 --sample 5000   # szybki test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Dodaj root projektu do PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (  # noqa: E402
    ensure_prepared,
    get_xy,
    load_splits,
    resolve_feature_set,
    sample_frame,
)
from src.data.prepare import SITE_COL  # noqa: E402
from src.evaluation.cross_validation import cross_validate_model, make_strata  # noqa: E402
from src.models import MODEL_ALIASES, MODEL_REGISTRY, get_model, resolve_model_name  # noqa: E402
from src.utils.config import (  # noqa: E402
    CV_N_SPLITS,
    DEFAULT_SAMPLE_WEIGHT_STRATEGY,
    TABLES_DIR,
)
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

CONSOLE_METRICS = [
    "quadratic_weighted_kappa", "f1_macro", "auc_macro", "log_loss",
    "undertriage_rate", "critical_miss_rate", "overtriage_rate", "expected_cost",
]

# Parametry sprzętowe ustawiane w runtime — przy CV z parametrów runa wybiera je bieżąca maszyna
_HARDWARE_PARAMS = ("device", "task_type", "devices", "gpu_ram_part")
WEIGHT_STRATEGIES = ("none", "balanced", "sqrt_balanced", "custom")


def _config_from_run(run_ref: str) -> dict:
    """Model, zestaw cech, hiperparametry, wagi klas i strategia wag z run_info.json runa."""
    from src.ensemble import load_run

    run = load_run(run_ref)
    info = run.info
    params = {k: v for k, v in (info.get("params") or {}).items() if k not in _HARDWARE_PARAMS}
    class_weights = info.get("class_weights")
    # `weights_strategy` (nowe runy; 'optuna_qwk' = wagi klas z Optuny) lub `sample_weight_strategy`
    weights = info.get("weights_strategy") or info.get("sample_weight_strategy")
    if weights not in WEIGHT_STRATEGIES:
        weights = None          # wagi klas z runa (class_weights) mają pierwszeństwo
    log.info(f"Parametry z runa: {run.name}")
    return {
        "model": run.model,
        "feature_set": info.get("feature_set", "triage"),
        "params": params,
        "class_weights": {int(k): float(v) for k, v in class_weights.items()} if class_weights else None,
        "weights": weights,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified K-fold cross-validation na zbiorze TRAIN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m", default=None,
        choices=list(MODEL_REGISTRY) + list(MODEL_ALIASES),
        help="Model (domyślne hiperparametry z config.py)",
    )
    parser.add_argument(
        "--from-run", default=None,
        help="Katalog w runs/ (nazwa lub ścieżka) — model, zestaw cech i hiperparametry z tego runa",
    )
    parser.add_argument("--feature-set", "-f", default=None,
                        help="triage | full (domyślnie: triage albo zestaw runa z --from-run)")
    parser.add_argument("--k", type=int, default=CV_N_SPLITS, help="Liczba foldów")
    parser.add_argument(
        "--weights", default=None,
        choices=list(WEIGHT_STRATEGIES),
        help=f"Wagi klas (domyślnie: {DEFAULT_SAMPLE_WEIGHT_STRATEGY} albo strategia runa)",
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Stratyfikowana próbka TRAIN — do szybkich testów")
    args = parser.parse_args()

    if not args.model and not args.from_run:
        parser.error("Podaj --model <nazwa> albo --from-run <run>")

    run_cfg = _config_from_run(args.from_run) if args.from_run else {}
    model_name = run_cfg.get("model") or resolve_model_name(args.model)
    if args.model and resolve_model_name(args.model) != model_name:
        parser.error(f"--model {args.model} nie zgadza się z modelem runa ({model_name})")
    feature_set = resolve_feature_set(args.feature_set or run_cfg.get("feature_set") or "triage")
    weights = args.weights or run_cfg.get("weights") or DEFAULT_SAMPLE_WEIGHT_STRATEGY
    params = run_cfg.get("params")
    class_weights = run_cfg.get("class_weights")

    log.info("=" * 60)
    log.info(f"CROSS-VALIDATION: {model_name} ({args.k}-fold, zestaw '{feature_set}', TRAIN)")
    log.info("=" * 60)

    # 1. Dane — wyłącznie TRAIN
    ensure_prepared()
    df_train = load_splits(("train",), feature_set=feature_set)["train"]
    df_train = sample_frame(df_train, args.sample)
    X, y = get_xy(df_train, feature_set)
    log.info(f"TRAIN: {X.shape[0]:,} × {X.shape[1]}"
             + (f" (próbka {args.sample:,} — tylko do testów)" if args.sample else ""))
    # Warstwy foldów: ESI × oddział (gdy kolumna oddziału jest w splicie)
    site = df_train[SITE_COL] if SITE_COL in df_train.columns else None
    strata = make_strata(y, site, n_splits=args.k)

    # 2. Factory dla modelu
    def model_factory():
        model = get_model(model_name, params=params or None)
        model.optuna_class_weights = class_weights
        return model

    # 3. CV
    suffix = f"_n{args.sample}" if args.sample else ""
    save_path = TABLES_DIR / f"cv_{model_name}_{feature_set}{suffix}.csv"

    cv_df = cross_validate_model(
        model_factory=model_factory,
        X=X,
        y=y,
        n_splits=args.k,
        sample_weight_strategy=weights,
        feature_set=feature_set,
        save_path=save_path,
        model_name=model_name,
        strata=strata,
    )

    # 4. Podsumowanie na konsoli
    print(f"\n=== CV {model_name} / {feature_set} — {len(cv_df)} foldów (mean ± std) ===")
    print(f"{'metryka':28s} {'argmax':>18s} {'koszt':>18s}")
    for metric in CONSOLE_METRICS:
        cells = []
        for rule in ("argmax", "cost"):
            col = f"{rule}_{metric}"
            cells.append(f"{cv_df[col].mean():.4f} ± {cv_df[col].std():.4f}" if col in cv_df else "—")
        print(f"{metric:28s} {cells[0]:>18s} {cells[1]:>18s}")

    log.info(f"✓ CV zakończone! Wyniki: {save_path.parent}")


if __name__ == "__main__":
    main()
