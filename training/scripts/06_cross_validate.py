"""
Krok 6 (opcjonalny): K-fold cross-validation modelu.

Uruchamia stratified 10-fold CV i raportuje metryki ± odchylenie standardowe.
Zapisuje pełną historię CV do CSV oraz log do logs/evaluation/.

Użycie:
    python scripts/06_cross_validate.py --model xgboost --k 10
    python scripts/06_cross_validate.py --model lightgbm --k 5 --temporal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Dodaj root projektu do PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import build_feature_groups, split_features  # noqa: E402
from src.evaluation.cross_validation import cross_validate_model  # noqa: E402
from src.models import MODEL_REGISTRY, get_model  # noqa: E402
from src.utils.config import (  # noqa: E402
    CV_N_SPLITS,
    PROCESSED_PARQUET,
    TABLES_DIR,
)
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="K-fold cross-validation")
    parser.add_argument(
        "--model", "-m", required=True, choices=list(MODEL_REGISTRY.keys()),
    )
    parser.add_argument(
        "--feature-set", "-f", default="triage_only",
        choices=["triage_only", "full", "top"],
    )
    parser.add_argument("--k", type=int, default=CV_N_SPLITS, help="Liczba foldów")
    parser.add_argument(
        "--temporal", action="store_true",
        help="TimeSeriesSplit zamiast StratifiedKFold",
    )
    parser.add_argument("--weights", default="custom", choices=["balanced", "custom", "none"])
    parser.add_argument(
        "--data", default=str(PROCESSED_PARQUET),
        help="Ścieżka do processed parquet (pełen dataset)",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"CROSS-VALIDATION: {args.model} ({args.k}-fold)")
    log.info("=" * 60)

    # 1. Załaduj dane
    log.info(f"\nŁadowanie: {args.data}")
    df = pd.read_parquet(args.data)
    groups = build_feature_groups(df)
    X, y, _ = split_features(df, groups, feature_set=args.feature_set)

    # 2. Factory dla modelu
    def model_factory():
        return get_model(args.model)

    # 3. Run CV
    cv_strategy = "temporal" if args.temporal else "stratified"
    save_path = TABLES_DIR / f"cv_{args.model}_{args.feature_set}.csv"

    cv_df = cross_validate_model(
        model_factory=model_factory,
        X=X,
        y=y,
        n_splits=args.k,
        cv_strategy=cv_strategy,
        sample_weight_strategy=args.weights,
        save_path=save_path,
    )

    log.info(f"\n✓ CV zakończone! Wyniki: {save_path.parent}")


if __name__ == "__main__":
    main()
