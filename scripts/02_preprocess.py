"""
Krok 2: Preprocessing + ESI→MTS mapping + temporal split.

Wykonuje:
    1. Wczytanie raw parquet (lub .RData jeśli brak parquet).
    2. Mapowanie ESI → MTS (Strategia 2: enhanced z dyskryminatorami).
    3. Identyfikację grup cech.
    4. Imputację braków + enkodowanie kategorycznych.
    5. Chronologiczny podział train/val/test.
    6. Zapis do data/processed/{train,val,test}.parquet.

Użycie:
    python scripts/02_preprocess.py
    python scripts/02_preprocess.py --simple-mapping  # bez Strategii 2
    python scripts/02_preprocess.py --stratified      # zamiast chronological
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Dodaj root projektu do PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.esi_mts_mapping import map_dataframe_to_mts  # noqa: E402
from src.data.load_data import load_dataset  # noqa: E402
from src.data.preprocessing import preprocess_dataframe  # noqa: E402
from src.data.splits import (  # noqa: E402
    chronological_split,
    save_splits,
    stratified_split,
)
from src.utils.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    PROCESSED_PARQUET,
    TEST_PARQUET,
    TRAIN_PARQUET,
    VAL_PARQUET,
)
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocessing Yale EMMLC + temporal split")
    parser.add_argument(
        "--simple-mapping", action="store_true",
        help="Użyj prostego mapowania ESI → MTS (bez Strategii 2)",
    )
    parser.add_argument(
        "--stratified", action="store_true",
        help="Użyj podziału stratyfikowanego zamiast chronologicznego (NIEZALECANE)",
    )
    parser.add_argument(
        "--encode", choices=["onehot", "label"], default="label",
        help="Metoda enkodowania kategorycznych (label dla XGBoost/LGBM)",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("KROK 2: Preprocessing + ESI→MTS + Split")
    log.info("=" * 60)

    # 1. Załaduj dane
    log.info("\n[1/4] Ładowanie raw parquet…")
    df = load_dataset()
    log.info(f"      Załadowano: {df.shape[0]:,} × {df.shape[1]:,}")

    # 2. ESI → MTS
    log.info("\n[2/4] Mapowanie ESI → MTS…")
    df = map_dataframe_to_mts(
        df,
        use_enhanced=not args.simple_mapping,
        drop_invalid=True,
    )

    # 3. Preprocessing (imputacja, enkodowanie)
    log.info("\n[3/4] Preprocessing (imputacja + enkodowanie)…")
    df, groups = preprocess_dataframe(
        df,
        encode_method=args.encode,
        impute=True,
    )

    # Zapisz pełny przetworzony parquet (przyda się do EDA)
    log.info(f"\n      Zapisywanie pełnego processed parquet: {PROCESSED_PARQUET}")
    df.to_parquet(PROCESSED_PARQUET, engine="pyarrow", compression="snappy", index=False)

    # 4. Split
    log.info("\n[4/4] Podział train/val/test…")
    if args.stratified:
        log.warning("Używam podziału STRATYFIKOWANEGO (zalecany jest CHRONOLOGICZNY).")
        splits = stratified_split(df, target_col="mts_numeric")
    else:
        splits = chronological_split(df, target_col="mts_numeric")

    # 5. Zapis splitów
    paths = save_splits(splits, output_dir=DATA_PROCESSED_DIR)

    log.info("\n✓ Preprocessing zakończony!")
    log.info(f"  Train: {paths['train']}")
    log.info(f"  Val:   {paths['val']}")
    log.info(f"  Test:  {paths['test']}")
    log.info(f"\n  Następny krok: python scripts/03_train.py --model xgboost")


if __name__ == "__main__":
    main()
