"""
Krok 1: Konwersja .RData → .parquet.

Plik Yale EMMLC ('5v_cleandf.rdata', ~100 MB) wczytuje się powoli przez pyreadr.
Konwertujemy go raz do parquet, by kolejne ładowania były ~10× szybsze.

Użycie:
    python scripts/01_convert_rdata.py
    python scripts/01_convert_rdata.py --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Dodaj root projektu do PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import convert_rdata_to_parquet  # noqa: E402
from src.utils.config import RAW_PARQUET, RDATA_FILE  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Konwertuje .RData → .parquet")
    parser.add_argument(
        "--rdata", default=str(RDATA_FILE),
        help="Ścieżka do pliku .RData",
    )
    parser.add_argument(
        "--parquet", default=str(RAW_PARQUET),
        help="Wyjściowy plik parquet",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Nadpisz istniejący parquet",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("KROK 1: Konwersja Yale EMMLC .RData → .parquet")
    log.info("=" * 60)

    parquet_path = convert_rdata_to_parquet(
        rdata_path=args.rdata,
        parquet_path=args.parquet,
        overwrite=args.overwrite,
    )

    log.info(f"\n✓ Gotowe! Parquet zapisany w: {parquet_path}")
    log.info(f"  Następny krok: python scripts/02_preprocess.py")


if __name__ == "__main__":
    main()
