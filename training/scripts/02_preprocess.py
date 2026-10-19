"""
Krok 2: przygotowanie danych Yale EMMLC do treningu modeli ESI.

Wykonuje (szczegóły: src/data/prepare.py):
    1. Wczytanie surowego parquet (lub .RData, jeśli brak parquet).
    2. Etykieta = natywne ESI 1–5 nadane przy triażu w Yale.
    3. Czyszczenie: °F → °C, wartości niemożliwe → NaN, duplikaty, brak ESI.
    4. Wykluczenie zmiennych specyficznych dla USA i kolumn-przecieków.
    5. Cechy kliniczne (src/features/engineering.py).
    6. Podział train / val / test + manifest + listy cech.
    7. Raport z audytu danych: training/DATA_AUDIT.md.

Użycie:
    python scripts/02_preprocess.py                 # podział losowy (domyślny)
    python scripts/02_preprocess.py --split site    # oddział B jako test zewnętrzny
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prepare import prepare_dataset  # noqa: E402
from src.utils.config import DATA_AUDIT_MD  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Przygotowanie danych Yale EMMLC → ESI")
    parser.add_argument(
        "--split", choices=["random", "site"], default="random",
        help="random = stratyfikowany losowy; site = oddział HOLDOUT_SITE (B) jako test zewnętrzny",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("KROK 2: przygotowanie danych ESI")
    log.info("=" * 60)
    paths = prepare_dataset(strategy=args.split)

    log.info("Gotowe:")
    for name, path in paths.items():
        log.info(f"  {name:5s}: {path}")
    log.info(f"  audyt: {DATA_AUDIT_MD}")
    log.info("Następny krok: python scripts/run_all.py  (albo python scripts/03_train.py --model all)")


if __name__ == "__main__":
    main()
