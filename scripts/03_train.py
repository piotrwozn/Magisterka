"""
Krok 3: Trening modelu.

Cienki wrapper na `src.models.train.main()` — pozwala uruchomić trening
tym samym CLI co `python -m src.models.train`.

Użycie:
    python scripts/03_train.py --model xgboost
    python scripts/03_train.py --model lightgbm --tune --n-trials 50
    python scripts/03_train.py --model stacking --feature-set full
    python scripts/03_train.py --model ebm --weights custom
"""

from __future__ import annotations

import sys
from pathlib import Path

# Dodaj root projektu do PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import main  # noqa: E402


if __name__ == "__main__":
    main()
