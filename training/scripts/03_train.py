"""
Krok 3: trening modeli ESI.

Cienki wrapper na `src.models.train.main()` — to samo CLI co `python -m src.models.train`.

Użycie:
    python scripts/03_train.py --model all                     # 6 modeli, zestaw 'triage'
    python scripts/03_train.py --model xgboost
    python scripts/03_train.py --model lightgbm,catboost --feature-set full
    python scripts/03_train.py --model all --tune              # Optuna (długo)
    python scripts/03_train.py --model all --sample 20000 --eval-sample 10000 --tag smoke

Wyniki: runs/<run>/ (model, metryki, predykcje val/test) + runs/leaderboard.csv.
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
