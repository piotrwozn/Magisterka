"""
Jedno polecenie na całą noc: przygotowanie danych → tuning + trening każdego modelu →
ewaluacja na VAL → porównanie → podsumowanie. Wznawialne: po awarii, restarcie albo Ctrl+C
uruchom TO SAMO polecenie — gotowe modele są pomijane, Optuna dokańcza brakujące próby.

Użycie (z katalogu training/, w aktywnym venv):
    python scripts/00_check_env.py                    # najpierw: kontrola środowiska
    python scripts/run_all.py --profile smoke         # 5–10 min: test całego pipeline'u
    python scripts/run_all.py                         # właściwy przebieg (profil standard, zestaw triage)
    python scripts/run_all.py --status                # stan bez uruchamiania
    python scripts/run_all.py --profile thorough      # dłuższy tuning
    python scripts/run_all.py --feature-set full      # z historią pacjenta (osobna noc, ≥ 32 GB RAM)
    python scripts/run_all.py --models lightgbm,xgboost --dry-run

Opcje: --profile {smoke,quick,standard,thorough}, --feature-set triage|full|triage,full,
--models all|lista, --split random|site, --no-tune, --lanes auto|1|2, --force, --fail-fast,
--dry-run, --status, --skip-preflight, --no-keep-awake. Szczegóły: src/pipeline/orchestrator.py.

Kody wyjścia: 0 = wszystko gotowe, 1 = któryś krok nieudany, 2 = preflight / dane, 130 = Ctrl+C.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def run(argv: list[str] | None = None) -> int:
    try:
        from src.pipeline.orchestrator import main
    except KeyboardInterrupt:
        return 130
    return main(argv)


if __name__ == "__main__":
    sys.exit(run())
