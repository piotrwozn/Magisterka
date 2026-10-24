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
    python scripts/run_all.py --stage analyses        # po treningu: analizy do pracy (seedy, leave-site-out,
                                                      # krzywa uczenia — scripts/09–11, wznawialne)
    python scripts/run_all.py --stage all             # trening, potem analizy (jedna noc + kilka godzin)

Opcje: --profile {smoke,quick,standard,thorough}, --feature-set triage|full|triage,full,
--models all|lista, --split random|site, --no-tune, --lanes auto|1|2, --force, --fail-fast,
--dry-run, --status, --skip-preflight, --no-keep-awake, --stage train|analyses|all (domyślnie train).
Szczegóły: src/pipeline/orchestrator.py. Wyniki analiz: runs/analyses/<analiza>__<cechy>__<profil>/.

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
    except ImportError as exc:          # świeży venv bez pakietów — wskazówka zamiast tracebacku
        print(f"[BŁĄD ] {type(exc).__name__}: {exc}\n"
              f"          → python -m pip install -r requirements-lock.txt   (w aktywnym venv), "
              f"potem: python scripts/00_check_env.py")
        return 2
    return main(argv)


if __name__ == "__main__":
    sys.exit(run())
