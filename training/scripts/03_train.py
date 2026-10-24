"""
Krok 3: trening modeli ESI (CLI z `src.models.train.main`).

Użycie:
    python scripts/03_train.py --model all                          # 6 modeli, zestaw 'triage'
    python scripts/03_train.py --model xgboost
    python scripts/03_train.py --model lightgbm,catboost --feature-set full
    python scripts/03_train.py --model all --tune                   # Optuna, profil 'standard' (noc)
    python scripts/03_train.py --model all --tune --profile quick   # test pipeline'u (~30 min)
    python scripts/03_train.py --model all --tune --profile smoke   # test instalacji (minuty)
    python scripts/03_train.py --model ebm --tune --n-trials 10 --tune-timeout 7200 --n-jobs 16

Profile (`--profile smoke|quick|standard|thorough`, config.TUNING_PROFILES) ustalają liczbę prób,
próbkę do tuningu, foldy, limit czasu i próbki danych; jawne --n-trials / --tune-rows /
--cv-folds / --tune-timeout / --sample / --eval-sample mają pierwszeństwo.
Tuning jest wznawialny: po awarii lub Ctrl+C uruchom to samo polecenie — Optuna dokończy
brakujące próby i brakujący czas.

`--n-jobs N` ustawia SOR_AI_N_JOBS (i domyślnie OMP_NUM_THREADS) PRZED importem pakietu `src`
(config ustala liczbę wątków przy imporcie). Na Windows proces na starcie wyłącza dławienie
energii (EcoQoS, `hardware.disable_power_throttling`).

Tag runa (`--tag`, domyślnie): nazwa profilu dla tuningu z budżetem profilu, `<profil>-notune`
bez `--tune`, `<profil>-custom`, gdy podano nadpisania (--n-trials, --tune-rows, --cv-folds,
--tune-timeout, --sample, --eval-sample, --weights ≠ none, --tune-metric qwk, --filter, --no-save).

Kody wyjścia: 0 = OK, 1 = co najmniej jeden model nieudany (pozostałe trenowane dalej),
130 = przerwano (Ctrl+C).

Wyniki: runs/<run>/ (model, metryki, predykcje val/test, report.md, DONE.json)
+ runs/leaderboard.csv (metryki VAL wszystkich ukończonych runów).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Dodaj root projektu do PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def apply_n_jobs(argv: list[str]) -> int | None:
    """`--n-jobs N` → SOR_AI_N_JOBS (+ OMP_NUM_THREADS, jeśli nieustawione) przed importem `src`."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--n-jobs", type=int, default=None)
    known, _ = pre.parse_known_args(argv)
    if known.n_jobs is None or known.n_jobs <= 0:
        return None
    os.environ["SOR_AI_N_JOBS"] = str(known.n_jobs)
    os.environ.setdefault("OMP_NUM_THREADS", str(known.n_jobs))
    return known.n_jobs


def run(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    apply_n_jobs(argv)
    # Windows: bez dławienia energii (EcoQoS) przy zminimalizowanej konsoli — lekki import (ctypes)
    from src.utils.hardware import disable_power_throttling

    disable_power_throttling()
    try:
        from src.models.train import main
    except KeyboardInterrupt:
        return 130
    return main(argv)


if __name__ == "__main__":
    sys.exit(run())
