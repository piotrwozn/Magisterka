"""
Wspólne CLI skryptów analiz (09–11) — lekki moduł: NIE importuje `src.utils.config`.

Liczba wątków (`--n-jobs` → SOR_AI_N_JOBS, OMP_NUM_THREADS) musi być ustawiona przed importem
config (N_JOBS ustalane przy imporcie), dokładnie jak w scripts/03_train.py.
"""

from __future__ import annotations

import argparse
import os


def apply_n_jobs(argv: list[str]) -> int | None:
    """`--n-jobs N` → SOR_AI_N_JOBS (+ OMP_NUM_THREADS, jeśli nieustawione) przed importem `src.utils`."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--n-jobs", type=int, default=None)
    known, _ = pre.parse_known_args(argv)
    if known.n_jobs is None or known.n_jobs <= 0:
        return None
    os.environ["SOR_AI_N_JOBS"] = str(known.n_jobs)
    os.environ.setdefault("OMP_NUM_THREADS", str(known.n_jobs))
    return known.n_jobs


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"oczekiwano liczby > 0, podano {value}")
    return number


def base_parser(description: str, default_models: str) -> argparse.ArgumentParser:
    """Opcje wspólne: zestaw cech, tag runów źródłowych, modele, --force, --n-jobs."""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--feature-set", "-f", default="triage", help="Zestaw cech runów źródłowych")
    parser.add_argument("--tag", default=None,
                        help="Tag runów źródłowych (najnowszy ukończony, dostrojony run każdego modelu); "
                             "domyślnie profil domyślny (standard)")
    parser.add_argument("--models", default=default_models,
                        help="Lista po przecinku (aliasy: lr, lgbm, xgb, cb, rf, pfn, icl), 'all' (6 modeli "
                             "klasycznych) albo 'tfm' (TabPFN + TabICL — refity TFM tylko na jawne żądanie)")
    parser.add_argument("--force", action="store_true", help="Policz od nowa także gotowe jednostki")
    parser.add_argument("--n-jobs", type=positive_int, default=None,
                        help="Wątki CPU (SOR_AI_N_JOBS; ustawiane przed importem config)")
    return parser


def parse_model_list(spec: str) -> list[str]:
    """
    Lista modeli: 'all' (albo pusty tekst) → 6 modeli KLASYCZNYCH (modele fundamentalne TFM tylko jawnie:
    'tfm', 'tabpfn', 'tabicl'); nazwy i aliasy rozwiązuje `src.models.resolve_model_list`.
    """
    from src.models import resolve_model_list

    return resolve_model_list(spec)


__all__ = ["apply_n_jobs", "base_parser", "parse_model_list", "positive_int"]
