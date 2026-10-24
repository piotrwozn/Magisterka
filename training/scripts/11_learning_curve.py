"""
Krok 11 (analiza C9): krzywa uczenia — refit z najlepszymi parametrami na ułamkach zbioru
dopasowania X_fit, metryki na val, dopasowanie potęgowe a + b·n^(−c) i ekstrapolacja.

Użycie (z katalogu training/):
    python scripts/11_learning_curve.py                                  # lightgbm, xgboost; 5/10/25/50/100%
    python scripts/11_learning_curve.py --models lightgbm --fractions 0.1,0.25,1
    python scripts/11_learning_curve.py --tag smoke --models lightgbm --fractions 0.25,0.5,1   # test

Źródło: najnowszy ukończony, dostrojony run każdego modelu z tagiem `--tag`. Refity:
runs/<czas>_<model>_<cechy>_<tag>-lc<ułamek>/ (DONE.json `custom: true`, `refit_of`; predykcje tylko
na val). Wynik: runs/analyses/learning_curve__<cechy>__<tag>/results.json, summary.md,
learning_curve.png. Wznawialne (--force: od nowa). Czas ≈ suma ułamków × czas treningu finalnego.

Kody wyjścia: 0 = OK, 1 = któraś jednostka nieudana, 2 = brak runów źródłowych / złe argumenty,
130 = Ctrl+C.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.cli import apply_n_jobs, base_parser, parse_model_list  # noqa: E402


def run(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    apply_n_jobs(argv)
    parser = base_parser("C9: krzywa uczenia (refity na ułamkach X_fit, metryki val)", "lightgbm,xgboost")
    parser.add_argument("--fractions", default="0.05,0.1,0.25,0.5,1.0", help="Ułamki X_fit po przecinku, (0, 1]")
    parser.add_argument("--seed", type=int, default=None, help="Seed modelu (domyślnie jak w runie źródłowym)")
    args = parser.parse_args(argv)

    from src.utils.hardware import disable_power_throttling

    disable_power_throttling()
    try:
        from src.analysis.common import NoSourceRuns
        from src.analysis.learning_curve import parse_fractions, run_learning_curve
        from src.utils.logger import get_logger

        log = get_logger("11_learning_curve")
        try:
            fractions = parse_fractions(args.fractions)
        except ValueError as exc:
            log.error(str(exc))
            return 2
        try:
            results = run_learning_curve(feature_set=args.feature_set, tag=args.tag,
                                         models=parse_model_list(args.models), fractions=fractions,
                                         seed=args.seed, force=args.force)
        except NoSourceRuns as exc:
            log.error(str(exc))
            return 2
    except KeyboardInterrupt:
        print("Przerwano (Ctrl+C) — uruchom to samo polecenie, gotowe jednostki zostaną pominięte.")
        return 130
    return 1 if results["failures"] else 0


if __name__ == "__main__":
    sys.exit(run())
