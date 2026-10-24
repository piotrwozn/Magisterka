"""
Krok 10 (analiza C3): walidacja leave-site-out z korektą rozkładu klas.

Dla każdego wyłączonego oddziału (domyślnie A i B) i modelu: refit z najlepszymi parametrami na
wierszach TRAIN pozostałych oddziałów, ocena na wierszach TRAIN + VAL wyłączonego oddziału
(nigdy test): surowo, z korektą na znany lokalny rozkład ESI i z rozkładem z EM bez etykiet
(Saerens i in. 2002); reguła „undertriage ≤ T” przeniesiona z VAL pozostałych oddziałów
i dopasowana lokalnie (połowa oddziału → druga połowa).

Użycie (z katalogu training/):
    python scripts/10_leave_site_out.py                               # A, B × lightgbm, xgboost, catboost, logreg
    python scripts/10_leave_site_out.py --sites A --models lightgbm
    python scripts/10_leave_site_out.py --tag smoke --sites A --models logreg,lightgbm   # test (sekundy)

Refity: runs/<czas>_<model>_<cechy>_<tag>-siteout<S>/ (DONE.json `custom: true`, `refit_of`).
Wynik: runs/analyses/site_out__<cechy>__<tag>/results.json + summary.md. Wznawialne (--force: od nowa).
Czas: ≈ liczba oddziałów × czas treningu finalnego każdego modelu.

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
    parser = base_parser("C3: leave-site-out + korekta rozkładu klas (znany / EM)",
                         "lightgbm,xgboost,catboost,logreg")
    parser.add_argument("--sites", default="A,B", help="Wyłączane oddziały po przecinku")
    args = parser.parse_args(argv)

    from src.utils.hardware import disable_power_throttling

    disable_power_throttling()
    try:
        from src.analysis.common import NoSourceRuns
        from src.analysis.site_out import parse_sites, run_leave_site_out
        from src.utils.logger import get_logger

        log = get_logger("10_leave_site_out")
        try:
            sites = parse_sites(args.sites)
        except ValueError as exc:
            log.error(str(exc))
            return 2
        try:
            results = run_leave_site_out(feature_set=args.feature_set, tag=args.tag,
                                         models=parse_model_list(args.models), sites=sites, force=args.force)
        except NoSourceRuns as exc:
            log.error(str(exc))
            return 2
    except KeyboardInterrupt:
        print("Przerwano (Ctrl+C) — uruchom to samo polecenie, gotowe jednostki zostaną pominięte.")
        return 130
    return 1 if results["failures"] else 0


if __name__ == "__main__":
    sys.exit(run())
