"""
Krok 9 (analiza C1): zmienność między seedami — refit każdego finalnego modelu z najlepszymi
parametrami (run_info.json) na kilku seedach, bez tuningu.

Użycie (z katalogu training/):
    python scripts/09_seed_refits.py                               # tag 'standard', wszystkie modele, 3 seedy
    python scripts/09_seed_refits.py --seeds 5 --models lightgbm,xgboost
    python scripts/09_seed_refits.py --tag smoke --seeds 2 --models logreg,lightgbm   # test (sekundy)
    python scripts/09_seed_refits.py --vary-es-split               # seed zmienia też podział fit / ES

Źródło: najnowszy ukończony, dostrojony run każdego modelu z tagiem `--tag` (refit zachowuje jego
próbki danych). Refity: runs/<czas>_<model>_<cechy>_<tag>-seed<k>/ (DONE.json `custom: true`,
`refit_of`; bez pliku modelu, chyba że --save-models). Wynik: runs/analyses/seeds__<cechy>__<tag>/
results.json + summary.md. Wznawialne: gotowe jednostki (model × seed) są pomijane (--force: od nowa).
Wyniki są kumulatywne: wywołanie dla innych modeli dopisuje je do policzonych wcześniej (model liczony
ponownie zastępuje swoje stare wyniki).
Czas: ≈ (liczba seedów) × czas treningu finalnego każdego modelu (GBM minuty, EBM ~1 h na pełnych danych).

Kody wyjścia: 0 = OK, 1 = któraś jednostka nieudana, 2 = brak runów źródłowych, 130 = Ctrl+C.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.cli import apply_n_jobs, base_parser, parse_model_list, positive_int  # noqa: E402


def run(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    apply_n_jobs(argv)
    parser = base_parser("C1: refity finalnych modeli na kilku seedach (średnia ± SD metryk)", "all")
    parser.add_argument("--seeds", type=positive_int, default=3, help="Liczba nowych seedów (1…N)")
    parser.add_argument("--no-source", action="store_true",
                        help="Nie licz runa źródłowego jako jednej z wartości")
    parser.add_argument("--vary-es-split", action="store_true",
                        help="Seed zmienia też podział fit / early stopping (domyślnie tylko seed modelu)")
    parser.add_argument("--save-models", action="store_true", help="Zapisz model.joblib refitów (duże pliki RF)")
    args = parser.parse_args(argv)

    from src.utils.hardware import disable_power_throttling

    disable_power_throttling()
    try:
        from src.analysis.common import NoSourceRuns
        from src.analysis.seeds import run_seed_refits
        from src.utils.logger import get_logger

        log = get_logger("09_seed_refits")
        try:
            results = run_seed_refits(
                feature_set=args.feature_set, tag=args.tag, models=parse_model_list(args.models),
                n_seeds=args.seeds, include_source=not args.no_source, vary_es_split=args.vary_es_split,
                force=args.force, save_models=args.save_models,
            )
        except NoSourceRuns as exc:
            log.error(str(exc))
            return 2
    except KeyboardInterrupt:
        print("Przerwano (Ctrl+C) — uruchom to samo polecenie, gotowe jednostki zostaną pominięte.")
        return 130
    return 1 if results["failures"] else 0


if __name__ == "__main__":
    sys.exit(run())
