"""
Krok 15 (eksperyment B, modele TFM): czy dołączenie TabPFN-3.5 / TabICLv2 do LightGBM poprawia
prawdopodobieństwa ESI na pełnych danych — wyłącznie na VAL (powtarzany cross-fitting), z bramką
ustaloną z góry. TEST ocenia raz `08_final_report.py` poleceniem z recommendation.json.

Użycie (z katalogu training/):
    python scripts/15_tfm_blend.py --feature-set triage --tag thorough --models tabpfn,tabicl
    python scripts/15_tfm_blend.py --tag thorough --plan                  # runy, warianty, szacunek czasu
    python scripts/15_tfm_blend.py --runs RUN_LGBM RUN_TABPFN --repeats 3 # jawne runy
    python scripts/15_tfm_blend.py --tag smoke --repeats 2 --bootstrap 200   # szybki test

Wejście: zapisane predykcje VAL najnowszych ukończonych, dostrojonych runów lightgbm i TFM z tagiem (bez
GPU, bez ponownej predykcji). Brak runu lightgbm albo żadnego runu TFM → linia `skipped: <powód>`, kod 0.

Wynik: runs/analyses/tfm_blend__<cechy>__<tag>/ — results.json, recommendation.json (runy + dokładne
polecenie 08), report.md (po polsku, do pracy), summary.md, figures/, units/, preds/. Wznawialne: gotowe
powtórzenia cross-fittingu są pomijane (--force: od nowa). Szczegóły: src/analysis/tfm_blend.py, docs/TFM.md.

Kody wyjścia: 0 = OK (także `skipped: …`), 1 = któreś powtórzenie nieudane, 2 = złe argumenty / runy
niewyrównane, 130 = Ctrl+C.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.cli import apply_n_jobs, positive_int  # noqa: E402


def unit_float(value: str) -> float:
    number = float(value)
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError(f"oczekiwano liczby w (0, 1), podano {value}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Eksperyment B: LightGBM + modele fundamentalne (TabPFN, TabICL) — łączenie na VAL z bramką",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--feature-set", "-f", default="triage", help="Zestaw cech runów")
    parser.add_argument("--tag", default=None,
                        help="Tag runów (najnowszy ukończony, dostrojony run lightgbm i TFM); domyślnie profil "
                             "domyślny (standard); przy --runs — tag runu lightgbm")
    parser.add_argument("--models", default="tfm",
                        help="TFM do sprawdzenia po przecinku (tabpfn, tabicl, aliasy pfn / icl, 'tfm' = oba); "
                             "lightgbm zawsze (może być na liście)")
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Jawne runy (nazwy katalogów w runs/ albo ścieżki): lightgbm + najwyżej 2 TFM")
    parser.add_argument("--repeats", type=positive_int, default=5, help="Powtórzenia dwukrotnego cross-fittingu VAL")
    parser.add_argument("--bootstrap", type=positive_int, default=1_000,
                        help="Losowania sparowanego bootstrapu (warstwy ESI)")
    parser.add_argument("--methods", default="mean,geometric", help="Metody łączenia: mean, geometric")
    parser.add_argument("--seed", type=int, default=None, help="Seed foldów i bootstrapu (domyślnie config.RANDOM_SEED)")
    parser.add_argument("--target-undertriage", type=unit_float, default=None, metavar="T",
                        help="Cel niedotriażowania reguły progi T (domyślnie z config, jak w 08)")
    parser.add_argument("--fit-population", choices=["vc", "all"], default=None,
                        help="Wiersze dopasowania progów (domyślnie z config, jak w 08)")
    parser.add_argument("--safety-net", action=argparse.BooleanOptionalAction, default=True,
                        help="Siatka bezpieczeństwa B2 reguły progi T (jak w 08)")
    parser.add_argument("--no-reference", action="store_true",
                        help="Bez odniesienia: zestaw modeli klasycznych z tym samym tagiem")
    parser.add_argument("--no-plots", action="store_true", help="Bez wykresów")
    parser.add_argument("--plan", action="store_true", help="Tylko plan: runy, warianty, szacunek czasu (bez liczenia)")
    parser.add_argument("--force", action="store_true", help="Policz od nowa także gotowe powtórzenia")
    parser.add_argument("--n-jobs", type=positive_int, default=None,
                        help="Wątki CPU (SOR_AI_N_JOBS; ustawiane przed importem config)")
    return parser


def run(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    apply_n_jobs(argv)
    args = build_parser().parse_args(argv)

    try:
        from src.analysis.common import AnalysisArgumentError
        from src.analysis.tfm_blend import (
            BlendConfig,
            BlendSkipped,
            analysis_path,
            decision_lines,
            run_tfm_blend,
        )
        from src.utils import config
        from src.utils.logger import get_logger

        log = get_logger("15_tfm_blend")
        cfg = BlendConfig(
            feature_set=args.feature_set, tag=args.tag, models=args.models, runs=args.runs, repeats=args.repeats,
            n_boot=args.bootstrap, methods=args.methods,
            seed=config.RANDOM_SEED if args.seed is None else args.seed,
            target=config.DECISION_TARGET_UNDERTRIAGE if args.target_undertriage is None else args.target_undertriage,
            safety_net=args.safety_net,
            fit_population=args.fit_population or config.DECISION_FIT_POPULATION,
            reference=not args.no_reference, plots=not args.no_plots, force=args.force,
        )
        try:
            results = run_tfm_blend(cfg, plan_only=args.plan)
        except BlendSkipped as exc:
            print(f"skipped: {exc}")
            log.info(f"Eksperyment B pominięty: {exc}")
            return 0
        except (AnalysisArgumentError, FileNotFoundError) as exc:
            log.error(str(exc))
            return 2
    except KeyboardInterrupt:
        print("Przerwano (Ctrl+C) — uruchom to samo polecenie, gotowe powtórzenia zostaną pominięte.")
        return 130
    if args.plan:
        print("\n".join(results["plan_lines"]))
        return 0
    root = analysis_path(results["feature_set"], results["tag"])
    print("\n".join(decision_lines(results)))
    print(f"Raport: {root / 'report.md'}")
    print(f"Rekomendacja: {root / 'recommendation.json'}")
    return 1 if results.get("failures") else 0


if __name__ == "__main__":
    sys.exit(run())
