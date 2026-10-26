"""
Krok 14 (eksperyment A, modele TFM): krzywa uczenia przy MAŁEJ liczbie lokalnych etykiet — TabPFN-3.5
i TabICLv2 „z pudełka” vs LightGBM strojony na tej samej podpróbie (układ sparowany).

Użycie (z katalogu training/):
    python scripts/14_small_data_curve.py --feature-set triage --tag thorough --models tabpfn,tabicl
    python scripts/14_small_data_curve.py --tag thorough --plan               # plan i szacunek czasu, bez liczenia
    python scripts/14_small_data_curve.py --tag smoke --sizes 500,1000 --repeats 2 --eval-rows 2000 \\
        --lgbm-trials 3 --foreign-rows 2000 --transport-rows 10000          # szybki test

Scenariusze: pooled (wszystkie oddziały; ocena na --eval-rows wierszach VAL) i site_local (tylko oddział
--local-site; ocena na wszystkich jego wierszach VAL). Modele: lightgbm_tuned (Optuna na podpróbie),
lightgbm_default, TFM z --models (+ w site_local: TFM z --foreign-rows wierszami innych oddziałów i LightGBM
przeniesiony z innych oddziałów). LightGBM liczony zawsze; brak torch / biblioteki / wag → dany TFM pominięty
(linia `skipped: …` na wyjściu i w raporcie). Na CPU jednostki TFM z kontekstem > TFM_CPU_MAX_CONTEXT są
pomijane (GPU wymagane do pełnego eksperymentu).

Wynik: runs/analyses/small_data__<cechy>__<tag>/ (results.json, units.csv, report.md, summary.md, figures/,
samples/, preds/, units/). Wznawialne: gotowe jednostki są pomijane (--force: od nowa; próby zostają).
Szczegóły: src/analysis/small_data.py, docs/TFM.md.

Kody wyjścia: 0 = OK (także gdy TFM pominięte — linie `skipped:`), 1 = któraś jednostka nieudana,
2 = złe argumenty / brak danych, 130 = Ctrl+C.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.cli import apply_n_jobs, positive_int  # noqa: E402


def non_negative_int(value: str) -> int:
    number = int(str(value).replace("_", ""))
    if number < 0:
        raise argparse.ArgumentTypeError(f"oczekiwano liczby ≥ 0, podano {value}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Eksperyment A: TFM (TabPFN, TabICL) vs strojony LightGBM przy małej liczbie danych",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--feature-set", "-f", default="triage", help="Zestaw cech")
    parser.add_argument("--tag", default=None,
                        help="Tag analizy (katalog small_data__<cechy>__<tag>) i runa lightgbm, którego parametry "
                             "dostaje model przeniesiony; domyślnie profil domyślny (standard)")
    parser.add_argument("--models", default="tabpfn,tabicl",
                        help="Porównywane TFM po przecinku (aliasy pfn, icl; 'tfm' = oba; 'none' = tylko LightGBM); "
                             "LightGBM (strojony i domyślny) liczony zawsze")
    parser.add_argument("--scenarios", default="pooled,site_local", help="pooled, site_local (po przecinku)")
    parser.add_argument("--sizes", default="500,1000,2000,5000,10000,50000", help="Rozmiary podprób n (≥ 20)")
    parser.add_argument("--repeats", default="10,10,10,5,5,3",
                        help="Powtórzenia dla kolejnych n (jedna wartość = dla wszystkich)")
    parser.add_argument("--eval-rows", type=non_negative_int, default=20_000,
                        help="pooled: warstwowa próba wierszy VAL do oceny (0 = cały VAL)")
    parser.add_argument("--local-eval-rows", type=non_negative_int, default=0,
                        help="site_local: próba wierszy VAL oddziału (0 = wszystkie wiersze VAL oddziału)")
    parser.add_argument("--local-site", default=None, help="Oddział scenariusza site_local (domyślnie HOLDOUT_SITE)")
    parser.add_argument("--foreign-rows", type=non_negative_int, default=10_000,
                        help="site_local: wiersze innych oddziałów dołączane do kontekstu TFM (0 = bez +foreign)")
    parser.add_argument("--transport-rows", type=non_negative_int, default=100_000,
                        help="site_local: wiersze innych oddziałów dla LightGBM przeniesionego (0 = bez)")
    parser.add_argument("--lgbm-trials", type=positive_int, default=40, help="Próby Optuny na jednostkę LightGBM")
    parser.add_argument("--bootstrap", type=non_negative_int, default=1_000,
                        help="Losowania dwupoziomowego bootstrapu (0 = tylko przedziały t)")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto", help="Urządzenie TFM")
    parser.add_argument("--seed", type=int, default=None, help="Seed losowania prób (domyślnie config.RANDOM_SEED)")
    parser.add_argument("--plan", action="store_true", help="Tylko plan: jednostki i orientacyjny czas (bez liczenia)")
    parser.add_argument("--force", action="store_true", help="Policz od nowa także gotowe jednostki (próby zostają)")
    parser.add_argument("--n-jobs", type=positive_int, default=None,
                        help="Wątki CPU (SOR_AI_N_JOBS; ustawiane przed importem config)")
    return parser


def run(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    apply_n_jobs(argv)
    args = build_parser().parse_args(argv)

    from src.utils.hardware import disable_power_throttling

    disable_power_throttling()
    try:
        from src.analysis.common import AnalysisArgumentError
        from src.analysis.small_data import SmallDataConfig, run_small_data
        from src.utils import config
        from src.utils.logger import get_logger

        log = get_logger("14_small_data_curve")
        cfg = SmallDataConfig(
            feature_set=args.feature_set, tag=args.tag, models=args.models, scenarios=args.scenarios,
            sizes=args.sizes, repeats=args.repeats, eval_rows=args.eval_rows, local_eval_rows=args.local_eval_rows,
            local_site=args.local_site or config.HOLDOUT_SITE, foreign_rows=args.foreign_rows,
            transport_rows=args.transport_rows, lgbm_trials=args.lgbm_trials, n_boot=args.bootstrap,
            device=args.device, seed=config.RANDOM_SEED if args.seed is None else args.seed, force=args.force,
        )
        try:
            results = run_small_data(cfg, plan_only=args.plan)
        except (AnalysisArgumentError, FileNotFoundError) as exc:
            log.error(str(exc))
            return 2
    except KeyboardInterrupt:
        print("Przerwano (Ctrl+C) — uruchom to samo polecenie, gotowe jednostki zostaną pominięte.")
        return 130
    for model, reason in (results.get("skipped_models") or {}).items():
        print(f"skipped: {model} — {reason}")
    if args.plan:
        print("\n".join(results["plan_lines"]))
        return 0
    from src.analysis.small_data import analysis_path

    root = analysis_path(results["feature_set"], results["tag"])
    print(f"Raport: {root / 'report.md'}")
    print("\n".join(results.get("decision") or []))
    return 1 if results.get("failures") else 0


if __name__ == "__main__":
    sys.exit(run())
