"""
Krok 7: porównanie wytrenowanych modeli i materiał do ręcznego łączenia — WYŁĄCZNIE na VAL.

Dla najnowszego ukończonego runa każdego modelu (albo wskazanych runów):
    1. Przebudowa runs/leaderboard.csv ze wszystkich ukończonych runów (metryki VAL).
    2. Tabela metryk na VAL (reguła argmax i kosztowa) + podgrupa z kompletem parametrów
       życiowych (`vitals_complete == 1`: HR, SBP, RR zmierzone — populacja jak w TOPSOR),
       kalibracja: ECE (top-label) i temperatura T dopasowana na VAL (T > 1 = model zbyt pewny).
    3. Opcjonalnie (`--bootstrap N`): 95% przedziały ufności na VAL dla QWK i undertriage.
    4. Różnorodność modeli na VAL (niezgodność, korelacja błędów).
    5. Punkt odniesienia: równowagowa średnia wybranych modeli (VAL).
    6. Raport Markdown: runs/comparison_<zestaw>_<czas>.md (albo --out).

Zbiór testowy nie jest tu pokazywany: wybór modeli, wag i progów odbywa się na VAL, a test
oceniasz raz, na końcu, dla wybranej receptury (src.ensemble).

Domyślnie pomijane: runy na próbkach (--include-samples), z wagami klas (--include-weighted),
bez tuningu (--include-untuned), nieukończone (bez DONE.json), bez pliku modelu, z innej wersji
danych. Jawna lista --runs nie jest filtrowana.

Następny krok po wyborze receptury na VAL: scripts/08_final_report.py (raz, na TEST).

Użycie:
    python scripts/07_compare_runs.py                         # zestaw 'triage'
    python scripts/07_compare_runs.py --feature-set full
    python scripts/07_compare_runs.py --tag standard          # tylko runy z profilu 'standard'
    python scripts/07_compare_runs.py --runs RUN_A RUN_B ...  # konkretne runy (np. z run_all.py)
    python scripts/07_compare_runs.py --include-samples       # także runy testowe na próbkach
    python scripts/07_compare_runs.py --include-untuned       # także runy bez tuningu (-notune)
    python scripts/07_compare_runs.py --tag standard --bootstrap 1000   # + przedziały ufności (VAL)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.ensemble import (  # noqa: E402
    aligned_probas,
    blend,
    diversity,
    evaluate,
    fit_temperature,
    latest_runs,
    load_run,
    resolve_feature_set,
)
from src.evaluation.statistics import bootstrap_samples, ece_top_label, format_ci  # noqa: E402
from src.utils import config  # noqa: E402
from src.utils.fileio import write_text_atomic  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

KEY_METRICS = [
    ("quadratic_weighted_kappa", "QWK"),
    ("auc_macro", "AUC"),
    ("log_loss", "log loss"),
    ("undertriage_rate", "undertriage"),
    ("critical_miss_rate", "crit. miss"),
    ("overtriage_rate", "overtriage"),
    ("expected_cost", "koszt"),
]
RULE_FREE_METRICS = ("auc_macro", "log_loss")     # liczone z prawdopodobieństw — bez reguły decyzyjnej
# Podgrupa vitals_complete == 1 — skrót (pełne metryki: 04_evaluate.py)
VC_METRICS = [
    ("quadratic_weighted_kappa", "QWK vc"),
    ("undertriage_rate", "undertriage vc"),
]
# --bootstrap: przedziały ufności na VAL (szybkie porównanie — tylko dwie kluczowe metryki)
CI_METRICS = [
    ("quadratic_weighted_kappa", "QWK"),
    ("undertriage_rate", "undertriage"),
]
BOOTSTRAP_SEED = 42


def _md_table(df: pd.DataFrame) -> str:
    """DataFrame → tabela Markdown (bez zależności od `tabulate`)."""
    def fmt(v) -> str:
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def metrics_row(name: str, y: np.ndarray, proba: np.ndarray, vc_mask: np.ndarray | None = None) -> dict:
    """Wiersz tabeli: AUC i log loss (bez reguły), metryki obu reguł decyzyjnych + podgrupa vc."""
    row: dict = {"model": name}
    for rule in ("argmax", "cost"):
        m = evaluate(y, proba, rule)
        for key, label in KEY_METRICS:
            if key in RULE_FREE_METRICS:
                if rule == "argmax":
                    row[label] = m.get(key)
                continue
            row[f"{label} [{rule}]"] = m.get(key)
    if vc_mask is not None and vc_mask.any():
        for rule in ("argmax", "cost"):
            m = evaluate(y[vc_mask], proba[vc_mask], rule)
            for key, label in VC_METRICS:
                row[f"{label} [{rule}]"] = m.get(key)
    # Kalibracja na VAL: ECE top-label (0 = idealnie) i temperatura T (1 = bez zmian)
    row["ECE"] = ece_top_label(y, proba)
    row["T"] = fit_temperature(proba, y)
    return row


def ci_table(
    y: np.ndarray,
    probas: dict[str, np.ndarray],
    n_boot: int,
    mask: np.ndarray | None = None,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Tabela 'wartość [95% CI]' dla QWK i undertriage (obie reguły); te same losowania dla wszystkich modeli."""
    if mask is not None:
        y = y[mask]
        probas = {k: p[mask] for k, p in probas.items()}
    res = bootstrap_samples(y, probas, ("argmax", "cost"), [m for m, _ in CI_METRICS], n_boot=n_boot, seed=seed)
    rows = []
    for name in probas:
        row: dict = {"model": name}
        for rule in ("argmax", "cost"):
            ci = res.ci(name, rule).set_index("metric")
            for key, label in CI_METRICS:
                r = ci.loc[key]
                percent = key != "quadratic_weighted_kappa"
                row[f"{label} [{rule}]"] = format_ci(r["value"], r["ci_low"], r["ci_high"],
                                                      digits=3, percent=percent)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Porównanie runów na VAL i materiał do łączenia modeli")
    parser.add_argument("--feature-set", default="triage", help="triage | full (stare nazwy też działają)")
    parser.add_argument("--runs", nargs="*", default=None, help="Nazwy katalogów w runs/ albo ścieżki")
    parser.add_argument("--tag", default=None, help="Tylko runy z tym tagiem (np. nazwa profilu)")
    parser.add_argument("--include-samples", action="store_true",
                        help="Uwzględnij runy trenowane na próbkach (--sample)")
    parser.add_argument("--include-weighted", action="store_true",
                        help="Uwzględnij runy z wagami klas (nieskalibrowane prawdopodobieństwa)")
    parser.add_argument("--include-untuned", action="store_true",
                        help="Uwzględnij runy bez tuningu (parametry domyślne, tag <profil>-notune)")
    parser.add_argument("--out", type=Path, default=None, help="Ścieżka raportu .md")
    parser.add_argument("--bootstrap", type=int, default=0, metavar="N",
                        help="Losowania bootstrap do 95%% przedziałów ufności QWK i undertriage na VAL (0 = bez)")
    args = parser.parse_args(argv)
    if args.bootstrap < 0:
        parser.error("--bootstrap musi być ≥ 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    feature_set = resolve_feature_set(args.feature_set)

    from src.models.train import rebuild_leaderboard

    try:
        board = rebuild_leaderboard()
        log.info(f"Leaderboard (VAL, wszystkie ukończone runy): {board}")
    except Exception as exc:  # noqa: BLE001 — porównanie działa także bez leaderboardu
        log.warning(f"Nie udało się przebudować leaderboardu: {exc}")

    if args.runs:
        runs = [load_run(r) for r in args.runs]
        other = sorted({r.info.get("feature_set") for r in runs} - {feature_set})
        if other:
            log.warning(f"Wskazane runy mają też inne zestawy cech: {other}")
    else:
        runs = latest_runs(feature_set, include_samples=args.include_samples, tag=args.tag,
                           include_weighted=args.include_weighted, require_tuned=not args.include_untuned)
    if not runs:
        log.error("Brak ukończonych runów spełniających filtry (próbki, wagi klas, bez tuningu — zob. "
                  "--include-samples / --include-weighted / --include-untuned). Najpierw: "
                  "python scripts/run_all.py (albo python scripts/03_train.py --model all --tune)")
        return 1
    log.info("Runy: " + ", ".join(r.name for r in runs))

    y_val, p_val, vc_val = aligned_probas(runs, "val", return_vc=True)

    table = pd.DataFrame([metrics_row(name, y_val, p, vc_val) for name, p in p_val.items()])
    table = table.sort_values("log loss").reset_index(drop=True)
    div = diversity(p_val, y_val).sort_values("corr_errors").reset_index(drop=True)
    baseline = pd.DataFrame([metrics_row("średnia równowagowa", y_val, blend(p_val), vc_val)])
    ci_all = ci_vc = None
    if args.bootstrap:
        with_baseline = {**p_val, "średnia równowagowa": blend(p_val)}
        ci_all = ci_table(y_val, with_baseline, args.bootstrap)
        if vc_val is not None and vc_val.any():
            ci_vc = ci_table(y_val, with_baseline, args.bootstrap, mask=vc_val)

    pd.set_option("display.width", 220)
    print(f"\n=== Modele na VAL ({feature_set}, n={len(y_val):,}"
          + (f", vitals complete: {int(vc_val.sum()):,}" if vc_val is not None else "") + ") ===")
    print(table.round(4).to_string(index=False))
    if len(div):
        print("\n=== Różnorodność na VAL (najmniej skorelowane błędy u góry) ===")
        print(div.round(3).to_string(index=False))
    print("\n=== Punkt odniesienia: równowagowa średnia modeli (VAL) ===")
    print(baseline.round(4).to_string(index=False))
    if ci_all is not None:
        print(f"\n=== 95% przedziały ufności na VAL (QWK: bootstrap {args.bootstrap}×, stratyfikowany po ESI; "
              f"undertriage: Clopper–Pearson z k/n) ===")
        print(ci_all.to_string(index=False))
        if ci_vc is not None:
            print("\n--- podgrupa vc ---")
            print(ci_vc.to_string(index=False))

    out = args.out or (Path(config.RUNS_DIR) / f"comparison_{feature_set}_{datetime.now():%Y%m%d_%H%M%S}.md")
    lines = [
        f"# Porównanie modeli — {feature_set} (VAL)",
        "",
        "Runy: " + ", ".join(f"`{r.name}`" for r in runs),
        "",
        f"VAL: {len(y_val):,} wizyt"
        + (f"; podgrupa *vitals complete* (HR, SBP i RR zmierzone przy triażu): {int(vc_val.sum()):,}"
           if vc_val is not None else "; brak kolumny vitals_complete (stare runy)") + ".",
        "",
        "## Modele na VAL",
        "",
        _md_table(table),
        "",
        "Reguła *cost*: poziom minimalizujący oczekiwany koszt (niedotriażowanie 3× droższe). "
        "Kolumny *vc*: podgrupa z kompletem parametrów życiowych (populacja jak w TOPSOR). "
        "*ECE*: średni błąd kalibracji najbardziej prawdopodobnego poziomu (0 = deklarowana pewność "
        "równa trafności). *T*: temperatura dopasowana na VAL (T > 1 — model zbyt pewny siebie, "
        "T < 1 — zbyt ostrożny; argmax się nie zmienia).",
        "",
    ]
    if ci_all is not None:
        lines += [
            f"## 95% przedziały ufności na VAL (bootstrap {args.bootstrap}×)",
            "",
            "QWK: bootstrap percentylowy, losowanie osobno w każdej klasie ESI; te same losowania dla "
            "wszystkich modeli. Undertriage (odsetek pacjentów): dokładny przedział Clopper–Pearsona "
            "z liczebności k/n. Nakładające się przedziały NIE dowodzą braku różnicy — do porównania dwóch "
            "modeli służy sparowany bootstrap (`src.evaluation.statistics.paired_bootstrap_diff`, raport 08).",
            "",
            _md_table(ci_all),
            "",
        ]
        if ci_vc is not None:
            lines += ["Podgrupa *vc*:", "", _md_table(ci_vc), ""]
    if len(div):
        lines += [
            "## Różnorodność na VAL",
            "",
            "`corr_errors` bliskie 0 = modele mylą się na innych pacjentach (dobry materiał do łączenia).",
            "",
            _md_table(div),
            "",
        ]
    lines += [
        "## Punkt odniesienia: równowagowa średnia (VAL)",
        "",
        _md_table(baseline),
        "",
        "Własne połączenie: `from src.ensemble import ...` — wagi, kalibracja i progi na VAL; "
        "test oceniasz raz, na końcu, tą samą recepturą: `python scripts/08_final_report.py` "
        "(przedziały ufności, kalibracja, plik wag dla backendu).",
    ]
    write_text_atomic(out, "\n".join(lines) + "\n")
    log.info(f"Raport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
