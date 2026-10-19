"""
Krok 7: porównanie wytrenowanych modeli i materiał do ręcznego łączenia.

Dla najnowszego runa każdego modelu (albo wskazanych runów):
    1. Tabela metryk na VAL (argmax i reguła kosztowa) — wybór kandydatów.
    2. Różnorodność modeli na VAL (niezgodność, korelacja błędów).
    3. Punkt odniesienia: równowagowa średnia wszystkich modeli (VAL i TEST).
    4. Raport Markdown: runs/comparison_<czas>.md

Samo łączenie robisz ręcznie (src/ensemble): wagi dobierasz na VAL,
na TEST tylko sprawdzasz wybraną recepturę.

Użycie:
    python scripts/07_compare_runs.py                         # zestaw 'triage'
    python scripts/07_compare_runs.py --feature-set full
    python scripts/07_compare_runs.py --runs RUN_A RUN_B ...  # konkretne runy
    python scripts/07_compare_runs.py --include-samples       # także runy testowe na próbkach
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.ensemble import aligned_probas, blend, diversity, evaluate, latest_runs, load_run  # noqa: E402
from src.utils.config import RUNS_DIR  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

KEY_METRICS = [
    ("quadratic_weighted_kappa", "QWK"),
    ("auc_macro", "AUC"),
    ("log_loss", "log loss"),
    ("undertriage_rate", "undertriage"),
    ("overtriage_rate", "overtriage"),
    ("expected_cost", "koszt"),
]


def _md_table(df: pd.DataFrame) -> str:
    """DataFrame → tabela Markdown (bez zależności od `tabulate`)."""
    def fmt(v) -> str:
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _metrics_row(name: str, y, proba) -> dict:
    row = {"model": name}
    for rule in ("argmax", "cost"):
        m = evaluate(y, proba, rule)
        for key, label in KEY_METRICS:
            if rule == "cost" and key in ("auc_macro", "log_loss"):
                continue  # nie zależą od reguły decyzyjnej
            row[f"{label} [{rule}]"] = m.get(key)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Porównanie runów i materiał do łączenia modeli")
    parser.add_argument("--feature-set", default="triage")
    parser.add_argument("--runs", nargs="*", default=None, help="Nazwy katalogów w runs/")
    parser.add_argument("--include-samples", action="store_true",
                        help="Uwzględnij runy trenowane na próbkach (--sample)")
    args = parser.parse_args()

    if args.runs:
        runs = [load_run(r) for r in args.runs]
    else:
        runs = latest_runs(args.feature_set, include_samples=args.include_samples)
    if not runs:
        log.error("Brak runów. Najpierw: python scripts/03_train.py --model all")
        sys.exit(1)
    log.info("Runy: " + ", ".join(r.name for r in runs))

    y_val, p_val = aligned_probas(runs, "val")
    y_test, p_test = aligned_probas(runs, "test")

    table = pd.DataFrame([_metrics_row(name, y_val, p) for name, p in p_val.items()])
    table = table.sort_values("log loss [argmax]").reset_index(drop=True)
    div = diversity(p_val, y_val).sort_values("corr_errors")

    eq_val = blend(p_val)
    eq_test = blend(p_test)
    baseline = pd.DataFrame([
        {"zbiór": "val", **_metrics_row("średnia równowagowa", y_val, eq_val)},
        {"zbiór": "test", **_metrics_row("średnia równowagowa", y_test, eq_test)},
    ])

    pd.set_option("display.width", 200)
    print("\n=== Modele na VAL ===")
    print(table.round(4).to_string(index=False))
    print("\n=== Różnorodność na VAL (najmniej skorelowane błędy u góry) ===")
    print(div.round(3).to_string(index=False))
    print("\n=== Punkt odniesienia: równowagowa średnia wszystkich modeli ===")
    print(baseline.round(4).to_string(index=False))

    out = RUNS_DIR / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.md"
    lines = [
        f"# Porównanie modeli ({args.feature_set})",
        "",
        "Runy: " + ", ".join(f"`{r.name}`" for r in runs),
        "",
        "## Modele na VAL",
        "",
        _md_table(table),
        "",
        "## Różnorodność na VAL",
        "",
        "`corr_errors` bliskie 0 = modele mylą się na innych pacjentach (dobry materiał do łączenia).",
        "",
        _md_table(div),
        "",
        "## Punkt odniesienia: równowagowa średnia",
        "",
        _md_table(baseline),
        "",
        "Własne połączenie: `from src.ensemble import ...` (wagi i kalibracja na VAL, raport na TEST).",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"Raport: {out}")


if __name__ == "__main__":
    main()
