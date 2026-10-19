"""
Krok 5: Generowanie wyjaśnień (SHAP, reguły ESI, opcjonalnie Ollama) dla runa.

Dla losowo wybranych N pacjentów ze zbioru val (domyślnie) lub test generuje:
    1. Predykcję ESI 1–5 (reguła argmax i kosztowa) z kolorem TOPSOR i czasem oczekiwania.
    2. SHAP per pacjent + waterfall (metoda dobrana do modelu: TreeSHAP, CatBoost,
       liniowy dla regresji logistycznej, wkłady termów EBM).
    3. Regułową weryfikację parametrów życiowych (ESI, punkty A / B / D) i jej
       zgodność z predykcją modelu.
    4. (Opcjonalnie) Wyjaśnienie medyczne z lokalnego LLM (Ollama).
    5. Globalny wykres SHAP (średnie |SHAP| per poziom ESI + beeswarm dla ESI 1 i ESI 2).
Wszystko zapisywane jako JSON + PNG.

Użycie:
    python scripts/05_explain.py                                   # najnowszy run (zestaw 'triage')
    python scripts/05_explain.py --model-name xgboost --n-patients 10
    python scripts/05_explain.py --run 20261019_115324_xgboost_triage_smoke --split test
    python scripts/05_explain.py --model-name catboost --include-samples --esi 1 --use-ollama
    python scripts/05_explain.py --model path/do/model.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Dodaj root projektu do PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_split  # noqa: E402
from src.data.prepare import META_COLS, VITALS_COMPLETE_COL  # noqa: E402
from src.ensemble import list_runs, load_run  # noqa: E402
from src.evaluation.metrics import decide  # noqa: E402
from src.explain.esi_rules import check_consistency, rule_based_triage  # noqa: E402
from src.explain.ollama_medical import OllamaMedicalExplainer  # noqa: E402
from src.explain.shap_explainer import (  # noqa: E402
    ExplainerUnavailable,
    SHAPTriageExplainer,
    load_background,
    shap_summary_plot,
    shap_waterfall_plot,
)
from src.models import MODEL_REGISTRY, load_model, resolve_model_name  # noqa: E402
from src.utils.config import (  # noqa: E402
    CLASS_NAMES,
    CLASS_NAMES_PL,
    ESI_COL,
    HIGH_ACUITY_CLASSES,
    OLLAMA_DEFAULT_MODEL,
    REPORTS_DIR,
    SHAP_BACKGROUND_SIZE,
    SHAP_TOP_FEATURES,
    TOPSOR_COLORS,
    TOPSOR_MAX_WAIT_MINUTES,
)
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

PERMUTATION_SUMMARY_LIMIT = 50   # metoda permutacyjna jest wolna — mniejsza próbka globalna


# ─────────────────────────────────────────
# Model / run
# ─────────────────────────────────────────
def latest_run_with_model(
    feature_set: str = "triage",
    model_name: str | None = None,
    include_samples: bool = False,
):
    """Najnowszy run z zapisanym model.joblib (opcjonalnie danego modelu) albo None."""
    table = list_runs()
    if table.empty:
        return None
    table = table[table["feature_set"] == feature_set]
    if model_name is not None:
        table = table[table["model"] == resolve_model_name(model_name)]
    if not include_samples:
        table = table[table["train_sample"].isna() & table["eval_sample"].isna()]
    for name in sorted(table["run"], reverse=True):     # nazwa zaczyna się od znacznika czasu
        run = load_run(name)
        if (run.path / "model.joblib").exists():
            return run
    return None


def resolve_run(args: argparse.Namespace) -> tuple[str, Path]:
    """Zwraca (nazwa, ścieżka model.joblib) wg --model / --run / najnowszego runa."""
    if args.model:
        path = Path(args.model)
        return path.stem, path
    if args.run:
        run = load_run(args.run)
    else:
        run = latest_run_with_model(args.feature_set, args.model_name, args.include_samples)
        if run is None:
            hint = "" if args.include_samples else " (runy na próbkach są pomijane — dodaj --include-samples)"
            raise SystemExit(
                f"Brak runów z zapisanym modelem dla zestawu '{args.feature_set}'"
                + (f" i modelu '{args.model_name}'" if args.model_name else "")
                + f"{hint}. Wytrenuj model: python scripts/03_train.py --model xgboost"
            )
    path = run.path / "model.joblib"
    if not path.exists():
        raise SystemExit(f"Run {run.name} nie ma zapisanego modelu ({path})")
    return run.name, path


def level_info(class_idx: int) -> dict:
    color = TOPSOR_COLORS[class_idx]
    return {
        "esi": class_idx + 1,
        "class": CLASS_NAMES[class_idx],
        "label_pl": CLASS_NAMES_PL[class_idx],
        "topsor_color": color,
        "max_wait_minutes": TOPSOR_MAX_WAIT_MINUTES[color],
    }


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wyjaśnienia predykcji ESI (SHAP, reguły ESI, Ollama) dla wybranego runa",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_argument_group("model")
    src.add_argument("--run", default=None, help="Nazwa lub ścieżka katalogu runa w runs/")
    src.add_argument("--model", default=None, help="Ścieżka do model.joblib (zamiast runa)")
    src.add_argument("--model-name", default=None,
                     help=f"Najnowszy run tego modelu ({', '.join(MODEL_REGISTRY)}; aliasy lr/lgbm/xgb/cb/rf)")
    src.add_argument("--feature-set", default="triage", choices=["triage", "full"],
                     help="Zestaw cech przy wyborze najnowszego runa")
    src.add_argument("--include-samples", action="store_true",
                     help="Uwzględnij runy trenowane na próbkach (--sample / --eval-sample)")

    data = parser.add_argument_group("pacjenci")
    data.add_argument("--split", default="val", choices=["val", "test"], help="Zbiór, z którego losujemy pacjentów")
    data.add_argument("--n-patients", type=int, default=5, help="Ilu pacjentów wyjaśnić")
    data.add_argument("--esi", type=int, default=None, choices=[1, 2, 3, 4, 5],
                      help="Tylko pacjenci z danym (prawdziwym) poziomem ESI")
    data.add_argument("--seed", type=int, default=42)

    expl = parser.add_argument_group("wyjaśnienia")
    expl.add_argument("--rule", default="argmax", choices=["argmax", "cost"],
                      help="Reguła decyzyjna, której klasę wyjaśnia SHAP")
    expl.add_argument("--top", type=int, default=SHAP_TOP_FEATURES, help="Ile top cech SHAP zapisać")
    expl.add_argument("--summary-size", type=int, default=500,
                      help="Pacjentów do globalnego wykresu SHAP (0 = pomiń)")
    expl.add_argument("--background", type=int, default=SHAP_BACKGROUND_SIZE,
                      help="Próbka tła z train (tylko regresja logistyczna / metoda permutacyjna)")
    expl.add_argument("--use-ollama", action="store_true", help="Włącz wyjaśnienia medyczne LLM (Ollama)")
    expl.add_argument("--ollama-model", default=OLLAMA_DEFAULT_MODEL)
    parser.add_argument("--out-dir", default=None,
                        help="Katalog wyników (domyślnie results/reports/explanations_<run>_<czas>)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_name, model_path = resolve_run(args)

    log.info("=" * 60)
    log.info(f"GENEROWANIE WYJAŚNIEŃ: {run_name}")
    log.info("=" * 60)

    # 1. Model
    log.info(f"[1/5] Ładowanie modelu: {model_path}")
    model = load_model(model_path)          # klasa wg nazwy modelu w artefakcie
    if not model.feature_names:
        raise SystemExit("Model nie ma zapisanych nazw cech (feature_names) — wytrenuj go ponownie.")
    features = list(model.feature_names)
    log.info(f"    {model.name} | zestaw cech: {model.feature_set} | cech: {len(features)}")

    # 2. Pacjenci z wybranego splitu
    log.info(f"[2/5] Ładowanie danych ({args.split})…")
    meta = [c for c in META_COLS if c not in features]
    try:
        df = load_split(args.split, columns=meta + features)
    except Exception as e:
        raise SystemExit(f"Nie udało się wczytać splitu '{args.split}' z cechami modelu: {e}") from e
    X_all = df[features].astype(np.float32)      # globalny SHAP: cały split (bez filtra ESI)
    candidates = df if args.esi is None else df[df[ESI_COL] == args.esi]
    if candidates.empty:
        raise SystemExit(f"Brak pacjentów z ESI {args.esi} w zbiorze {args.split}")
    patients = candidates.sample(n=min(args.n_patients, len(candidates)), random_state=args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPORTS_DIR / f"explanations_{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. SHAP
    log.info("[3/5] Inicjalizacja SHAP…")
    shap_explainer = None
    method = SHAPTriageExplainer.method_for(model)
    try:
        background = None
        if method in ("linear", "permutation"):
            log.info(f"    Metoda '{method}' — próbka tła z train ({args.background} wierszy)")
            background = load_background(features, n=args.background, seed=args.seed)
        shap_explainer = SHAPTriageExplainer(model, background_data=background).fit()
    except ExplainerUnavailable as e:
        log.warning(f"    SHAP niedostępny: {e} — kontynuuję bez SHAP")

    if shap_explainer is not None and args.summary_size > 0:
        size = args.summary_size
        if shap_explainer.method == "permutation":
            size = min(size, PERMUTATION_SUMMARY_LIMIT)
        log.info(f"    Globalny SHAP na {min(size, len(X_all))} pacjentach…")
        sample = X_all.sample(n=min(size, len(X_all)), random_state=args.seed)
        values = shap_explainer.explain_dataset(sample)
        shown = shap_explainer.feature_frame(sample)
        shap_summary_plot(values, shown, save_path=out_dir / "shap_summary_all_levels.png")
        for idx in HIGH_ACUITY_CLASSES:
            shap_summary_plot(values, shown, class_idx=idx,
                              save_path=out_dir / f"shap_beeswarm_{CLASS_NAMES[idx].lower()}.png")

    # 4. Ollama (opcjonalnie)
    ollama = None
    if args.use_ollama:
        log.info(f"    Łączenie z Ollama ({args.ollama_model})…")
        ollama = OllamaMedicalExplainer(model_name=args.ollama_model)
        if not ollama.is_available():
            log.warning("    Ollama niedostępne — pomijam wyjaśnienia LLM.")
            log.warning(f"    Dostępne modele: {ollama.list_available_models()}")
            ollama = None

    # 5. Pacjenci
    log.info(f"[4/5] Wyjaśnianie {len(patients)} pacjentów ({args.split})…")
    all_explanations = []
    for i, (idx, row) in enumerate(patients.iterrows(), start=1):
        patient_row = X_all.loc[[idx]]
        patient_data = row.to_dict()
        true_idx = int(row[ESI_COL]) - 1
        proba = np.asarray(model.predict_proba(patient_row), dtype=float)
        preds = {rule: int(decide(proba, rule)[0]) for rule in ("argmax", "cost")}
        explained_idx = preds[args.rule]
        probabilities = dict(zip(CLASS_NAMES, proba[0].round(4).tolist()))

        log.info(f"  ─── Pacjent {i}/{len(patients)} (row_id={int(row['row_id'])}) ───")
        log.info(f"    Prawdziwy: ESI {true_idx + 1} | argmax: ESI {preds['argmax'] + 1} | "
                 f"kosztowa: ESI {preds['cost'] + 1}")

        # SHAP
        shap_exp, shap_info = None, None
        if shap_explainer is not None:
            try:
                shap_exp = shap_explainer.explain_patient(patient_row, top_n=args.top, class_idx=explained_idx)
                shap_info = {
                    "method": shap_exp["method"],
                    "output_space": shap_exp["output_space"],
                    "explained_class": shap_exp["explained_class"],
                    "base_value": shap_exp["base_value"],
                    "top_features_for": shap_exp["top_features_for"],
                    "top_features_against": shap_exp["top_features_against"],
                }
                shap_waterfall_plot(
                    shap_explainer, patient_row, class_idx=explained_idx,
                    save_path=out_dir / f"patient_{i}_row{int(row['row_id'])}_waterfall.png",
                )
            except Exception as e:      # wyjaśnienie nie może przerwać całego raportu
                log.warning(f"    SHAP błąd: {e}")
                shap_info = {"error": str(e)}
        else:
            shap_info = {"error": "SHAP niedostępny dla tego modelu"}

        # Reguły ESI (parametry życiowe + skargi)
        rule_check = rule_based_triage(patient_data)
        consistency = {
            rule: check_consistency(pred, rule_check["suggested_class_idx"]) for rule, pred in preds.items()
        }
        log.info(f"    Reguły: {rule_check['label_pl']} | {consistency[args.rule]['verdict']}")

        # Ollama
        ollama_text = None
        if ollama is not None:
            try:
                ollama_text = ollama.explain(
                    patient_data=patient_data,
                    predicted_class=CLASS_NAMES[explained_idx],
                    predicted_class_pl=CLASS_NAMES_PL[explained_idx],
                    probabilities=probabilities,
                    shap_explanation=shap_exp,
                    rule_check=rule_check,
                )
            except Exception as e:
                log.warning(f"    Ollama błąd: {e}")

        all_explanations.append({
            "row_id": int(row["row_id"]),
            "site": row.get("site"),
            "admitted": None if pd.isna(row.get("admitted")) else int(row["admitted"]),
            VITALS_COMPLETE_COL: (
                None if pd.isna(row.get(VITALS_COMPLETE_COL)) else int(row[VITALS_COMPLETE_COL])
            ),
            "true": level_info(true_idx),
            "prediction": {rule: level_info(pred) for rule, pred in preds.items()},
            "explained_rule": args.rule,
            "probabilities": probabilities,
            "shap": shap_info,
            "rule_check": {
                "suggested_esi": rule_check["suggested_esi"],
                "topsor_color": rule_check["suggested_category"],
                "label_pl": rule_check["label_pl"],
                "decision_point": rule_check["decision_point"],
                "advisory": rule_check["advisory"],
                "max_wait_minutes": rule_check["max_wait_minutes"],
                "triggered_rules": rule_check["triggered_rules"],
                "missing_vitals": rule_check["missing_vitals"],
                "notes": rule_check["notes"],
            },
            "consistency": consistency,
            "ollama_explanation": ollama_text,
        })

    # 6. Zapis
    log.info("[5/5] Zapis…")
    payload = {
        "run": run_name,
        "model": model.name,
        "feature_set": model.feature_set,
        "split": args.split,
        "esi_filter": args.esi,
        "shap_method": shap_explainer.method if shap_explainer else None,
        "created": timestamp,
        "patients": all_explanations,
    }
    json_path = out_dir / "explanations.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    log.info("✓ Wyjaśnienia zapisane:")
    log.info(f"  Katalog:          {out_dir}")
    log.info(f"  Wyjaśnienia JSON: {json_path}")


if __name__ == "__main__":
    main()
