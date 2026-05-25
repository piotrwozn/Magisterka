"""
Krok 5: Generowanie wyjaśnień (SHAP, MTS rules, opcjonalnie Ollama).

Dla losowo wybranych N pacjentów z test set generuje:
    1. SHAP per-pacjent + waterfall plot.
    2. MTS rule check.
    3. (Opcjonalnie) Ollama medical reasoning.
    4. Zapis wszystkich wyjaśnień jako JSON + HTML/PNG.

Użycie:
    python scripts/05_explain.py --model models/xgboost_*.joblib
    python scripts/05_explain.py --model-name xgboost --n-patients 10 --use-ollama
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Dodaj root projektu do PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import build_feature_groups, split_features  # noqa: E402
from src.explain.mts_rules import check_consistency, rule_based_triage  # noqa: E402
from src.explain.ollama_medical import OllamaMedicalExplainer  # noqa: E402
from src.explain.shap_explainer import (  # noqa: E402
    SHAPTriageExplainer,
    shap_summary_plot,
    shap_waterfall_plot,
)
from src.models import MODEL_REGISTRY  # noqa: E402
from src.models.base import BaseTriageModel  # noqa: E402
from src.utils.config import (  # noqa: E402
    CLASS_NAMES,
    CLASS_NAMES_PL,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TEST_PARQUET,
    TRAIN_PARQUET,
)
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


def find_latest_model(model_name: str) -> Path:
    candidates = sorted(
        MODELS_DIR.glob(f"{model_name}*.joblib"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Brak modelu '{model_name}*.joblib' w {MODELS_DIR}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generowanie wyjaśnień dla pacjentów")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--feature-set", default="triage_only", choices=["triage_only", "full", "top"])
    parser.add_argument("--n-patients", type=int, default=5, help="Ilu pacjentów wyjaśnić")
    parser.add_argument("--use-ollama", action="store_true", help="Włącz medical reasoning (Ollama)")
    parser.add_argument("--ollama-model", default="mistral")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Model
    if args.model:
        model_path = Path(args.model)
    elif args.model_name:
        model_path = find_latest_model(args.model_name)
    else:
        parser.error("Podaj --model lub --model-name")

    log.info("=" * 60)
    log.info(f"GENEROWANIE WYJAŚNIEŃ: {model_path.name}")
    log.info("=" * 60)

    # 1. Załaduj model
    log.info(f"\n[1/5] Ładowanie modelu: {model_path}")
    model = BaseTriageModel.load(model_path)

    # 2. Załaduj train (background) + test (pacjenci do wyjaśnienia)
    log.info(f"\n[2/5] Ładowanie danych…")
    df_train = pd.read_parquet(TRAIN_PARQUET)
    df_test = pd.read_parquet(TEST_PARQUET)
    groups = build_feature_groups(df_train)
    X_train, _, _ = split_features(df_train, groups, feature_set=args.feature_set)
    X_test, y_test, _ = split_features(df_test, groups, feature_set=args.feature_set)

    if model.feature_names:
        for col in model.feature_names:
            if col not in X_train.columns:
                X_train[col] = 0
            if col not in X_test.columns:
                X_test[col] = 0
        X_train = X_train[model.feature_names]
        X_test = X_test[model.feature_names]

    # 3. SHAP — globalny + per pacjent
    log.info(f"\n[3/5] Inicjalizacja SHAP…")
    background_sample = X_train.sample(min(200, len(X_train)), random_state=args.seed)
    shap_explainer = SHAPTriageExplainer(
        model=model,
        background_data=background_sample,
        class_names=CLASS_NAMES,
    )
    shap_explainer.fit()

    # Globalny summary plot (na próbce 1000)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = FIGURES_DIR / f"explanations_{model.name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir = REPORTS_DIR / f"explanations_{model.name}_{timestamp}"
    json_dir.mkdir(parents=True, exist_ok=True)

    log.info("    Generowanie globalnego SHAP summary…")
    sample_for_summary = X_test.sample(min(1000, len(X_test)), random_state=args.seed)
    shap_values_global = shap_explainer.explain_dataset(sample_for_summary)
    shap_summary_plot(
        shap_values_global,
        sample_for_summary,
        save_path=out_dir / "shap_summary_global.png",
    )

    # 4. Pacjenci — losowa próbka
    log.info(f"\n[4/5] Wyjaśnianie {args.n_patients} losowych pacjentów…")
    test_indices = X_test.sample(args.n_patients, random_state=args.seed).index.tolist()

    ollama = None
    if args.use_ollama:
        log.info(f"    Łączenie z Ollama ({args.ollama_model})…")
        ollama = OllamaMedicalExplainer(model_name=args.ollama_model)
        if not ollama.is_available():
            log.warning(f"    Ollama niedostępne — pomijam medical reasoning.")
            log.warning(f"    Dostępne modele: {ollama.list_available_models()}")
            ollama = None

    all_explanations = []

    for i, idx in enumerate(test_indices):
        log.info(f"\n  ─── Pacjent {i + 1} / {args.n_patients} (idx={idx}) ───")

        patient_row = X_test.loc[[idx]]
        patient_data = patient_row.iloc[0].to_dict()
        true_label = int(y_test.loc[idx])

        # SHAP
        shap_exp = shap_explainer.explain_patient(patient_row)
        log.info(f"    Predykcja: {shap_exp['predicted_class']}, true: {CLASS_NAMES[true_label]}")

        # Waterfall plot
        shap_waterfall_plot(
            shap_explainer,
            patient_row,
            class_idx=shap_exp["predicted_class_idx"],
            save_path=out_dir / f"patient_{i+1}_idx{idx}_waterfall.png",
        )

        # MTS rules
        rule_check = rule_based_triage(patient_data)
        consistency = check_consistency(
            ml_prediction=shap_exp["predicted_class_idx"],
            rule_prediction=rule_check["suggested_class_idx"],
        )

        # Ollama (opcjonalnie)
        ollama_text = None
        if ollama is not None:
            try:
                ollama_text = ollama.explain(
                    patient_data=patient_data,
                    predicted_class=shap_exp["predicted_class"],
                    predicted_class_pl=CLASS_NAMES_PL[shap_exp["predicted_class_idx"]],
                    probabilities=shap_exp["probabilities"],
                    shap_explanation=shap_exp,
                    rule_check=rule_check,
                )
            except Exception as e:
                log.warning(f"    Ollama błąd: {e}")
                ollama_text = None

        explanation = {
            "patient_idx": int(idx),
            "true_class": CLASS_NAMES[true_label],
            "predicted_class": shap_exp["predicted_class"],
            "probabilities": shap_exp["probabilities"],
            "top_features_for": shap_exp["top_features_for"],
            "top_features_against": shap_exp["top_features_against"],
            "rule_check": {
                "suggested_category": rule_check["suggested_category"],
                "max_wait_minutes": rule_check["max_wait_minutes"],
                "triggered_rules": rule_check["triggered_rules"],
            },
            "consistency": consistency,
            "ollama_explanation": ollama_text,
        }
        all_explanations.append(explanation)

    # 5. Zapis JSON-a z wszystkimi wyjaśnieniami
    log.info(f"\n[5/5] Zapis…")
    json_path = json_dir / "explanations.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_explanations, f, indent=2, ensure_ascii=False, default=str)

    log.info(f"\n✓ Wyjaśnienia zapisane:")
    log.info(f"  Wykresy:          {out_dir}")
    log.info(f"  Wyjaśnienia JSON: {json_path}")


if __name__ == "__main__":
    main()
