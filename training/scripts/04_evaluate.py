"""
Krok 4: Pełna ewaluacja zapisanego modelu.

Wykonuje:
    1. Załadowanie modelu z dysku.
    2. Predykcję na test set.
    3. Pełną ewaluację (QWK, AUC, undertriage, confusion matrix).
    4. Generowanie wszystkich wykresów.
    5. Zapis raportu do results/.

Użycie:
    python scripts/04_evaluate.py --model models/xgboost_20250510_143000.joblib
    python scripts/04_evaluate.py --model-name xgboost  # użyje najnowszego modelu
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Dodaj root projektu do PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import build_feature_groups, split_features  # noqa: E402
from src.evaluation.metrics import full_evaluation  # noqa: E402
from src.evaluation.visualizations import (  # noqa: E402
    plot_calibration,
    plot_confusion_matrix,
    plot_feature_importances,
    plot_roc_curves,
)
from src.models import MODEL_REGISTRY  # noqa: E402
from src.models.base import BaseTriageModel  # noqa: E402
from src.utils.config import (  # noqa: E402
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TEST_PARQUET,
)
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


def find_latest_model(model_name: str) -> Path:
    """Znajduje najnowszy zapisany model o danej nazwie."""
    candidates = sorted(
        MODELS_DIR.glob(f"{model_name}*.joblib"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Nie znaleziono żadnego modelu '{model_name}*.joblib' w {MODELS_DIR}"
        )
    log.info(f"Znaleziono najnowszy model: {candidates[0]}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ewaluacja zapisanego modelu na test set")
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Ścieżka do pliku .joblib modelu",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Nazwa modelu — użyje najnowszego z models/",
    )
    parser.add_argument(
        "--feature-set", "-f",
        choices=["triage_only", "full", "top"],
        default="triage_only",
    )
    parser.add_argument(
        "--test-data", type=str, default=str(TEST_PARQUET),
        help="Plik parquet z test set",
    )
    args = parser.parse_args()

    # Wybór modelu
    if args.model:
        model_path = Path(args.model)
    elif args.model_name:
        model_path = find_latest_model(args.model_name)
    else:
        parser.error("Podaj --model <ścieżka> lub --model-name <nazwa>")

    log.info("=" * 60)
    log.info(f"EWALUACJA MODELU: {model_path.name}")
    log.info("=" * 60)

    # 1. Załaduj model
    log.info(f"\n[1/4] Ładowanie modelu: {model_path}")
    # Ładujemy bez zgadywania klasy — używamy BaseTriageModel.load + sprawdzamy nazwę
    model = BaseTriageModel.load(model_path)
    model_name = model.name
    log.info(f"      Model: {model_name} (run_id: {model.run_id})")
    log.info(f"      Liczba cech: {len(model.feature_names) if model.feature_names else 'unknown'}")

    # 2. Załaduj test set
    log.info(f"\n[2/4] Ładowanie test set: {args.test_data}")
    df_test = pd.read_parquet(args.test_data)
    groups = build_feature_groups(df_test)
    X_test, y_test, _ = split_features(df_test, groups, feature_set=args.feature_set)

    # Wyrównaj kolumny do modelu (jeśli model był trenowany na innym zestawie)
    if model.feature_names:
        missing = [c for c in model.feature_names if c not in X_test.columns]
        if missing:
            log.warning(f"Brakuje {len(missing)} kolumn — uzupełniam zerami: {missing[:5]}…")
            for col in missing:
                X_test[col] = 0
        X_test = X_test[model.feature_names]

    log.info(f"      Test set: {X_test.shape[0]:,} × {X_test.shape[1]}")

    # 3. Predykcje + ewaluacja
    log.info(f"\n[3/4] Predykcje + ewaluacja…")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = full_evaluation(y_test, y_pred, y_proba, prefix="test", print_report=True)

    # 4. Wykresy
    log.info(f"\n[4/4] Generowanie wykresów…")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_dir = FIGURES_DIR / f"{model_name}_eval_{timestamp}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        y_test, y_pred,
        save_path=fig_dir / "confusion_matrix.png",
        title=f"Confusion Matrix — {model_name}",
    )
    plot_roc_curves(
        y_test, y_proba,
        save_path=fig_dir / "roc_curves.png",
        title=f"ROC Curves — {model_name}",
    )
    plot_calibration(
        y_test, y_proba,
        save_path=fig_dir / "calibration.png",
        title=f"Calibration — {model_name}",
    )

    importances = model.feature_importances()
    if importances is not None:
        plot_feature_importances(
            importances,
            save_path=fig_dir / "feature_importances.png",
            title="Top 20 najważniejszych cech",
        )

    log.info(f"\n✓ Ewaluacja zakończona!")
    log.info(f"  Wykresy: {fig_dir}")

    # Zapis metryk do CSV
    report_csv = REPORTS_DIR / f"{model_name}_eval_{timestamp}.csv"
    metrics_flat = {k: v for k, v in metrics.items() if not isinstance(v, list)}
    pd.DataFrame([metrics_flat]).to_csv(report_csv, index=False)
    log.info(f"  Metryki CSV: {report_csv}")


if __name__ == "__main__":
    main()
