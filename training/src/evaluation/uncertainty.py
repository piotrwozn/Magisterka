"""
Uncertainty Quantification dla ensemble modeli triażowych.

Analizuje niepewność predykcji na poziomie:
  - Per-model:   confidence (max proba), entropy, margin (top1−top2)
  - Cross-model: agreement, conflict (różnica ≥2 stopnie MTS)

Użycie:
    python -m src.evaluation.uncertainty --model-paths models/xgboost_*.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils.config import CLASS_NAMES, MODELS_DIR, REPORTS_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)

CLASS_NAMES_PL = ["Czerwony", "Pomarańczowy", "Żółty", "Zielony", "Niebieski"]


# ─────────────────────────────────────────
# Metryki niepewności per-model
# ─────────────────────────────────────────
def _confidence(proba: np.ndarray) -> np.ndarray:
    """Maksymalne prawdopodobieństwo — pewność modelu."""
    return proba.max(axis=1)


def _entropy(proba: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Entropia rozkładu prawdopodobieństw — im wyższa, tym większa niepewność."""
    p = np.clip(proba, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _margin(proba: np.ndarray) -> np.ndarray:
    """Margines między top-1 a top-2 — mały margines = niepewność."""
    sorted_proba = np.sort(proba, axis=1)[:, ::-1]
    return sorted_proba[:, 0] - sorted_proba[:, 1]


def _predicted_class(proba: np.ndarray) -> np.ndarray:
    return proba.argmax(axis=1)


# ─────────────────────────────────────────
# Analiza cross-model (ensemble-level)
# ─────────────────────────────────────────
def compute_model_uq(
    proba: np.ndarray, y_true: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Metryki UQ dla pojedynczego modelu."""
    from sklearn.metrics import cohen_kappa_score

    preds = _predicted_class(proba)
    result = {
        "prediction": preds,
        "confidence": _confidence(proba),
        "entropy": _entropy(proba),
        "margin": _margin(proba),
    }
    for i in range(proba.shape[1]):
        result[f"proba_class_{i}"] = proba[:, i]

    if y_true is not None:
        result["correct"] = (preds == y_true).astype(float)
        result["accuracy"] = float(result["correct"].mean())
        result["qwk"] = float(cohen_kappa_score(y_true, preds, weights="quadratic"))
    return result


def compute_ensemble_uq(
    probas: dict[str, np.ndarray],
    y_true: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Metryki UQ dla całego ensemble.

    Parameters
    ----------
    probas : dict[str, np.ndarray]
        {model_name: np.ndarray shape (n, 5)}
    y_true : np.ndarray, optional

    Returns
    -------
    dict z kluczami:
        per_model      : {model_name: dict} — metryki per model
        ensemble       : dict — metryki ensemble (średnia prob)
        agreement      : np.ndarray (n,) — ile modeli zgadza się z większością
        unanimity      : np.ndarray (n,) — czy wszystkie modele zgodne
        max_disagreement: np.ndarray (n,) — max różnica między predykcjami (w stopniach MTS)
        conflict_mask  : np.ndarray (n,) — True gdy max_disagreement ≥ 2
        vote_matrix    : np.ndarray (n, 5) — liczba głosów per klasa
    """
    model_names = list(probas.keys())
    n_models = len(model_names)
    n_samples = next(iter(probas.values())).shape[0]

    # Per-model
    per_model = {}
    for name in model_names:
        per_model[name] = compute_model_uq(probas[name], y_true)

    # Ensemble (średnia prob)
    ensemble_proba = np.mean([probas[name] for name in model_names], axis=0)
    ensemble = compute_model_uq(ensemble_proba, y_true)
    ensemble_preds = ensemble["prediction"]

    # Voting — ile modeli głosuje na każdą klasę (one-hot encoding)
    all_preds_stacked = np.column_stack([per_model[name]["prediction"] for name in model_names])
    vote_matrix = (all_preds_stacked[:, :, None] == np.arange(5)).sum(axis=1)

    # Agreement: ile modeli głosuje na klasę większościową
    majority_class = vote_matrix.argmax(axis=1)
    majority_votes = vote_matrix.max(axis=1)
    agreement = majority_votes / n_models

    # Unanimity: wszystkie modele zgodne
    unanimity = majority_votes == n_models

    # Max disagreement: największa różnica między predykcjami modeli (w stopniach MTS)
    all_preds = np.column_stack([per_model[name]["prediction"] for name in model_names])
    max_disagreement = all_preds.max(axis=1) - all_preds.min(axis=1)

    # Conflict: różnica ≥ 2 stopnie
    conflict_mask = max_disagreement >= 2

    # Pairwise correlation między modelami
    pairwise_pred_qwk = {}
    pairwise_proba_r = {}
    pairwise_per_class_r: dict[str, dict[str, float]] = {}
    pairwise_cross_confusion: dict[str, list[list[int]]] = {}
    if n_models > 1:
        from scipy.stats import pearsonr
        from sklearn.metrics import cohen_kappa_score, confusion_matrix

        for i in range(n_models):
            for j in range(i + 1, n_models):
                ni, nj = model_names[i], model_names[j]
                pair_key = f"{ni} vs {nj}"

                # QWK między predykcjami
                pairwise_pred_qwk[pair_key] = float(cohen_kappa_score(
                    per_model[ni]["prediction"],
                    per_model[nj]["prediction"],
                    weights="quadratic",
                ))

                # Pearson r między spłaszczonymi probami
                flat_i = probas[ni].ravel()
                flat_j = probas[nj].ravel()
                r, _ = pearsonr(flat_i, flat_j)
                pairwise_proba_r[pair_key] = float(r)

                # Per-class probability correlation
                per_class_r: dict[str, float] = {}
                for cls in range(5):
                    rc, _ = pearsonr(probas[ni][:, cls], probas[nj][:, cls])
                    per_class_r[CLASS_NAMES[cls]] = float(rc)
                pairwise_per_class_r[pair_key] = per_class_r

                # Cross-confusion: gdy model A mówi klasa X, co mówi model B?
                cc = confusion_matrix(
                    per_model[ni]["prediction"],
                    per_model[nj]["prediction"],
                    labels=list(range(5)),
                )
                pairwise_cross_confusion[pair_key] = cc.tolist()

    result: dict[str, Any] = {
        "per_model": per_model,
        "ensemble": ensemble,
        "ensemble_proba": ensemble_proba,
        "vote_matrix": vote_matrix,
        "majority_class": majority_class,
        "agreement": agreement,
        "unanimity": unanimity,
        "max_disagreement": max_disagreement,
        "conflict_mask": conflict_mask,
        "pairwise_pred_qwk": pairwise_pred_qwk,
        "pairwise_proba_r": pairwise_proba_r,
        "pairwise_per_class_r": pairwise_per_class_r,
        "pairwise_cross_confusion": pairwise_cross_confusion,
        "n_models": n_models,
    }

    if y_true is not None:
        ensemble_correct = (ensemble_preds == y_true).astype(float)
        by_agreement = {}
        for level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            mask = agreement >= level
            if mask.sum() > 0:
                acc = ensemble_correct[mask].mean()
            else:
                acc = float("nan")
            by_agreement[f"≥{level:.0%}"] = {
                "count": int(mask.sum()),
                "accuracy": float(acc),
            }
        result["accuracy_by_agreement"] = by_agreement

    return result


# ─────────────────────────────────────────
# Raport tekstowy
# ─────────────────────────────────────────
def ensemble_report(result: dict[str, Any]) -> str:
    """Generuje czytelny raport UQ."""
    lines = []
    lines.append("=" * 60)
    lines.append("SOR-AI: Uncertainty Quantification Report")
    lines.append("=" * 60)
    lines.append("")

    n = len(result["agreement"])
    n_conflict = int(result["conflict_mask"].sum())
    n_unanimous = int(result["unanimity"].sum())

    lines.append(f"Próbek:             {n}")
    lines.append(f"Modeli w ensemble:   {len(result['per_model'])}")
    lines.append(f"")
    lines.append(f"--- Ensemble Agreement ---")
    lines.append(f"Pełna zgodność (unanimity): {n_unanimous:>6} ({n_unanimous/n*100:>5.1f}%)")
    lines.append(f"Średnia zgodność:           {result['agreement'].mean()*100:>5.1f}%")
    lines.append(f"Konflikt (różnica ≥2):      {n_conflict:>6} ({n_conflict/n*100:>5.1f}%)")
    lines.append(f"")

    # Distribution of agreement levels
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    lines.append(f"Rozkład zgodności:")
    for i in range(len(edges) - 1):
        mask = (result["agreement"] >= edges[i]) & (result["agreement"] < edges[i + 1])
        count = int(mask.sum())
        bar = "█" * (count * 50 // max(n, 1))
        lines.append(f"  [{edges[i]:.0%}–{edges[i+1]:.0%}) {count:>5} ({count/n*100:>4.1f}%) {bar}")
    # Top bin includes 1.0
    mask_top = result["agreement"] == 1.0
    count_top = int(mask_top.sum())
    bar_top = "█" * (count_top * 50 // max(n, 1))
    lines.append(f"  [100%         ] {count_top:>5} ({count_top/n*100:>4.1f}%) {bar_top}")
    lines.append(f"")

    # Per-model stats
    lines.append(f"--- Per-Model Summary ---")
    header = f"{'Model':<15s} {'QWK':>8s} {'Acc':>7s} {'Conf':>6s} {'Entropy':>8s} {'Margin':>6s}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, md in result["per_model"].items():
        qwk = md.get("qwk", float("nan"))
        acc = md.get("accuracy", float("nan"))
        lines.append(
            f"{name:<15s} "
            f"{qwk:>8.4f} "
            f"{acc:>7.4f} "
            f"{md['confidence'].mean():>6.3f} "
            f"{md['entropy'].mean():>8.4f} "
            f"{md['margin'].mean():>6.3f}"
        )

    ens = result["ensemble"]
    lines.append(f"  {'ENSEMBLE (avg prob)':<15s} "
                  f"{ens.get('qwk', float('nan')):>8.4f} "
                  f"{ens.get('accuracy', float('nan')):>7.4f} "
                  f"{ens['confidence'].mean():>6.3f} "
                  f"{ens['entropy'].mean():>8.4f} "
                  f"{ens['margin'].mean():>6.3f}")
    lines.append(f"")

    if "accuracy_by_agreement" in result:
        lines.append(f"--- Accuracy by Agreement Level ---")
        for level, data in result["accuracy_by_agreement"].items():
            lines.append(f"  Agreement {level:>5s}: n={data['count']:>6}, accuracy={data['accuracy']:.4f}")

    lines.append(f"")

    # Compact summary na koniec
    lines.append(f"═" * 50)
    lines.append(f"ENSEMBLE SUMMARY (modele: {', '.join(result['per_model'].keys())})")
    lines.append(f"═" * 50)
    for name, md in result["per_model"].items():
        lines.append(f"  {name:<15s}  QWK={md.get('qwk', float('nan')):.4f}  "
                      f"ACC={md.get('accuracy', float('nan')):.4f}  "
                      f"Conf={md['confidence'].mean():.3f}")
    ens = result["ensemble"]
    lines.append(f"  {'──' + '─'*13}")
    lines.append(f"  {'ENSEMBLE':<15s}  QWK={ens.get('qwk', float('nan')):.4f}  "
                  f"ACC={ens.get('accuracy', float('nan')):.4f}  "
                  f"Conf={ens['confidence'].mean():.3f}")
    lines.append(f"")

    # Pairwise correlation between models
    if result.get("pairwise_pred_qwk"):
        lines.append(f"─" * 50)
        lines.append(f"PAIRWISE MODEL CORRELATION")
        lines.append(f"─" * 50)
        lines.append(f"{'Pair':<30s} {'Pred QWK':>10s} {'Proba r':>10s}")
        lines.append(f"{'─' * 52}")
        for pair in result["pairwise_pred_qwk"]:
            q = result["pairwise_pred_qwk"][pair]
            r = result["pairwise_proba_r"].get(pair, float("nan"))
            lines.append(f"{pair:<30s} {q:>10.4f} {r:>10.4f}")
        lines.append(f"")

        # Per-class probability correlation
        if result.get("pairwise_per_class_r"):
            lines.append(f"  Per-class Proba r:")
            cls_header = f"{'Pair':<22s}"
            for cn in CLASS_NAMES:
                cls_header += f" {cn:>8s}"
            lines.append(cls_header)
            for pair, per_class in result["pairwise_per_class_r"].items():
                line = f"{pair:<22s}"
                for cn in CLASS_NAMES:
                    line += f" {per_class[cn]:>8.4f}"
                lines.append(line)
            lines.append(f"")

        # Cross-confusion matrices
        if result.get("pairwise_cross_confusion"):
            lines.append(f"  Cross-confusion (rows=model_A, cols=model_B):")
            for pair, cc in result["pairwise_cross_confusion"].items():
                models_in_pair = pair.split(" vs ")
                lines.append(f"  {pair}:")
                header = f"       "
                for cn in CLASS_NAMES:
                    header += f" {cn:>8s}"
                lines.append(header)
                for row_idx, row in enumerate(cc):
                    line = f"  {models_in_pair[0][:7]:>4s} {CLASS_NAMES[row_idx]:>4s}"
                    for val in row:
                        line += f" {val:>8d}"
                    lines.append(line)
                # Agreement %
                total = sum(sum(row) for row in cc)
                agree = sum(cc[i][i] for i in range(5))
                lines.append(f"  {' ':<11s} Zgodność: {agree/total*100:.1f}%")
                lines.append(f"")

    # Conflict examples
    conflict_idx = np.where(result["conflict_mask"])[0]
    if len(conflict_idx) > 0:
        lines.append(f"--- Konflikt (różnica ≥2 stopnie): {len(conflict_idx)} przypadków ---")
        for idx in conflict_idx[:10]:
            preds = {}
            for name in result["per_model"]:
                preds[name] = int(result["per_model"][name]["prediction"][idx])
            majority = int(result["majority_class"][idx])
            lines.append(f"  Sample {idx}: majority={CLASS_NAMES[majority]}, "
                         f"models={preds}, "
                         f"margin={result['ensemble']['margin'][idx]:.3f}")
        if len(conflict_idx) > 10:
            lines.append(f"  ... i {len(conflict_idx) - 10} więcej")

    return "\n".join(lines)


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
def load_models(model_paths: list[Path]) -> dict[str, Any]:
    """Ładuje modele z plików .joblib."""
    models = {}
    for path in model_paths:
        artifact = joblib.load(path)
        name = artifact.get("name", path.stem.split("_")[0])
        models[name] = artifact["model"]
        log.info(f"Załadowano '{name}': {path} ({type(artifact['model']).__name__})")
    return models


def compute_predictions(
    models: dict[str, Any],
    X: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Oblicza prawdopodobieństwa dla wszystkich modeli."""
    probas = {}
    for name, model in models.items():
        probas[name] = model.predict_proba(X)
        log.info(f"  {name}: proba shape {probas[name].shape}")
    return probas


# ─────────────────────────────────────────
# SHAP Analysis (opcjonalny)
# ─────────────────────────────────────────
SHAP_CLASS_NAMES = ["Red", "Orange", "Yellow", "Green", "Blue"]


def _model_supports_shap(model: Any, name: str) -> bool:
    """Sprawdza czy model ma TreeSHAP (natywny)."""
    import xgboost
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier

    if isinstance(model, (xgboost.XGBClassifier, lgb.LGBMClassifier)):
        return True
    if isinstance(model, RandomForestClassifier):
        return True
    if hasattr(model, "get_feature_importance"):
        from catboost import CatBoostClassifier
        if isinstance(model, CatBoostClassifier):
            return True
    return False


def _predict_shap_xgb(model: Any, X: pd.DataFrame) -> np.ndarray:
    """SHAP values przez XGBoost natywny predict (pred_contribs).

    Returns shape (n, n_features, n_classes).
    """
    import xgboost as _xgb
    booster = model.get_booster()
    contribs = booster.predict(_xgb.DMatrix(X), pred_contribs=True)
    # contribs shape: (n, n_classes, n_features+1)
    # → transpose to (n, n_features+1, n_classes) → drop bias → (n, n_features, n_classes)
    return np.transpose(contribs, (0, 2, 1))[:, :-1, :]


def _predict_shap_lgb(model: Any, X: pd.DataFrame) -> np.ndarray:
    """SHAP values przez LightGBM natywny predict (pred_contrib).

    LightGBM zwraca 2D array (n, (n_features+1) * n_classes) dla multiclass,
    gdzie pierwsze (n_features+1) kolumn to klasa 0, potem klasa 1, itd.
    Reshape do 3D → transpose do (n, n_features+1, n_classes) → drop bias.
    """
    import lightgbm as lgb
    contribs = model.booster_.predict(X, pred_contrib=True)
    n_features = X.shape[1]
    n_classes = model.booster_.num_class() if hasattr(model.booster_, 'num_class') else 5
    expected_cols = (n_features + 1) * n_classes
    if contribs.ndim == 2 and contribs.shape[1] == expected_cols:
        contribs = contribs.reshape(-1, n_classes, n_features + 1)
        contribs = np.transpose(contribs, (0, 2, 1))
    # contribs shape oczekiwana: (n, n_features+1, n_classes)
    return contribs[:, :-1, :]  # drop bias column


def _predict_shap_cb(model: Any, X: pd.DataFrame) -> np.ndarray:
    """SHAP values przez CatBoost natywny."""
    from catboost import Pool
    pool = Pool(X)
    # CatBoost zwraca (n, n_features+1, n_classes) lub (n, n_features+1)
    sv = model.get_feature_importance(data=pool, type="ShapValues")
    if sv.ndim == 3:
        return sv[:, :-1, :]  # (n, n_features, n_classes)
    # dla binary: (n, n_features+1) → (n, n_features, 1)
    return sv[:, :-1, np.newaxis]


def _predict_shap_rf(model: Any, X: pd.DataFrame) -> np.ndarray:
    """SHAP dla RandomForest — używa shap.TreeExplainer ale z małą próbką."""
    import shap
    # Tylko 200 samples dla RF (wolniejszy)
    X_small = X.sample(n=min(200, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model, data=X_small, feature_perturbation="interventional")
    sv = explainer.shap_values(X_small, check_additivity=False)
    # sv: (n_samples, n_features, n_classes) dla multiclass
    return sv


def compute_model_shap(
    model: Any,
    name: str,
    X_background: pd.DataFrame,
    X_eval: pd.DataFrame,
    feature_names: list[str],
    n_samples: int = 500,
) -> dict[str, Any] | None:
    """Oblicza SHAP values dla modelu na próbce danych.

    Używa natywnych metod modeli zamiast shap biblioteki dla wydajności.
    """
    import xgboost
    import lightgbm as lgb

    if not _model_supports_shap(model, name):
        return None

    # Próbkuj eval
    if len(X_eval) > n_samples:
        X_eval_shap = X_eval.sample(n=n_samples, random_state=42)
    else:
        X_eval_shap = X_eval

    log.info(f"  SHAP {name}: {len(X_eval_shap)} samples")

    try:
        if isinstance(model, xgboost.XGBClassifier):
            shap_values = _predict_shap_xgb(model, X_eval_shap)
        elif isinstance(model, lgb.LGBMClassifier):
            shap_values = _predict_shap_lgb(model, X_eval_shap)
        elif hasattr(model, "get_feature_importance"):
            shap_values = _predict_shap_cb(model, X_eval_shap)
        else:
            shap_values = _predict_shap_rf(model, X_eval_shap)
    except Exception as e:
        log.warning(f"  SHAP {name} failed: {e}")
        return None

    # shap_values shape: (n, n_features, n_classes)
    n_classes = shap_values.shape[2]

    # Top features per class
    top_per_class = {}
    for cls in range(n_classes):
        importances = np.abs(shap_values[:, :, cls]).mean(axis=0)
        top_idx = np.argsort(importances)[-10:][::-1]
        top_per_class[SHAP_CLASS_NAMES[cls]] = [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in top_idx
        ]

    # Global top features (mean |SHAP| across all classes)
    global_importances = np.abs(shap_values).mean(axis=(0, 2))
    global_top_idx = np.argsort(global_importances)[-10:][::-1]
    global_top = [
        {"feature": feature_names[i], "importance": float(global_importances[i])}
        for i in global_top_idx
    ]

    return {
        "model": name,
        "global_top_features": global_top,
        "top_features_per_class": top_per_class,
        "n_eval_samples": len(X_eval_shap),
    }


def shap_section(
    shap_results: dict[str, dict[str, Any] | None],
    n_models: int,
) -> list[str]:
    """Generuje sekcję SHAP do raportu."""
    lines = []
    available = {name: res for name, res in shap_results.items() if res is not None}
    if not available:
        return lines

    lines.append(f"")
    lines.append(f"─" * 60)
    lines.append(f"SHAP ANALYSIS — Top features per model")
    lines.append(f"─" * 60)
    lines.append(f"")

    # Global top features per model
    for name, res in available.items():
        lines.append(f"  {name}:")
        lines.append(f"  {'Feature':<30s} {'Importance':>10s}")
        lines.append(f"  {'─' * 42}")
        for feat in res["global_top_features"][:8]:
            lines.append(f"  {feat['feature']:<30s} {feat['importance']:>10.4f}")
        lines.append(f"")

    # Feature overlap between models
    all_features = {}
    for name, res in available.items():
        for feat in res["global_top_features"]:
            f = feat["feature"]
            if f not in all_features:
                all_features[f] = []
            all_features[f].append(name)

    # Features appearing in multiple models
    shared = {f: models for f, models in all_features.items() if len(models) > 1}
    if shared:
        lines.append(f"  Wspólne cechy w top-10 wielu modeli:")
        for feat, models in sorted(shared.items(),
                                    key=lambda x: len(x[1]), reverse=True)[:5]:
            lines.append(f"    {feat:<30s} → {', '.join(models)}")

    # Features unique to one model
    unique = {f: models[0] for f, models in all_features.items() if len(models) == 1}
    if unique:
        lines.append(f"  Cechy unikalne dla jednego modelu:")
        for feat, model_name in sorted(unique.items(),
                                        key=lambda x: x[1])[:8]:
            lines.append(f"    {feat:<30s} ← tylko {model_name}")

    lines.append(f"")
    return lines


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uncertainty Quantification dla ensemble modeli triażowych"
    )
    parser.add_argument(
        "--model-paths", "-m",
        nargs="+",
        type=Path,
        default=sorted(Path(MODELS_DIR).glob("*.joblib")),
        help="Ścieżki do plików modeli .joblib",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=REPORTS_DIR / "uncertainty_report.txt",
        help="Ścieżka pliku raportu",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Opcjonalnie: zapisz szczegółowe wyniki jako JSON",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Dodaj SHAP analysis (na próbce 500 testowych, ~1-3 min)",
    )
    parser.add_argument(
        "--shap-samples",
        type=int,
        default=500,
        help="Liczba próbek do SHAP (default: 500)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Ładuj dane testowe
    from src.data.preprocessing import build_feature_groups, split_features
    from src.models.train import load_splits

    log.info("Ładowanie danych testowych…")
    splits = load_splits()
    groups = build_feature_groups(splits["train"])
    X_test, y_test, _ = split_features(splits["test"], groups, "triage_only")
    feature_names = list(X_test.columns)
    log.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Ładuj modele
    log.info(f"Ładowanie modeli ({len(args.model_paths)})…")
    models = load_models(args.model_paths)

    # Predykcje
    log.info("Obliczanie predykcji…")
    probas = compute_predictions(models, X_test)

    # UQ
    log.info("Analiza Uncertainty Quantification…")
    result = compute_ensemble_uq(probas, y_true=y_test.values)

    # SHAP (opcjonalny)
    shap_results = {}
    if args.shap and len(models) > 0:
        log.info(f"SHAP analysis ({args.shap_samples} samples)…")
        X_train, y_train, _ = split_features(splits["train"], groups, "triage_only")
        bg = X_train.sample(n=min(200, len(X_train)), random_state=42)

        for name, model in models.items():
            sr = compute_model_shap(
                model, name, bg, X_test, feature_names,
                n_samples=args.shap_samples,
            )
            if sr is not None:
                shap_results[name] = sr
            else:
                shap_results[name] = None
                log.info(f"  {name}: SHAP pominięty (brak TreeSHAP)")

    # Raport
    report_parts = [ensemble_report(result)]
    if shap_results:
        report_parts.append("\n".join(shap_section(shap_results, len(models))))
    report = "".join(report_parts)

    print(report)

    # Zapisz raport
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"Raport zapisany: {args.output}")

    # JSON
    if args.json:
        serializable = _make_serializable(result)
        if shap_results:
            serializable["shap"] = {
                name: res for name, res in shap_results.items() if res is not None
            }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        log.info(f"JSON zapisany: {args.json}")


def _make_serializable(result: dict[str, Any]) -> dict[str, Any]:
    """Konwertuje tablice numpy na listy do JSON."""
    out = {}
    for k, v in result.items():
        if k == "per_model":
            out[k] = {
                name: {
                    mk: mv.tolist() if isinstance(mv, np.ndarray) else mv
                    for mk, mv in md.items()
                }
                for name, md in v.items()
            }
        elif k == "ensemble":
            out[k] = {
                mk: mv.tolist() if isinstance(mv, np.ndarray) else mv
                for mk, mv in v.items()
            }
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _make_serializable(v)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
