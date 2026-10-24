"""
Uncertainty Quantification dla zestawu modeli ESI (5 klas: ESI 1–5).

Analizuje niepewność predykcji na poziomie:
  - Per-model:   confidence (max proba), entropy, margin (top1−top2)
  - Cross-model: agreement, conflict (różnica ≥2 poziomy ESI między modelami)

Źródła prawdopodobieństw:
  - runy z runs/ (domyślnie: najnowszy run z tuningiem każdego modelu) — zapisane
    predictions_<split>.parquet, bez ponownego liczenia;
  - pliki model.joblib (--model-paths) — predykcje liczone na nowo na wskazanym splicie,
    każdy model na własnym zestawie cech z artefaktu.

Protokół: analizy robocze na `val`; `--split test` wyłącznie do raportu końcowego.

Użycie:
    python -m src.evaluation.uncertainty                                  # najnowsze runy 'triage', val
    python -m src.evaluation.uncertainty --runs RUN_A RUN_B --split test
    python -m src.evaluation.uncertainty --include-untuned                # także runy bez tuningu (-notune)
    python -m src.evaluation.uncertainty --include-samples --include-untuned --sample 3000  # runy z próbek
    python -m src.evaluation.uncertainty --model-paths runs/*/model.joblib --shap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import (
    CLASS_NAMES,
    N_CLASSES,
    RANDOM_SEED,
    REPORTS_DIR,
    RUNS_DIR,
    SHAP_BACKGROUND_SIZE,
    SHAP_TOP_FEATURES,
    TARGET_COL,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


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
    """Metryki UQ dla pojedynczego modelu (predykcja = argmax)."""
    from sklearn.metrics import cohen_kappa_score

    preds = _predicted_class(proba)
    result = {
        "prediction": preds,
        "confidence": _confidence(proba),
        "entropy": _entropy(proba),
        "margin": _margin(proba),
    }
    for i in range(proba.shape[1]):
        result[f"proba_{CLASS_NAMES[i].lower()}" if i < len(CLASS_NAMES) else f"proba_class_{i}"] = proba[:, i]

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
    Metryki UQ dla całego zestawu modeli.

    Parameters
    ----------
    probas : dict[str, np.ndarray]
        {model_name: np.ndarray shape (n, 5)}
    y_true : np.ndarray, optional
        Klasy 0–4 (= ESI 1–5).

    Returns
    -------
    dict z kluczami:
        per_model       : {model_name: dict} — metryki per model
        ensemble        : dict — metryki ensemble (średnia prob)
        agreement       : np.ndarray (n,) — odsetek modeli zgodnych z większością
        unanimity       : np.ndarray (n,) — czy wszystkie modele zgodne
        max_disagreement: np.ndarray (n,) — max różnica między predykcjami (w poziomach ESI)
        conflict_mask   : np.ndarray (n,) — True gdy max_disagreement ≥ 2
        vote_matrix     : np.ndarray (n, 5) — liczba głosów per klasa
    """
    model_names = list(probas.keys())
    n_models = len(model_names)

    # Per-model
    per_model = {}
    for name in model_names:
        per_model[name] = compute_model_uq(probas[name], y_true)

    # Ensemble (średnia prob)
    ensemble_proba = np.mean([probas[name] for name in model_names], axis=0)
    ensemble = compute_model_uq(ensemble_proba, y_true)
    ensemble_preds = ensemble["prediction"]

    # Voting — ile modeli głosuje na każdą klasę (one-hot encoding)
    all_preds = np.column_stack([per_model[name]["prediction"] for name in model_names])
    vote_matrix = (all_preds[:, :, None] == np.arange(N_CLASSES)).sum(axis=1)

    # Agreement: ile modeli głosuje na klasę większościową
    majority_class = vote_matrix.argmax(axis=1)
    majority_votes = vote_matrix.max(axis=1)
    agreement = majority_votes / n_models

    # Unanimity: wszystkie modele zgodne
    unanimity = majority_votes == n_models

    # Max disagreement: największa różnica między predykcjami modeli (w poziomach ESI)
    max_disagreement = all_preds.max(axis=1) - all_preds.min(axis=1)

    # Conflict: różnica ≥ 2 poziomy ESI
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
                r, _ = pearsonr(probas[ni].ravel(), probas[nj].ravel())
                pairwise_proba_r[pair_key] = float(r)

                # Per-class probability correlation
                per_class_r: dict[str, float] = {}
                for cls in range(N_CLASSES):
                    rc, _ = pearsonr(probas[ni][:, cls], probas[nj][:, cls])
                    per_class_r[CLASS_NAMES[cls]] = float(rc)
                pairwise_per_class_r[pair_key] = per_class_r

                # Cross-confusion: gdy model A mówi ESI X, co mówi model B?
                cc = confusion_matrix(
                    per_model[ni]["prediction"],
                    per_model[nj]["prediction"],
                    labels=list(range(N_CLASSES)),
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
            acc = ensemble_correct[mask].mean() if mask.sum() > 0 else float("nan")
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
    lines.append("SOR-AI: Uncertainty Quantification Report (ESI 1–5)")
    lines.append("=" * 60)
    lines.append("")

    n = len(result["agreement"])
    n_conflict = int(result["conflict_mask"].sum())
    n_unanimous = int(result["unanimity"].sum())

    lines.append(f"Próbek:              {n}")
    lines.append(f"Modeli w zestawie:   {len(result['per_model'])}")
    lines.append("")
    lines.append("--- Zgodność modeli ---")
    lines.append(f"{'Pełna zgodność (unanimity):':<31s}{n_unanimous:>6} ({n_unanimous/n*100:>5.1f}%)")
    lines.append(f"{'Średnia zgodność:':<31s}{result['agreement'].mean()*100:>6.1f}%")
    lines.append(f"{'Konflikt (różnica ≥2 poziomy):':<31s}{n_conflict:>6} ({n_conflict/n*100:>5.1f}%)")
    lines.append("")

    # Rozkład poziomów zgodności
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    lines.append("Rozkład zgodności:")
    for i in range(len(edges) - 1):
        mask = (result["agreement"] >= edges[i]) & (result["agreement"] < edges[i + 1])
        count = int(mask.sum())
        bar = "█" * (count * 50 // max(n, 1))
        lines.append(f"  [{edges[i]:.0%}–{edges[i+1]:.0%}) {count:>5} ({count/n*100:>4.1f}%) {bar}")
    # Ostatni przedział obejmuje 1.0
    count_top = int((result["agreement"] == 1.0).sum())
    bar_top = "█" * (count_top * 50 // max(n, 1))
    lines.append(f"  [100%         ] {count_top:>5} ({count_top/n*100:>4.1f}%) {bar_top}")
    lines.append("")

    # Per-model stats
    lines.append("--- Podsumowanie per model (predykcja = argmax) ---")
    header = f"{'Model':<22s} {'QWK':>8s} {'Acc':>7s} {'Conf':>6s} {'Entropy':>8s} {'Margin':>6s}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, md in result["per_model"].items():
        lines.append(
            f"{name:<22s} "
            f"{md.get('qwk', float('nan')):>8.4f} "
            f"{md.get('accuracy', float('nan')):>7.4f} "
            f"{md['confidence'].mean():>6.3f} "
            f"{md['entropy'].mean():>8.4f} "
            f"{md['margin'].mean():>6.3f}"
        )

    ens = result["ensemble"]
    lines.append(f"{'ŚREDNIA (avg prob)':<22s} "
                 f"{ens.get('qwk', float('nan')):>8.4f} "
                 f"{ens.get('accuracy', float('nan')):>7.4f} "
                 f"{ens['confidence'].mean():>6.3f} "
                 f"{ens['entropy'].mean():>8.4f} "
                 f"{ens['margin'].mean():>6.3f}")
    lines.append("")

    if "accuracy_by_agreement" in result:
        lines.append("--- Accuracy średniej wg poziomu zgodności ---")
        for level, data in result["accuracy_by_agreement"].items():
            lines.append(f"  Zgodność {level:>5s}: n={data['count']:>6}, accuracy={data['accuracy']:.4f}")
        lines.append("")

    # Pairwise correlation between models
    if result.get("pairwise_pred_qwk"):
        lines.append("─" * 60)
        lines.append("KORELACJA MODELI (PARAMI)")
        lines.append("─" * 60)
        lines.append(f"{'Para':<40s} {'Pred QWK':>10s} {'Proba r':>10s}")
        lines.append("─" * 62)
        for pair, q in result["pairwise_pred_qwk"].items():
            r = result["pairwise_proba_r"].get(pair, float("nan"))
            lines.append(f"{pair:<40s} {q:>10.4f} {r:>10.4f}")
        lines.append("")

        # Per-class probability correlation
        if result.get("pairwise_per_class_r"):
            lines.append("  Korelacja prawdopodobieństw per poziom ESI (Pearson r):")
            lines.append(f"{'Para':<40s}" + "".join(f" {cn:>8s}" for cn in CLASS_NAMES))
            for pair, per_class in result["pairwise_per_class_r"].items():
                lines.append(f"{pair:<40s}" + "".join(f" {per_class[cn]:>8.4f}" for cn in CLASS_NAMES))
            lines.append("")

        # Cross-confusion matrices
        if result.get("pairwise_cross_confusion"):
            lines.append("  Cross-confusion (wiersze = model A, kolumny = model B):")
            for pair, cc in result["pairwise_cross_confusion"].items():
                lines.append(f"  {pair}:")
                lines.append(f"  {'':>6s}" + "".join(f" {cn:>8s}" for cn in CLASS_NAMES))
                for row_idx, row in enumerate(cc):
                    lines.append(f"  {CLASS_NAMES[row_idx]:>6s}" + "".join(f" {val:>8d}" for val in row))
                total = sum(sum(row) for row in cc)
                agree = sum(cc[i][i] for i in range(len(cc)))
                lines.append(f"  {'':>6s} Zgodność: {agree / max(total, 1) * 100:.1f}%")
                lines.append("")

    # Conflict examples
    conflict_idx = np.where(result["conflict_mask"])[0]
    if len(conflict_idx) > 0:
        lines.append(f"--- Konflikt (różnica ≥2 poziomy ESI): {len(conflict_idx)} przypadków ---")
        for idx in conflict_idx[:10]:
            preds = {
                name: CLASS_NAMES[int(md["prediction"][idx])]
                for name, md in result["per_model"].items()
            }
            majority = int(result["majority_class"][idx])
            lines.append(f"  Wiersz {idx}: większość={CLASS_NAMES[majority]}, "
                         f"modele={preds}, "
                         f"margin={result['ensemble']['margin'][idx]:.3f}")
        if len(conflict_idx) > 10:
            lines.append(f"  ... i {len(conflict_idx) - 10} więcej")

    return "\n".join(lines)


# ─────────────────────────────────────────
# Źródła prawdopodobieństw: runy i modele
# ─────────────────────────────────────────
def _sample_positions(y: np.ndarray, n: int | None, seed: int = RANDOM_SEED) -> np.ndarray:
    """Stratyfikowana próbka pozycji wierszy (cała tablica, gdy n jest None lub ≥ len)."""
    positions = np.arange(len(y))
    if n is None or n >= len(y):
        return positions
    from sklearn.model_selection import train_test_split

    sample, _ = train_test_split(positions, train_size=n, random_state=seed, stratify=y)
    return np.sort(sample)


def probas_from_runs(
    runs: list,
    split: str = "val",
    sample: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Prawdopodobieństwa z zapisanych predykcji runów (wspólne wiersze, ta sama kolejność)."""
    from src.ensemble import aligned_probas

    y, probas = aligned_probas(runs, split)
    pos = _sample_positions(y, sample)
    return y[pos], {name: p[pos] for name, p in probas.items()}


def load_models(model_paths: list[Path]) -> dict[str, Any]:
    """
    Ładuje modele z plików model.joblib (artefakty BaseTriageModel.save).

    Klucz = nazwa modelu (z artefaktu); przy duplikatach — nazwa katalogu runa (albo pliku).
    Klasa modelu wybierana przez `src.models.load_model` (bez prób GPU).
    """
    from src.models import load_model

    loaded: list[tuple[str, Path, Any]] = []
    for path in model_paths:
        path = Path(path)
        model = load_model(path)
        name = model.name
        loaded.append((name, path, model))
        log.info(f"Załadowano '{name}' ({model.feature_set}): {path}")

    models: dict[str, Any] = {}
    for name, path, model in loaded:
        duplicate = sum(n == name for n, _, _ in loaded) > 1
        key = (path.parent.name if path.name == "model.joblib" else path.stem) if duplicate else name
        models[key] = model
    return models


def load_eval_frame(models: dict[str, Any], split: str = "val", sample: int | None = None) -> pd.DataFrame:
    """Split z kolumnami potrzebnymi wszystkim modelom ('full' ⊇ 'triage')."""
    from src.data.dataset import load_splits, sample_frame

    feature_sets = {m.feature_set or "triage" for m in models.values()}
    feature_set = "triage" if feature_sets == {"triage"} else "full"
    df = load_splits((split,), feature_set=feature_set)[split]
    return sample_frame(df, sample).reset_index(drop=True)


def model_features(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    """Macierz cech dokładnie w układzie z treningu modelu."""
    from src.data.dataset import get_xy

    X, _ = get_xy(df, model.feature_set or "triage")
    return X[model.feature_names] if model.feature_names else X


def compute_predictions(models: dict[str, Any], df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Oblicza prawdopodobieństwa dla wszystkich modeli (każdy na swoim zestawie cech)."""
    probas = {}
    for name, model in models.items():
        probas[name] = model.predict_proba(model_features(model, df))
        log.info(f"  {name}: proba shape {probas[name].shape}")
    return probas


# ─────────────────────────────────────────
# SHAP Analysis (opcjonalny)
# ─────────────────────────────────────────
def _model_supports_shap(model: Any) -> bool:
    """Czy surowy estymator ma natywny TreeSHAP (XGBoost, LightGBM, CatBoost, RF)."""
    import lightgbm as lgb
    import xgboost
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier

    return isinstance(
        model, (xgboost.XGBClassifier, lgb.LGBMClassifier, CatBoostClassifier, RandomForestClassifier)
    )


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
    contribs = model.booster_.predict(X, pred_contrib=True)
    n_features = X.shape[1]
    n_classes = model.booster_.num_model_per_iteration()
    expected_cols = (n_features + 1) * n_classes
    if contribs.ndim == 2 and contribs.shape[1] == expected_cols:
        contribs = contribs.reshape(-1, n_classes, n_features + 1)
        contribs = np.transpose(contribs, (0, 2, 1))
    # contribs shape oczekiwana: (n, n_features+1, n_classes)
    return contribs[:, :-1, :]  # drop bias column


def _predict_shap_cb(model: Any, X: pd.DataFrame) -> np.ndarray:
    """SHAP values przez CatBoost natywny.

    Multiclass: CatBoost zwraca (n, n_classes, n_features+1) → (n, n_features, n_classes).
    """
    from catboost import Pool
    sv = model.get_feature_importance(data=Pool(X), type="ShapValues")
    if sv.ndim == 3:
        return np.transpose(sv, (0, 2, 1))[:, :-1, :]
    # dla binary: (n, n_features+1) → (n, n_features, 1)
    return sv[:, :-1, np.newaxis]


def _predict_shap_rf(model: Any, X: pd.DataFrame, background: pd.DataFrame | None = None) -> np.ndarray:
    """SHAP dla RandomForest — shap.TreeExplainer.

    Bez tła: 'tree_path_dependent' (szybki, rozkład z liczności liści drzew).
    Z tłem: 'interventional' — dokładniejszy, ale wolny (minuty dla 500 drzew).
    """
    import shap
    if background is None:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    else:
        explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    sv = explainer.shap_values(X, check_additivity=False)
    if isinstance(sv, list):   # starsze wersje shap: lista (n, n_features) per klasa
        sv = np.stack(sv, axis=-1)
    return sv  # (n, n_features, n_classes)


def compute_model_shap(
    model: Any,
    name: str,
    X_eval: pd.DataFrame,
    feature_names: list[str],
    n_samples: int = 500,
    X_background: pd.DataFrame | None = None,
    top_k: int = SHAP_TOP_FEATURES,
) -> dict[str, Any] | None:
    """Oblicza SHAP values dla surowego estymatora na próbce danych.

    Używa natywnych metod modeli zamiast biblioteki shap (poza RF) dla wydajności.
    Modele bez TreeSHAP (logreg, EBM) → None.
    """
    import lightgbm as lgb
    import xgboost
    from catboost import CatBoostClassifier

    if not _model_supports_shap(model):
        return None

    # RF przez bibliotekę shap jest wolny — mniejsza próbka
    is_rf = not isinstance(model, (xgboost.XGBClassifier, lgb.LGBMClassifier, CatBoostClassifier))
    n = min(n_samples, SHAP_BACKGROUND_SIZE) if is_rf else n_samples
    X_eval_shap = X_eval.sample(n=n, random_state=RANDOM_SEED) if len(X_eval) > n else X_eval

    log.info(f"  SHAP {name}: {len(X_eval_shap)} próbek")

    try:
        if isinstance(model, xgboost.XGBClassifier):
            shap_values = _predict_shap_xgb(model, X_eval_shap)
        elif isinstance(model, lgb.LGBMClassifier):
            shap_values = _predict_shap_lgb(model, X_eval_shap)
        elif isinstance(model, CatBoostClassifier):
            shap_values = _predict_shap_cb(model, X_eval_shap)
        else:
            background = None
            if X_background is not None:
                background = X_background.sample(
                    n=min(SHAP_BACKGROUND_SIZE, len(X_background)), random_state=RANDOM_SEED
                )
            shap_values = _predict_shap_rf(model, X_eval_shap, background)
    except Exception as e:
        log.warning(f"  SHAP {name} nie powiódł się: {e}")
        return None

    # shap_values shape: (n, n_features, n_classes)
    n_classes = shap_values.shape[2]

    # Top cechy per poziom ESI
    top_per_class = {}
    for cls in range(n_classes):
        importances = np.abs(shap_values[:, :, cls]).mean(axis=0)
        top_idx = np.argsort(importances)[-top_k:][::-1]
        top_per_class[CLASS_NAMES[cls]] = [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in top_idx
        ]

    # Globalne top cechy (średnie |SHAP| po wszystkich klasach)
    global_importances = np.abs(shap_values).mean(axis=(0, 2))
    global_top_idx = np.argsort(global_importances)[-top_k:][::-1]
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

    lines.append("")
    lines.append("─" * 60)
    lines.append(f"SHAP — najważniejsze cechy per model ({len(available)}/{n_models} modeli z TreeSHAP)")
    lines.append("─" * 60)
    lines.append("")

    # Globalne top cechy per model
    for name, res in available.items():
        lines.append(f"  {name}:")
        lines.append(f"  {'Cecha':<34s} {'|SHAP|':>10s}")
        lines.append(f"  {'─' * 46}")
        for feat in res["global_top_features"][:8]:
            lines.append(f"  {feat['feature']:<34s} {feat['importance']:>10.4f}")
        lines.append("")

    # Wspólne cechy między modelami
    all_features: dict[str, list[str]] = {}
    for name, res in available.items():
        for feat in res["global_top_features"]:
            all_features.setdefault(feat["feature"], []).append(name)

    shared = {f: models for f, models in all_features.items() if len(models) > 1}
    if shared:
        lines.append(f"  Wspólne cechy w top-{SHAP_TOP_FEATURES} wielu modeli:")
        for feat, models in sorted(shared.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            lines.append(f"    {feat:<34s} → {', '.join(models)}")

    unique = {f: models[0] for f, models in all_features.items() if len(models) == 1}
    if unique:
        lines.append("  Cechy unikalne dla jednego modelu:")
        for feat, model_name in sorted(unique.items(), key=lambda x: x[1])[:8]:
            lines.append(f"    {feat:<34s} ← tylko {model_name}")

    lines.append("")
    return lines


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uncertainty Quantification dla zestawu modeli ESI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--runs", nargs="+", default=None,
        help="Katalogi runów (nazwy w runs/ lub ścieżki); domyślnie najnowszy run każdego modelu",
    )
    source.add_argument(
        "--model-paths", "-m", nargs="+", type=Path, default=None,
        help="Pliki model.joblib — predykcje liczone na nowo na --split",
    )
    parser.add_argument("--feature-set", "-f", default="triage",
                        help="Zestaw cech przy wyborze najnowszych runów")
    parser.add_argument("--include-samples", action="store_true",
                        help="Uwzględnij runy trenowane na próbkach (--sample)")
    parser.add_argument("--include-untuned", action="store_true",
                        help="Uwzględnij runy bez tuningu (parametry domyślne, tag <profil>-notune)")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="Zbiór analizy; 'test' wyłącznie do raportu końcowego")
    parser.add_argument("--sample", type=int, default=None,
                        help="Stratyfikowana próbka wierszy — do szybkich testów")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Plik raportu (domyślnie results/reports/uncertainty_<split>.txt)",
    )
    parser.add_argument("--json", type=Path, default=None,
                        help="Opcjonalnie: zapisz szczegółowe wyniki jako JSON")
    parser.add_argument("--shap", action="store_true",
                        help="Dodaj analizę SHAP (wymaga model.joblib; ~1–3 min)")
    parser.add_argument("--shap-samples", type=int, default=500, help="Liczba próbek do SHAP")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Modele / runy
    models: dict[str, Any] = {}
    if args.model_paths:
        log.info(f"Ładowanie modeli ({len(args.model_paths)})…")
        models = load_models(args.model_paths)
        df = load_eval_frame(models, args.split, args.sample)
        log.info(f"Predykcje na '{args.split}' ({len(df):,} wierszy)…")
        probas = compute_predictions(models, df)
        y_true = df[TARGET_COL].to_numpy()
    else:
        from src.ensemble import latest_runs, load_run

        if args.runs:
            runs = [load_run(r) for r in args.runs]
        else:
            runs = latest_runs(args.feature_set, include_samples=args.include_samples,
                               require_tuned=not args.include_untuned)
        if not runs:
            skipped = []
            if not args.include_untuned:
                skipped.append("bez tuningu (03_train.py bez --tune) — dodaj --include-untuned")
            if not args.include_samples:
                skipped.append("na próbkach — dodaj --include-samples")
            hint = f" (pomijane runy: {'; '.join(skipped)})" if skipped else ""
            raise SystemExit(f"Brak runów w {RUNS_DIR}{hint}. Najpierw: python scripts/run_all.py "
                             "(albo python scripts/03_train.py --model all --tune)")
        log.info("Runy: " + ", ".join(r.name for r in runs))
        y_true, probas = probas_from_runs(runs, args.split, args.sample)
        if args.shap:
            paths = [r.path / "model.joblib" for r in runs if (r.path / "model.joblib").exists()]
            models = load_models(paths)
    log.info(f"Wierszy: {len(y_true):,}, modeli: {len(probas)}")

    # UQ
    log.info("Analiza Uncertainty Quantification…")
    result = compute_ensemble_uq(probas, y_true=y_true)

    # SHAP (opcjonalny)
    shap_results: dict[str, dict[str, Any] | None] = {}
    if args.shap and models:
        log.info(f"Analiza SHAP ({args.shap_samples} próbek)…")
        df_shap = load_eval_frame(models, args.split, args.shap_samples)
        for name, model in models.items():
            X = model_features(model, df_shap)
            shap_results[name] = compute_model_shap(
                model.model, name, X, list(X.columns), n_samples=args.shap_samples,
            )
            if shap_results[name] is None:
                log.info(f"  {name}: SHAP pominięty (brak TreeSHAP albo błąd)")

    # Raport
    report = ensemble_report(result)
    if shap_results:
        report += "\n" + "\n".join(shap_section(shap_results, len(models)))
    print(report)

    output = args.output or REPORTS_DIR / f"uncertainty_{args.split}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    log.info(f"Raport zapisany: {output}")

    if args.json:
        serializable = _make_serializable(result)
        if shap_results:
            serializable["shap"] = {name: res for name, res in shap_results.items() if res is not None}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        log.info(f"JSON zapisany: {args.json}")


def _make_serializable(value: Any) -> Any:
    """Konwertuje tablice i skalary numpy (także zagnieżdżone) na typy JSON."""
    if isinstance(value, dict):
        return {k: _make_serializable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
