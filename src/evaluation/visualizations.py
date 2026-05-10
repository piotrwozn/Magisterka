"""
Wizualizacje wyników modelu triażowego.

Wykresy:
    1. Confusion matrix (annotowany heatmap)
    2. ROC curves per klasa (One-vs-Rest)
    3. Calibration plot (reliability diagram)
    4. Feature importances (top-N bar)
    5. Training history (loss/error per iteration)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, roc_curve

from src.utils.config import CLASS_NAMES, FIGURES_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)


# Kolory MTS (visual cue)
MTS_COLORS = {
    "Red": "#D32F2F",
    "Orange": "#F57C00",
    "Yellow": "#FBC02D",
    "Green": "#388E3C",
    "Blue": "#1976D2",
}


def _save_or_show(
    fig,
    save_path: Path | str | None,
    show: bool = False,
    dpi: int = 150,
) -> Path | None:
    """Zapisuje wykres lub pokazuje."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        log.info(f"Zapisano wykres: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


# ─────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    normalize: str | None = "true",
    save_path: Path | str | None = None,
    title: str = "Confusion Matrix",
    show: bool = False,
):
    """
    Annotowany confusion matrix.

    Parameters
    ----------
    normalize : {'true', 'pred', 'all', None}
        'true' (domyślne) — normalizacja po wierszach (recall per klasa).
        'pred' — po kolumnach (precision per klasa).
        None — surowe liczniki.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    if normalize == "true":
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    elif normalize == "pred":
        cm_norm = cm.astype(float) / (cm.sum(axis=0, keepdims=True) + 1e-12)
    elif normalize == "all":
        cm_norm = cm.astype(float) / cm.sum()
    else:
        cm_norm = cm.astype(float)

    fig, ax = plt.subplots(figsize=(8, 6))

    fmt = ".2%" if normalize else "d"
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Proporcja" if normalize else "Liczność"},
        ax=ax,
        annot_kws={"size": 11},
    )
    ax.set_xlabel("Predykcja MTS", fontsize=12)
    ax.set_ylabel("Prawdziwa MTS", fontsize=12)

    title_full = title
    if normalize:
        title_full += f" (normalize='{normalize}')"
    ax.set_title(title_full, fontsize=13, pad=12)

    plt.tight_layout()
    return _save_or_show(fig, save_path, show)


# ─────────────────────────────────────────
# ROC curves
# ─────────────────────────────────────────
def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str] | None = None,
    save_path: Path | str | None = None,
    title: str = "Krzywe ROC (One-vs-Rest)",
    show: bool = False,
):
    """
    Krzywe ROC dla każdej klasy (OvR).
    Kluczowe dla Red i Orange — chcemy jak najwyższy AUC.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, class_name in enumerate(class_names):
        if i >= y_proba.shape[1]:
            continue

        y_binary = (np.asarray(y_true) == i).astype(int)
        if y_binary.sum() == 0:
            continue

        try:
            fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
            roc_auc = auc(fpr, tpr)

            color = MTS_COLORS.get(class_name, None)
            ax.plot(
                fpr, tpr,
                label=f"{class_name} (AUC = {roc_auc:.3f})",
                color=color,
                linewidth=2,
            )
        except Exception as e:
            log.warning(f"Nie udało się narysować ROC dla {class_name}: {e}")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Losowy")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    return _save_or_show(fig, save_path, show)


# ─────────────────────────────────────────
# Calibration plot
# ─────────────────────────────────────────
def plot_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str] | None = None,
    n_bins: int = 10,
    save_path: Path | str | None = None,
    title: str = "Reliability Diagram",
    show: bool = False,
):
    """
    Wykres kalibracji (reliability diagram) per klasa.
    Idealny model: linia y=x.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Idealna kalibracja")

    for i, class_name in enumerate(class_names):
        if i >= y_proba.shape[1]:
            continue
        y_binary = (np.asarray(y_true) == i).astype(int)
        if y_binary.sum() == 0:
            continue

        try:
            prob_true, prob_pred = calibration_curve(
                y_binary, y_proba[:, i], n_bins=n_bins, strategy="quantile"
            )
            color = MTS_COLORS.get(class_name, None)
            ax.plot(
                prob_pred, prob_true,
                marker="o", label=class_name, color=color, linewidth=2,
            )
        except Exception as e:
            log.warning(f"Nie udało się narysować kalibracji dla {class_name}: {e}")

    ax.set_xlabel("Średnie predykowane prawdopodobieństwo", fontsize=12)
    ax.set_ylabel("Frakcja pozytywnych", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return _save_or_show(fig, save_path, show)


# ─────────────────────────────────────────
# Feature importances
# ─────────────────────────────────────────
def plot_feature_importances(
    importances: pd.DataFrame,
    top_n: int = 20,
    save_path: Path | str | None = None,
    title: str = "Top N najważniejszych cech",
    show: bool = False,
):
    """
    Bar chart top-N najważniejszych cech.

    Parameters
    ----------
    importances : pd.DataFrame
        Z kolumnami 'feature' i 'importance'.
    """
    top = importances.head(top_n).iloc[::-1]  # odwróć dla wykresu poziomego

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(
        top["feature"],
        top["importance"],
        color="steelblue",
        edgecolor="navy",
        alpha=0.85,
    )
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title.replace("N", str(top_n)), fontsize=13)
    ax.grid(axis="x", alpha=0.3)

    # Anotacja wartości
    for i, (idx, row) in enumerate(top.iterrows()):
        ax.text(
            row["importance"], i,
            f"  {row['importance']:.3f}",
            va="center", fontsize=9,
        )

    plt.tight_layout()
    return _save_or_show(fig, save_path, show)


# ─────────────────────────────────────────
# Training history
# ─────────────────────────────────────────
def plot_training_history(
    history: list[dict] | pd.DataFrame,
    metrics: list[str] | None = None,
    save_path: Path | str | None = None,
    title: str = "Historia treningu",
    show: bool = False,
):
    """
    Loss/error per iteration dla XGBoost/LightGBM.

    Parameters
    ----------
    history : list[dict] | pd.DataFrame
        Wynik z `tracker.data['training_history']` lub odpowiadający DataFrame.
    metrics : list[str], optional
        Lista metryk do narysowania (None → wszystkie poza 'iteration').
    """
    if isinstance(history, list):
        if not history:
            log.warning("Pusta historia treningu — pomijam plot.")
            return None
        history_df = pd.DataFrame(history)
    else:
        history_df = history.copy()

    if metrics is None:
        metrics = [c for c in history_df.columns if c != "iteration"]

    if not metrics:
        log.warning("Brak metryk do wyświetlenia.")
        return None

    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        nrows=n_metrics, ncols=1,
        figsize=(10, 3 * n_metrics),
        sharex=True,
    )
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric in history_df.columns:
            color = "tab:blue" if "train" in metric else "tab:orange"
            ax.plot(
                history_df["iteration"],
                history_df[metric],
                label=metric,
                color=color,
                linewidth=1.5,
            )
            ax.set_ylabel(metric, fontsize=11)
            ax.legend(loc="best", fontsize=9)
            ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Iteracja", fontsize=12)
    axes[0].set_title(title, fontsize=13)
    plt.tight_layout()
    return _save_or_show(fig, save_path, show)
