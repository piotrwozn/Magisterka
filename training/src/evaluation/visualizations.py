"""
Wizualizacje wyników modelu triażowego (5 klas: ESI 1–5).

Wykresy:
    1. Confusion matrix (annotowany heatmap)
    2. ROC curves per poziom ESI (One-vs-Rest)
    3. Calibration plot (reliability diagram)
    4. Feature importances (top-N bar)
    5. Training history (loss/error per iteration)
    6. Reliability diagram per poziom ESI + top-label, z liczebnością binów (plot_reliability)
    7. Wykres leśny metryk z przedziałami ufności (plot_metric_ci)
    8. Kompromis undertriage–QWK reguł decyzyjnych (plot_tradeoff; dane: decision.tradeoff_curve)

Kolory poziomów ESI = kolory TOPSOR (ESI 1 czerwony, …, ESI 5 niebieski),
przypisane po indeksie klasy 0–4 — nie po nazwie.

Backend: przy imporcie poza notebookiem ustawiany jest nieinteraktywny `Agg`
(zapis do plików na serwerze/CI bez okien). Nadpisanie: zmienna MPLBACKEND.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import auc, confusion_matrix, roc_curve  # noqa: E402

from src.utils.config import CLASS_NAMES, TOPSOR_HEX  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


# ─────────────────────────────────────────
# Kolory poziomów ESI (TOPSOR)
# ─────────────────────────────────────────
# Kolor klasy o indeksie i (0 = ESI 1, …, 4 = ESI 5) — odcienie z config.TOPSOR_HEX
ESI_COLORS: list[str] = [TOPSOR_HEX[name] for name in CLASS_NAMES]


def _class_color(index: int) -> str | None:
    """Kolor TOPSOR dla indeksu klasy (None poza zakresem → kolor domyślny)."""
    return ESI_COLORS[index] if 0 <= index < len(ESI_COLORS) else None


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
    title: str = "Macierz pomyłek",
    show: bool = False,
):
    """
    Annotowany confusion matrix (wiersze = prawdziwy poziom ESI, kolumny = predykcja).

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
        cm_norm = cm.astype(float) / max(cm.sum(), 1)
    else:
        cm_norm = cm

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
    ax.set_xlabel("Predykcja (poziom ESI)", fontsize=12)
    ax.set_ylabel("Prawdziwy poziom ESI", fontsize=12)

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
    Krzywe ROC dla każdego poziomu ESI (OvR).
    Kluczowe dla ESI 1 i ESI 2 — chcemy jak najwyższy AUC.
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
            ax.plot(
                fpr, tpr,
                label=f"{class_name} (AUC = {roc_auc:.3f})",
                color=_class_color(i),
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
    Wykres kalibracji (reliability diagram) per poziom ESI.
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
            ax.plot(
                prob_pred, prob_true,
                marker="o", label=class_name, color=_class_color(i), linewidth=2,
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
    title : str
        Fraza "Top N" zostanie zastąpiona faktyczną liczbą cech.
    """
    top = importances.head(top_n).iloc[::-1]  # odwróć dla wykresu poziomego

    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.3)))
    ax.barh(
        top["feature"],
        top["importance"],
        color="steelblue",
        edgecolor="navy",
        alpha=0.85,
    )
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title.replace("Top N", f"Top {len(top)}"), fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    ax.margins(x=0.1)  # miejsce na etykiety wartości

    # Anotacja wartości
    for i, value in enumerate(top["importance"]):
        ax.text(value, i, f"  {value:.3f}", va="center", fontsize=9)

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


# ─────────────────────────────────────────
# Wykres niezawodności (reliability diagram) — panel na każdy poziom ESI + top-label
# ─────────────────────────────────────────
# Kolor panelu „top-label” (nie jest poziomem ESI — neutralny grafit)
TOP_LABEL_COLOR = "#37474F"
# Kolejne zestawy prawdopodobieństw na jednym wykresie (np. przed / po temperaturze)
_SET_STYLES: list[tuple[str, str]] = [("-", "o"), ("--", "s"), (":", "^")]
_INK = "#263238"          # tekst i obwódki znaczników (żółty TOPSOR ginie na białym tle)
_GRID = "#B0BEC5"


def _as_proba_sets(proba: np.ndarray | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if isinstance(proba, dict):
        if not proba:
            raise ValueError("Pusty słownik prawdopodobieństw")
        if len(proba) > len(_SET_STYLES):
            raise ValueError(f"Najwyżej {len(_SET_STYLES)} zestawy prawdopodobieństw na jednym wykresie")
        return {str(k): np.asarray(v, dtype=float) for k, v in proba.items()}
    return {"": np.asarray(proba, dtype=float)}


def plot_reliability(
    y_true: np.ndarray,
    proba: np.ndarray | dict[str, np.ndarray],
    class_names: list[str] | None = None,
    path: Path | str | None = None,
    n_bins: int = 15,
    *,
    title: str | None = None,
    strategy: str = "uniform",
    min_count: int = 10,
    show: bool = False,
):
    """
    Wykres niezawodności: panel „top-label” (pewność najbardziej prawdopodobnego poziomu vs
    trafność) i po jednym panelu jeden-przeciw-reszcie dla każdego poziomu ESI (kolory TOPSOR).

    Pod każdym panelem — liczebność binów (skala log) dla pierwszego zestawu. Idealna
    kalibracja = przekątna; punkty PONIŻEJ przekątnej = model zbyt pewny siebie.

    Parameters
    ----------
    proba : (n, 5) albo {etykieta: (n, 5)} — do 3 zestawów na tych samych panelach
            (np. {"przed T": p, "po T": apply_temperature(p, T)}); ECE każdego w legendzie.
    path  : plik PNG (None → bez zapisu).
    min_count : biny z mniejszą liczbą wierszy nie trafiają na krzywą (szum pojedynczych
            pacjentów); ECE i histogram liczone ze wszystkich.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    from src.evaluation.statistics import calibration_error, reliability_table

    if class_names is None:
        class_names = CLASS_NAMES
    sets = _as_proba_sets(proba)
    y = np.asarray(y_true)
    n_classes = next(iter(sets.values())).shape[1]
    panels: list[tuple[str, int | None, str]] = [("Najbardziej prawdopodobny poziom (top-label)", None,
                                                    TOP_LABEL_COLOR)]
    panels += [(class_names[k] if k < len(class_names) else f"klasa {k}", k,
                _class_color(k) or TOP_LABEL_COLOR) for k in range(n_classes)]

    n_cols = 3
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig = plt.figure(figsize=(5.0 * n_cols, 5.6 * n_rows))
    outer = GridSpec(n_rows, n_cols, figure=fig, hspace=0.32, wspace=0.28)

    for i, (label, class_idx, color) in enumerate(panels):
        r, c = divmod(i, n_cols)
        cell = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[r, c], height_ratios=[4, 1], hspace=0.08)
        ax = fig.add_subplot(cell[0])
        ax_hist = fig.add_subplot(cell[1], sharex=ax)
        ax.plot([0, 1], [0, 1], linestyle="--", color=_GRID, linewidth=1.2, label="idealna kalibracja")
        for j, (set_label, p) in enumerate(sets.items()):
            table = reliability_table(y, p, class_idx=class_idx, n_bins=n_bins, strategy=strategy)
            if class_idx is None:
                hits = (p.argmax(axis=1) == y).astype(float)
                ece = calibration_error(p.max(axis=1), hits, n_bins, strategy)[0]
            else:
                ece = calibration_error(p[:, class_idx], (y == class_idx).astype(float), n_bins, strategy)[0]
            linestyle, marker = _SET_STYLES[j]
            name = f"{set_label}: ECE = {ece:.3f}" if set_label else f"ECE = {ece:.3f}"
            curve = table[table["count"] >= min_count]
            ax.plot(curve["mean_pred"], curve["observed_freq"], linestyle=linestyle, marker=marker,
                    color=color, markeredgecolor=_INK, markeredgewidth=0.8, markersize=6, linewidth=2,
                    label=name)
            if j == 0:
                widths = (table["bin_high"] - table["bin_low"]).to_numpy()
                ax_hist.bar(table["bin_low"], table["count"], width=widths * 0.92, align="edge",
                            color=color, edgecolor=_INK, linewidth=0.4, alpha=0.85)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(label, fontsize=11, color=_INK)
        ax.set_ylabel("odsetek trafień" if class_idx is None else f"odsetek {label}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8, framealpha=0.9, edgecolor="none")
        plt.setp(ax.get_xticklabels(), visible=False)
        ax_hist.set_yscale("log")
        ax_hist.set_ylabel("n", fontsize=9)
        ax_hist.set_xlabel("pewność (max p)" if class_idx is None else "przewidywane p", fontsize=10)
        ax_hist.grid(alpha=0.3, axis="y")

    fig.suptitle(title or "Kalibracja (wykres niezawodności)", fontsize=13, color=_INK, y=0.94)
    return _save_or_show(fig, path, show)


# ─────────────────────────────────────────
# Wykres leśny (forest plot) metryk z przedziałami ufności
# ─────────────────────────────────────────
_GROUP_STYLES: list[tuple[str, str]] = [("#37474F", "o"), ("#00897B", "s"), ("#8E24AA", "D")]


def plot_metric_ci(
    df: pd.DataFrame,
    path: Path | str | None = None,
    *,
    title: str | None = None,
    highlight: list[str] | None = None,
    metric_labels: dict[str, str] | None = None,
    show: bool = False,
):
    """
    Wykres leśny: wartość i przedział ufności każdego modelu / połączenia.

    df : kolumny `label`, `value`, `ci_low`, `ci_high`; opcjonalnie `metric` (osobny panel na
         każdą metrykę) i `group` (np. 'całość' / 'vc' — przesunięte znaczniki w jednym wierszu,
         różny kształt i kolor, z legendą). Kolejność wierszy = kolejność etykiet w df (pierwsza u góry).
    highlight : etykiety wyróżnione pogrubieniem (np. wybrane połączenie).
    metric_labels : metryka → podpis osi X.
    """
    required = {"label", "value", "ci_low", "ci_high"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Brak kolumn {sorted(missing)} w tabeli do wykresu leśnego")
    data = df.copy()
    if "metric" not in data.columns:
        data["metric"] = ""
    if "group" not in data.columns:
        data["group"] = ""
    labels = list(dict.fromkeys(data["label"]))
    metrics = list(dict.fromkeys(data["metric"]))
    groups = list(dict.fromkeys(data["group"]))
    if len(groups) > len(_GROUP_STYLES):
        raise ValueError(f"Najwyżej {len(_GROUP_STYLES)} grupy na wykresie leśnym")
    highlight = set(highlight or [])
    positions = {label: len(labels) - 1 - i for i, label in enumerate(labels)}
    # Pierwsza grupa u góry wiersza
    offsets = np.linspace(0.18, -0.18, len(groups)) if len(groups) > 1 else np.array([0.0])

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics) + 1.5, 0.45 * len(labels) + 1.6),
                             sharey=True, squeeze=False)
    for ax, metric in zip(axes[0], metrics, strict=True):
        part = data[data["metric"] == metric]
        for g, group in enumerate(groups):
            rows = part[part["group"] == group]
            if rows.empty:
                continue
            color, marker = _GROUP_STYLES[g]
            ypos = np.array([positions[label] for label in rows["label"]]) + offsets[g]
            values = rows["value"].to_numpy(dtype=float)
            low = rows["ci_low"].to_numpy(dtype=float)
            high = rows["ci_high"].to_numpy(dtype=float)
            # Przedział rysowany wprost od ci_low do ci_high: wartość punktowa metryki obciążonej
            # (np. ECE) może leżeć poza przedziałem percentylowym — errorbar by tego nie przyjął
            ok = np.isfinite(low) & np.isfinite(high)
            ax.hlines(ypos[ok], low[ok], high[ok], color=color, linewidth=1.6)
            for edge in (low, high):
                ax.plot(edge[ok], ypos[ok], linestyle="none", marker="|", markersize=8, color=color)
            ax.plot(values, ypos, linestyle="none", marker=marker, markersize=6, color=color,
                    label=str(group) if group else None)
        ax.set_xlabel((metric_labels or {}).get(metric, metric), fontsize=10)
        ax.grid(alpha=0.3, axis="x")
    axes[0][0].set_yticks([positions[label] for label in labels])
    axes[0][0].set_yticklabels(labels)
    for tick in axes[0][0].get_yticklabels():
        if tick.get_text() in highlight:
            tick.set_fontweight("bold")
    axes[0][0].set_ylim(-0.7, len(labels) - 0.3)
    if len(groups) > 1:
        handles, names = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, names, loc="upper right", fontsize=9, frameon=False, ncol=len(groups))
    fig.suptitle(title or "Metryki z 95% przedziałami ufności", fontsize=12, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_or_show(fig, path, show)


# ─────────────────────────────────────────
# Kompromis undertriage–QWK reguł decyzyjnych (C6)
# ─────────────────────────────────────────
# metoda → (podpis, kolor, styl linii, znacznik)
_TRADEOFF_STYLES: dict[str, tuple[str, str, str, str]] = {
    "cost": ("reguła kosztowa, mnożnik k", "#546E7A", "-", "o"),
    "target": ("B1: EL + progi (cel undertriage)", "#00897B", "-", "D"),
    "target_safety": ("B1 + siatka bezpieczeństwa (B2)", "#004D40", "--", "d"),
    "bayes_target": ("B4: decyzja Bayesa, λ do celu", "#8E24AA", "-", "s"),
}


def plot_tradeoff(
    df: pd.DataFrame,
    path: Path | str | None = None,
    *,
    chosen: dict | None = None,
    title: str | None = None,
    show: bool = False,
):
    """
    Wykres kompromisu: undertriage ESI1–2 → ESI3–5 [%] (oś X) vs QWK (oś Y).

    df : wynik `src.evaluation.decision.tradeoff_curve` (kolumny method, target, param,
         undertriage, qwk). Linie: reguła kosztowa (punkty podpisane k), reguły z celem T;
         argmax jako pojedynczy punkt.
    chosen : wybrany punkt pracy, np. {"method": "target_safety", "target": 0.15} — wyróżniony
             gwiazdką (brak takiego wiersza → ostrzeżenie, wykres bez wyróżnienia).
    """
    missing = {"method", "undertriage", "qwk"} - set(df.columns)
    if missing:
        raise ValueError(f"Brak kolumn {sorted(missing)} w tabeli kompromisu")
    data = df.copy()
    for col in ("target", "param"):
        if col not in data.columns:
            data[col] = np.nan
    data = data.dropna(subset=["undertriage", "qwk"])

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for method, (label, color, linestyle, marker) in _TRADEOFF_STYLES.items():
        part = data[data["method"] == method].sort_values("undertriage")
        if part.empty:
            continue
        # Siatka bezpieczeństwa zmienia mało decyzji — krzywa B1+B2 leży prawie na B1: puste znaczniki
        ax.plot(part["undertriage"] * 100, part["qwk"], linestyle=linestyle, marker=marker, color=color,
                markeredgecolor=_INK if method != "target_safety" else color, markeredgewidth=0.8,
                markerfacecolor="white" if method == "target_safety" else color,
                markersize=5, linewidth=1.8 if method != "target_safety" else 1.2, label=label)
        if method == "cost":
            for _, row in part.iterrows():
                ax.annotate(f"k={row['param']:g}", (row["undertriage"] * 100, row["qwk"]),
                            textcoords="offset points", xytext=(4, -11), fontsize=7.5, color=color)
    argmax = data[data["method"] == "argmax"]
    if not argmax.empty:
        row = argmax.iloc[0]
        ax.plot(row["undertriage"] * 100, row["qwk"], linestyle="none", marker="o", markersize=9,
                markerfacecolor="white", markeredgecolor=_INK, markeredgewidth=1.5, zorder=5, label="argmax")

    if chosen:
        sel = data["method"] == chosen.get("method")
        if chosen.get("target") is not None:
            sel &= np.isclose(data["target"].astype(float), float(chosen["target"]))
        if sel.any():
            row = data[sel].iloc[0]
            ax.plot(row["undertriage"] * 100, row["qwk"], linestyle="none", marker="*", markersize=17,
                    color="#D32F2F", markeredgecolor=_INK, markeredgewidth=0.8, zorder=6,
                    label="wybrany punkt pracy")
            ax.annotate(f"undertriage {row['undertriage']:.1%}, QWK {row['qwk']:.3f}",
                        (row["undertriage"] * 100, row["qwk"]), textcoords="offset points",
                        xytext=(10, -16), fontsize=8.5, color=_INK)
        else:
            log.warning(f"plot_tradeoff: brak wiersza dla wybranego punktu pracy {chosen}")

    ax.set_xlabel("undertriage ESI1–2 → ESI3–5 [%] (mniej = bezpieczniej)", fontsize=10)
    ax.set_ylabel("QWK", fontsize=10)
    ax.grid(alpha=0.35, color=_GRID, linewidth=0.6)
    ax.set_xlim(left=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.set_title(title or "Kompromis undertriage – QWK reguł decyzyjnych", fontsize=12, color=_INK)
    fig.tight_layout()
    return _save_or_show(fig, path, show)
