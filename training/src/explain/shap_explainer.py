"""
SHAP dla modeli triażowych ESI (5 klas: ESI 1–5, indeks 0–4).

Metoda dobierana automatycznie do typu modelu:
    xgboost / lightgbm / random_forest → shap.TreeExplainer (dokładny TreeSHAP)
    catboost → natywne wartości SHAP CatBoost (`get_feature_importance(type="ShapValues")`);
               shap.TreeExplainer dla CatBoost kończy proces błędem segmentacji
               (shap 0.54 + catboost 1.2), więc go omijamy
    logreg   → shap.LinearExplainer na danych po imputacji i standaryzacji
               (pokazywane są ORYGINALNE wartości cech pacjenta)
    ebm      → własne wkłady termów EBM (`eval_terms`) — EBM jest addytywny, więc to
               dokładne wyjaśnienie lokalne; interakcje "a & b" są osobnymi pozycjami
    inny model z predict_proba → shap.Explainer (permutacyjny) z małą próbką tła

Jednostki: logity / log-odds (boosting, logreg, EBM) albo prawdopodobieństwo
(random forest, metoda permutacyjna) — patrz `explainer.output_space`.

Model nieobsługiwany → wyjątek `ExplainerUnavailable` z czytelnym komunikatem
(skrypty i aplikacja pokazują go zamiast przerywać działanie).

Funkcjonalności:
    - Globalne wyjaśnienia (summary plot: słupki dla wszystkich klas lub beeswarm jednej klasy)
    - Lokalne wyjaśnienia per pacjent (waterfall)
    - Strukturalny dict z top cechami za / przeciw wybranej klasie
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import (
    CLASS_NAMES,
    N_CLASSES,
    RANDOM_SEED,
    SHAP_BACKGROUND_SIZE,
    SHAP_TOP_FEATURES,
    TOPSOR_HEX,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# Metody wyjaśniania i przestrzeń, w której wyrażone są wartości SHAP
METHOD_OUTPUT_SPACE: dict[str, str] = {
    "tree": "log-odds",                # xgboost / lightgbm (random forest → probability)
    "catboost": "log-odds",
    "linear": "log-odds",
    "ebm": "log-odds",
    "permutation": "probability",
}
PERMUTATION_MAX_BACKGROUND: int = 100   # tło metody permutacyjnej (koszt ∝ tło × cechy)
SKLEARN_TREE_CLASSES = ("RandomForestClassifier", "ExtraTreesClassifier", "DecisionTreeClassifier")



class ExplainerUnavailable(RuntimeError):
    """Nie da się wyjaśnić modelu metodą SHAP (komunikat zamiast awarii)."""


# ─────────────────────────────────────────
# Rozpoznanie typu modelu
# ─────────────────────────────────────────
def unwrap_model(model: Any) -> Any:
    """BaseTriageModel → surowy estymator (XGBClassifier, Pipeline, EBM …)."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(model, "feature_names") and hasattr(model, "is_fitted"):
        return inner
    return model


def detect_method(model: Any) -> str | None:
    """
    Metoda SHAP dla modelu: 'tree' | 'catboost' | 'linear' | 'ebm' | 'permutation'
    albo None, gdy model nie ma nawet `predict_proba`.
    """
    raw = unwrap_model(model)
    cls = type(raw).__name__
    if cls == "CatBoostClassifier":
        return "catboost"
    if cls in ("XGBClassifier", "LGBMClassifier", *SKLEARN_TREE_CLASSES):
        return "tree"
    if cls == "ExplainableBoostingClassifier":
        return "ebm"
    final = raw.steps[-1][1] if hasattr(raw, "steps") else raw
    if hasattr(final, "coef_") and hasattr(final, "intercept_"):
        return "linear"
    if hasattr(raw, "predict_proba") or hasattr(model, "predict_proba"):
        return "permutation"
    return None


def load_background(
    feature_names: list[str],
    n: int = SHAP_BACKGROUND_SIZE,
    split: str = "train",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Losowa próbka tła z przygotowanego splitu (tylko kolumny modelu, NaN zachowane)."""
    from src.data.dataset import load_split

    df = load_split(split, columns=list(feature_names))
    return df.sample(n=min(n, len(df)), random_state=seed).astype(np.float32)


# ─────────────────────────────────────────
# Explainer
# ─────────────────────────────────────────
class SHAPTriageExplainer:
    """
    Wrapper SHAP dla modeli triażowych.

    Użycie:
        explainer = SHAPTriageExplainer(model)        # BaseTriageModel lub surowy estymator
        explainer.fit()

        # Globalne (lista: jedna macierz (n, cechy) na klasę ESI)
        shap_values = explainer.explain_dataset(X_val)

        # Per pacjent
        single = explainer.explain_patient(X_val.iloc[[0]])
        # → dict: predicted_class, probabilities, top_features_for, top_features_against
    """

    def __init__(
        self,
        model: Any,
        background_data: pd.DataFrame | None = None,
        class_names: list[str] | None = None,
        max_background: int = SHAP_BACKGROUND_SIZE,
        method: str = "auto",
    ):
        """
        Parameters
        ----------
        model : BaseTriageModel | estymator z predict_proba
        background_data : pd.DataFrame, optional
            Próbka tła (np. `load_background`). Potrzebna dla metody permutacyjnej,
            zalecana dla regresji logistycznej; TreeSHAP / CatBoost / EBM jej nie używają.
        class_names : list[str], optional — domyślnie config.CLASS_NAMES
        max_background : int
        method : 'auto' (rozpoznanie typu modelu) lub wymuszona metoda, np. 'permutation'
        """
        self.model = model
        self.raw_model = unwrap_model(model)
        self.class_names = list(class_names or CLASS_NAMES)
        self.max_background = max_background
        self.model_name = getattr(model, "name", None) or type(self.raw_model).__name__

        self.method = detect_method(model) if method == "auto" else method
        if self.method is None:
            raise ExplainerUnavailable(
                f"Model '{self.model_name}' nie ma predict_proba — SHAP niedostępny."
            )
        self.output_space = METHOD_OUTPUT_SPACE.get(self.method, "log-odds")
        if self.method == "tree" and type(self.raw_model).__name__ in SKLEARN_TREE_CLASSES:
            self.output_space = "probability"     # lasy sklearn: SHAP w przestrzeni p

        names = getattr(model, "feature_names", None)
        if not names and background_data is not None:
            names = list(background_data.columns)
        self.feature_names: list[str] | None = list(names) if names else None

        if background_data is not None:
            limit = PERMUTATION_MAX_BACKGROUND if self.method == "permutation" else max_background
            if len(background_data) > limit:
                background_data = background_data.sample(n=limit, random_state=RANDOM_SEED)
            background_data = self._prepare(background_data)
        self.background_data = background_data

        self.explainer: Any = None
        self._preprocess: Any = None
        self.expected_value: np.ndarray | None = None
        self.is_fitted = False

    # ──────── Pomocnicze ────────
    @staticmethod
    def _unwrap_model(model: Any) -> Any:
        """Zachowane dla kompatybilności — patrz `unwrap_model`."""
        return unwrap_model(model)

    @staticmethod
    def method_for(model: Any) -> str | None:
        """Metoda, której użyje explainer (np. by zdecydować, czy wczytać tło)."""
        return detect_method(model)

    @property
    def needs_background(self) -> bool:
        return self.method in ("linear", "permutation")

    def _prepare(self, X: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """1 wiersz / ramka → DataFrame w kolejności cech modelu (brakujące → NaN)."""
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        if self.feature_names is not None:
            X = X.reindex(columns=self.feature_names)
        return X.astype(np.float32)

    @property
    def output_feature_names(self) -> list[str]:
        """Nazwy pozycji wyjaśnienia (dla EBM: termy, w tym interakcje)."""
        if self.method == "ebm":
            return [str(t) for t in self.raw_model.term_names_]
        if self.feature_names is None:
            raise ValueError("Nieznane nazwy cech — przekaż model z feature_names lub background_data.")
        return list(self.feature_names)

    def feature_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Wartości odpowiadające pozycjom wyjaśnienia (do wykresów); interakcje EBM → NaN."""
        X = self._prepare(X)
        if self.method != "ebm":
            return X
        cols = {}
        for name, feats in zip(self.output_feature_names, self.raw_model.term_features_):
            cols[name] = X.iloc[:, feats[0]].to_numpy() if len(feats) == 1 else np.full(len(X), np.nan)
        return pd.DataFrame(cols, index=X.index)

    # ──────── Fit ────────
    def fit(self) -> "SHAPTriageExplainer":
        """Przygotowuje explainer właściwy dla typu modelu."""
        log.info(f"SHAP: model '{self.model_name}', metoda '{self.method}' ({self.output_space})")

        if self.method == "tree":
            shap = _import_shap()
            target = self.raw_model
            best = getattr(target, "best_iteration", None) if type(target).__name__ == "XGBClassifier" else None
            if best is not None:
                # XGBoost po early stopping trzyma też drzewa za best_iteration, a predict_proba
                # ich nie używa — jawne przycięcie gwarantuje (niezależnie od wersji shap),
                # że wyjaśniamy dokładnie ten model, który daje predykcje.
                target = target.get_booster()[: int(best) + 1]
            try:
                self.explainer = shap.TreeExplainer(target)
            except Exception as e:
                raise ExplainerUnavailable(f"TreeExplainer nie obsługuje modelu '{self.model_name}': {e}") from e
            self.expected_value = np.asarray(self.explainer.expected_value, dtype=float).reshape(-1)

        elif self.method == "catboost":
            self.expected_value = None     # bias CatBoost odczytywany przy pierwszym wyjaśnieniu

        elif self.method == "linear":
            shap = _import_shap()
            raw = self.raw_model
            if hasattr(raw, "steps"):
                from sklearn.pipeline import Pipeline

                self._preprocess = Pipeline(raw.steps[:-1]) if len(raw.steps) > 1 else None
                clf = raw.steps[-1][1]
            else:
                clf = raw
            n_features = clf.coef_.shape[1]
            if self.background_data is not None:
                bg = self._transform(self.background_data)
            else:
                # Po standaryzacji średnia zbioru treningowego = 0 → tło zerowe
                log.warning("SHAP (linear): brak tła — przyjmuję średnią treningową (zera po standaryzacji)")
                bg = np.zeros((1, n_features))
            masker = shap.maskers.Independent(bg, max_samples=len(bg))
            self.explainer = shap.LinearExplainer(clf, masker)
            self.expected_value = np.asarray(self.explainer.expected_value, dtype=float).reshape(-1)

        elif self.method == "ebm":
            if not hasattr(self.raw_model, "eval_terms"):
                raise ExplainerUnavailable(
                    "Ta wersja interpret nie ma EBM.eval_terms — użyj model.explain_local()."
                )
            self.expected_value = np.asarray(self.raw_model.intercept_, dtype=float).reshape(-1)

        elif self.method == "permutation":
            if self.background_data is None or self.feature_names is None:
                raise ExplainerUnavailable(
                    f"Model '{self.model_name}' wymaga metody permutacyjnej, a ta potrzebuje "
                    "próbki tła (background_data, np. load_background(model.feature_names))."
                )
            shap = _import_shap()
            names = self.feature_names

            def predict_fn(values: np.ndarray) -> np.ndarray:
                return self.model.predict_proba(pd.DataFrame(values, columns=names).astype(np.float32))

            masker = shap.maskers.Independent(self.background_data.to_numpy(), max_samples=PERMUTATION_MAX_BACKGROUND)
            self.explainer = shap.explainers.Permutation(predict_fn, masker, output_names=self.class_names)
            self.expected_value = None

        else:
            raise ExplainerUnavailable(f"Nieznana metoda SHAP: '{self.method}'")

        self.is_fitted = True
        return self

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """Dane → przestrzeń wejścia klasyfikatora liniowego (po imputacji i skalowaniu)."""
        if self._preprocess is None:
            return np.asarray(X, dtype=float)
        return np.asarray(self._preprocess.transform(X), dtype=float)

    # ──────── Obliczenia ────────
    def explain_values(self, X: pd.DataFrame | pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """
        Wartości SHAP w jednolitym formacie.

        Returns
        -------
        values : np.ndarray (n_próbek, n_klas, n_pozycji)
        base   : np.ndarray (n_klas,) — wartość oczekiwana (bazowa) per klasa
        """
        if not self.is_fitted:
            self.fit()
        X = self._prepare(X)
        n, n_out = len(X), len(self.output_feature_names)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.method == "catboost":
                from catboost import Pool

                # (n, klasy, cechy + 1); ostatnia kolumna = wartość bazowa (bias)
                arr = np.asarray(self.raw_model.get_feature_importance(Pool(X), type="ShapValues"), dtype=float)
                if arr.ndim == 2:          # model jednowyjściowy
                    arr = arr[:, None, :]
                self.expected_value = arr[0, :, -1]
                return arr[:, :, :-1], self.expected_value
            if self.method == "ebm":
                # (n, termy, klasy) → (n, klasy, termy)
                arr = np.asarray(self.raw_model.eval_terms(X), dtype=float)
                arr = arr[:, :, None] if arr.ndim == 2 else arr
                return arr.transpose(0, 2, 1), self.expected_value

            if self.method in ("tree", "linear"):
                data = X if self.method == "tree" else self._transform(X)
                if self.method == "tree":
                    raw_values = self.explainer.shap_values(data, check_additivity=False)
                else:
                    raw_values = self.explainer.shap_values(data)
                # shap aktualizuje expected_value przy liczeniu (dla XGBoost wartość z fit()
                # bywa nieaktualna) — odczytujemy ją ponownie, by zachować addytywność
                base = np.asarray(self.explainer.expected_value, dtype=float).reshape(-1)
                self.expected_value = base
            else:  # permutation
                explanation = self.explainer(X.to_numpy(), max_evals=2 * X.shape[1] + 1, silent=True)
                raw_values = explanation.values
                base = np.asarray(explanation.base_values, dtype=float).reshape(n, -1)[0]
                self.expected_value = base

        if self.method == "linear":
            # Wejścia klasyfikatora = cechy + `missingindicator_<cecha>` (SimpleImputer(add_indicator=True)):
            # wkład wskaźnika braku dopisujemy do jego cechy, żeby wyjaśnienie dotyczyło cech modelu
            n_in = np.asarray(data).shape[1]
            values = _to_sample_class_feature(raw_values, n, len(self.class_names), n_in)
            values = self._fold_indicator_values(values)
        else:
            values = _to_sample_class_feature(raw_values, n, len(self.class_names), n_out)
        return values, np.asarray(base, dtype=float).reshape(-1)

    def _fold_indicator_values(self, values: np.ndarray) -> np.ndarray:
        """(n, C, wejścia klasyfikatora) → (n, C, cechy): wskaźniki braków sumowane z cechą bazową."""
        names = self.output_feature_names
        if values.shape[2] == len(names):
            return values
        try:
            inputs = [str(c) for c in self._preprocess.get_feature_names_out(names)]
        except (AttributeError, ValueError) as e:
            raise ExplainerUnavailable(f"SHAP (linear): nie da się odczytać nazw wejść klasyfikatora: {e}") from e
        if len(inputs) != values.shape[2]:
            raise ExplainerUnavailable(
                f"SHAP (linear): {values.shape[2]} wejść klasyfikatora, a {len(inputs)} nazw z preprocessingu"
            )
        index = {name: i for i, name in enumerate(names)}
        prefix = "missingindicator_"
        folded = np.zeros(values.shape[:2] + (len(names),), dtype=float)
        for j, name in enumerate(inputs):
            base = name[len(prefix):] if name.startswith(prefix) else name
            if base not in index:
                raise ExplainerUnavailable(f"SHAP (linear): nieznane wejście klasyfikatora '{name}'")
            folded[:, :, index[base]] += values[:, :, j]
        return folded

    # ──────── Globalne wyjaśnienia ────────
    def explain_dataset(self, X: pd.DataFrame) -> list[np.ndarray]:
        """
        SHAP dla całego zbioru.

        Returns
        -------
        list[np.ndarray] — jedna macierz (n_próbek, n_pozycji) na klasę ESI 1–5
        (format przyjmowany przez `shap_summary_plot`).
        """
        values, _ = self.explain_values(X)
        return [values[:, c, :] for c in range(values.shape[1])]

    # ──────── Per pacjent ────────
    def explain_patient(
        self,
        patient: pd.DataFrame | pd.Series,
        top_n: int = SHAP_TOP_FEATURES,
        class_idx: int | None = None,
    ) -> dict[str, Any]:
        """
        Strukturalne wyjaśnienie dla pojedynczego pacjenta.

        Parameters
        ----------
        patient : pd.DataFrame (1 wiersz) | pd.Series
        top_n : int — ile najważniejszych pozycji zwrócić
        class_idx : int, optional — klasa do wyjaśnienia (domyślnie argmax predykcji;
            np. decyzja kosztowa `decide(proba, "cost")`)

        Returns
        -------
        dict z kluczami:
            - predicted_class / predicted_class_idx / predicted_esi (argmax)
            - explained_class / explained_class_idx (klasa, której dotyczą cechy)
            - probabilities         {nazwa_klasy: p}
            - top_features_for      cechy podnoszące wynik wyjaśnianej klasy
            - top_features_against  cechy obniżające go
            - shap_values_per_class {nazwa_klasy: list[float]}
            - base_value, method, output_space, feature_names, patient_values
        """
        X = self._prepare(patient).iloc[[0]]
        proba = np.asarray(self.model.predict_proba(X), dtype=float)[0]
        predicted_idx = int(np.argmax(proba))
        explained_idx = predicted_idx if class_idx is None else int(class_idx)

        values, base = self.explain_values(X)
        sv = values[0]                                     # (n_klas, n_pozycji)
        names = self.output_feature_names
        shown = self._display_values(X)

        ranked = sorted(zip(names, sv[explained_idx], shown), key=lambda t: abs(t[1]), reverse=True)
        top = ranked[:top_n]

        def to_item(feature: str, value: float, patient_value: float | str) -> dict[str, Any]:
            return {"feature": str(feature), "shap_value": float(value), "patient_value": patient_value}

        return {
            "predicted_class": self.class_names[predicted_idx],
            "predicted_class_idx": predicted_idx,
            "predicted_esi": predicted_idx + 1,
            "explained_class": self.class_names[explained_idx],
            "explained_class_idx": explained_idx,
            "probabilities": {name: float(p) for name, p in zip(self.class_names, proba)},
            "top_features_for": [to_item(*t) for t in top if t[1] > 0],
            "top_features_against": [to_item(*t) for t in top if t[1] < 0],
            "shap_values_per_class": {
                name: sv[i].tolist() for i, name in enumerate(self.class_names) if i < len(sv)
            },
            "base_value": float(base[explained_idx]) if explained_idx < len(base) else None,
            "method": self.method,
            "output_space": self.output_space,
            "feature_names": list(names),
            "patient_values": list(shown),
        }

    def _display_values(self, X: pd.DataFrame) -> list[float | str]:
        """Wartości cech pacjenta odpowiadające pozycjom wyjaśnienia."""
        row = X.iloc[0]
        if self.method != "ebm":
            return [_safe_float(v) for v in row.to_numpy()]
        cols = list(X.columns)
        out: list[float | str] = []
        for feats in self.raw_model.term_features_:
            if len(feats) == 1:
                out.append(_safe_float(row.iloc[feats[0]]))
            else:
                out.append(", ".join(f"{cols[i]}={_safe_float(row.iloc[i])}" for i in feats))
        return out

    @staticmethod
    def _normalize_shap_values(shap_values: Any, n_features: int | None = None) -> np.ndarray:
        """
        Zachowane dla kompatybilności: wartości SHAP pierwszej próbki → (n_klas, n_cech).
        """
        if isinstance(shap_values, list):
            return np.array([np.asarray(sv)[0] if np.asarray(sv).ndim == 2 else sv for sv in shap_values])
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            n_feat = n_features or sv.shape[1]
            return _to_sample_class_feature(sv, sv.shape[0], N_CLASSES, n_feat)[0]
        if sv.ndim == 2:
            return sv[0:1]
        return sv


# ─────────────────────────────────────────
# Helpery
# ─────────────────────────────────────────
def _import_shap():
    try:
        import shap
    except ImportError as e:
        raise ExplainerUnavailable("Pakiet `shap` nie zainstalowany. Zainstaluj: `pip install shap`") from e
    return shap


def _to_sample_class_feature(raw: Any, n: int, n_classes: int, n_features: int) -> np.ndarray:
    """
    Formaty pakietu shap → (n, C, F).

    Lista per klasa [(n, F)] (starsze wersje) albo (n, F, C) (shap ≥ 0.45);
    (n, F) = model jednowyjściowy.
    """
    if isinstance(raw, list):
        return np.stack([np.asarray(v, dtype=float).reshape(n, n_features) for v in raw], axis=1)
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 2:
        return arr.reshape(n, 1, n_features)
    if arr.ndim == 3 and arr.shape[1] == n_features:
        return arr.transpose(0, 2, 1)
    if arr.ndim == 3 and arr.shape[1:] == (n_classes, n_features):
        return arr
    raise ValueError(f"Nieoczekiwany kształt wartości SHAP: {arr.shape}")


def _safe_float(v: Any) -> float | str:
    """Konwersja na float lub string jeśli się nie da."""
    try:
        f = float(v)
        if np.isnan(f):
            return "NaN"
        return f
    except (TypeError, ValueError):
        return str(v)


# ─────────────────────────────────────────
# Globalne wykresy SHAP
# ─────────────────────────────────────────
def shap_summary_plot(
    shap_values: Any,
    X: pd.DataFrame,
    class_names: list[str] | None = None,
    save_path: Path | str | None = None,
    max_display: int = 20,
    show: bool = False,
    class_idx: int | None = None,
) -> Path | None:
    """
    Globalny summary plot.

    class_idx=None — średnie |SHAP| dla wszystkich klas ESI (wykres słupkowy),
    class_idx=k    — beeswarm dla klasy k (np. 0 = ESI 1).
    `X` musi odpowiadać pozycjom wyjaśnienia (dla EBM: `explainer.feature_frame(X)`).
    """
    try:
        import shap
    except ImportError:
        log.error("Pakiet `shap` nie zainstalowany.")
        return None

    import matplotlib.pyplot as plt

    class_names = class_names or CLASS_NAMES
    values = shap_values
    title = "SHAP — wszystkie poziomy ESI"
    extra: dict[str, Any] = {}
    if class_idx is None and isinstance(shap_values, list) and len(shap_values) == N_CLASSES:
        # słupki w kolejności ESI 1 → 5 i w kolorach TOPSOR (shap indeksuje kolory pozycją słupka)
        extra["class_inds"] = "original"
        extra["color"] = lambda i: TOPSOR_HEX[CLASS_NAMES[i]]
    if class_idx is not None:
        values = shap_values[class_idx] if isinstance(shap_values, list) else np.asarray(shap_values)[:, class_idx, :]
        title = f"SHAP — {class_names[class_idx]}"

    fig = plt.figure(figsize=(10, max(8, max_display * 0.4)))
    try:
        shap.summary_plot(
            values,
            X,
            class_names=class_names,
            max_display=max_display,
            show=False,
            **extra,
        )
        plt.title(title)
    except Exception as e:
        log.warning(f"shap.summary_plot błąd: {e}")
        plt.close("all")
        return None

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Zapisano SHAP summary: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close("all")
    return save_path


def shap_waterfall_plot(
    explainer: SHAPTriageExplainer,
    patient: pd.DataFrame | pd.Series,
    class_idx: int | None = None,
    save_path: Path | str | None = None,
    max_display: int = 15,
    show: bool = False,
) -> Path | None:
    """
    Waterfall plot dla pojedynczego pacjenta — addytywny wkład każdej cechy
    w wynik wybranej klasy (domyślnie: klasa przewidziana przez argmax).
    """
    try:
        import shap
    except ImportError:
        log.error("Pakiet `shap` nie zainstalowany.")
        return None

    import matplotlib.pyplot as plt

    if not explainer.is_fitted:
        explainer.fit()

    X = explainer._prepare(patient).iloc[[0]]
    if class_idx is None:
        class_idx = int(np.argmax(explainer.model.predict_proba(X)[0]))

    values, base = explainer.explain_values(X)
    explanation = shap.Explanation(
        values=values[0, class_idx],
        base_values=float(base[class_idx]),
        data=np.asarray(explainer.feature_frame(X).iloc[0], dtype=float),
        feature_names=explainer.output_feature_names,
    )

    fig = plt.figure(figsize=(10, max(6, max_display * 0.3)))
    try:
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.title(f"{explainer.class_names[class_idx]} ({explainer.output_space})")
    except Exception as e:
        log.warning(f"shap.plots.waterfall błąd: {e}")
        plt.close("all")
        return None

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Zapisano SHAP waterfall: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close("all")
    return save_path
