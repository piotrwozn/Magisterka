"""
SHAP explainer dla modeli triażowych.

Dla XGBoost/LightGBM/RF używamy TreeSHAP — najszybszej i najdokładniejszej
metody (czas O(TLD²) gdzie T=drzewa, L=liście, D=głębokość).

Funkcjonalności:
    - Globalne wyjaśnienia (summary plot, beeswarm)
    - Lokalne wyjaśnienia per-pacjent (waterfall, force plot)
    - Strukturalny dict z top-features for/against każdej klasy
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import (
    CLASS_NAMES,
    FIGURES_DIR,
    SHAP_BACKGROUND_SIZE,
    SHAP_TOP_FEATURES,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


class SHAPTriageExplainer:
    """
    Wrapper SHAP dla modeli triażowych.

    Użycie:
        explainer = SHAPTriageExplainer(model, X_train_sample)
        explainer.fit()  # dla TreeExplainer to trywialne

        # Globalne
        shap_values = explainer.explain_dataset(X_test)

        # Per pacjent
        single = explainer.explain_patient(X_test.iloc[0])
        # → dict: predicted_class, probabilities, top_features_for, top_features_against
    """

    def __init__(
        self,
        model: Any,
        background_data: pd.DataFrame | None = None,
        class_names: list[str] | None = None,
        max_background: int = SHAP_BACKGROUND_SIZE,
    ):
        """
        Parameters
        ----------
        model : BaseTriageModel | sklearn-like
            Model z metodą predict/predict_proba. Akceptuje też nasz
            BaseTriageModel — wyciągamy wewnętrzny obiekt sklearn.
        background_data : pd.DataFrame, optional
            Dane tła (do KernelSHAP). Dla TreeSHAP nieużywane,
            ale używane do określenia kolumn.
        class_names : list[str], optional
        max_background : int
        """
        self.raw_model = self._unwrap_model(model)
        self.model = model
        self.class_names = class_names or CLASS_NAMES
        self.max_background = max_background

        # Ogranicz background do reprezentatywnej próbki
        if background_data is not None and len(background_data) > max_background:
            self.background_data = background_data.sample(
                n=max_background, random_state=42
            )
        else:
            self.background_data = background_data

        self.explainer = None
        self.expected_value: np.ndarray | None = None
        self.is_fitted = False

    @staticmethod
    def _unwrap_model(model: Any) -> Any:
        """Wyciąga obiekt sklearn-like z BaseTriageModel."""
        if hasattr(model, "model") and not callable(model):
            inner = getattr(model, "model", None)
            if inner is not None:
                return inner
        return model

    # ──────── Fit ────────
    def fit(self) -> "SHAPTriageExplainer":
        """Inicjalizuje TreeExplainer."""
        try:
            import shap
        except ImportError as e:
            raise ImportError("Pakiet `shap` nie zainstalowany. Zainstaluj: `pip install shap`") from e

        log.info("Tworzenie TreeExplainer SHAP…")

        # Spróbuj TreeExplainer (XGBoost/LightGBM/RF)
        try:
            self.explainer = shap.TreeExplainer(self.raw_model)
            log.info("Użyto TreeExplainer (szybki)")
        except Exception as e:
            log.warning(f"TreeExplainer nie zadziałał ({e}), fallback do KernelExplainer (powolne)")
            if self.background_data is None:
                raise ValueError(
                    "KernelExplainer wymaga background_data. Przekaż próbkę treningową."
                ) from e
            self.explainer = shap.KernelExplainer(
                self.raw_model.predict_proba,
                self.background_data,
            )

        self.expected_value = self.explainer.expected_value
        self.is_fitted = True
        log.info(f"SHAP fitted. expected_value shape: {np.asarray(self.expected_value).shape}")
        return self

    # ──────── Globalne wyjaśnienia ────────
    def explain_dataset(self, X: pd.DataFrame) -> Any:
        """
        SHAP values dla całego datasetu (lista 5 array dla 5 klas).

        Returns
        -------
        np.ndarray lub list[np.ndarray]
            shap_values[class_idx][sample_idx, feature_idx]
        """
        if not self.is_fitted:
            self.fit()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = self.explainer.shap_values(X)
        return shap_values

    # ──────── Per-pacjent ────────
    def explain_patient(
        self,
        patient: pd.DataFrame | pd.Series,
        top_n: int = SHAP_TOP_FEATURES,
    ) -> dict[str, Any]:
        """
        Strukturalne wyjaśnienie dla pojedynczego pacjenta.

        Parameters
        ----------
        patient : pd.DataFrame (1 wiersz) | pd.Series
        top_n : int
            Ile najważniejszych cech zwrócić.

        Returns
        -------
        dict z kluczami:
            - predicted_class       (str)
            - predicted_class_idx   (int)
            - probabilities         (dict {class_name: float})
            - top_features_for      (list — cechy push do predicted)
            - top_features_against  (list — cechy push przeciw)
            - shap_values_per_class (dict {class_name: list[float]})
        """
        if not self.is_fitted:
            self.fit()

        # Normalizuj do DataFrame
        if isinstance(patient, pd.Series):
            patient_df = patient.to_frame().T
        else:
            patient_df = patient.copy()

        feature_names = list(patient_df.columns)
        patient_values = patient_df.iloc[0].values

        # Predykcja
        proba = self.model.predict_proba(patient_df)[0]
        predicted_class_idx = int(np.argmax(proba))
        predicted_class = self.class_names[predicted_class_idx]

        # SHAP values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = self.explainer.shap_values(patient_df)

        # Format SHAP może być różny — normalizujemy
        sv_per_class = self._normalize_shap_values(shap_values)
        # sv_per_class: shape (n_classes, n_features)

        # Cechy dla predicted class
        sv_predicted = sv_per_class[predicted_class_idx]
        feat_pairs = list(zip(feature_names, sv_predicted, patient_values))
        feat_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        top_for = [
            {
                "feature": str(f),
                "shap_value": float(s),
                "patient_value": _safe_float(v),
            }
            for f, s, v in feat_pairs[:top_n]
            if s > 0
        ]
        top_against = [
            {
                "feature": str(f),
                "shap_value": float(s),
                "patient_value": _safe_float(v),
            }
            for f, s, v in feat_pairs[:top_n]
            if s < 0
        ]

        return {
            "predicted_class": predicted_class,
            "predicted_class_idx": predicted_class_idx,
            "probabilities": {
                name: float(p) for name, p in zip(self.class_names, proba)
            },
            "top_features_for": top_for,
            "top_features_against": top_against,
            "shap_values_per_class": {
                name: sv_per_class[i].tolist()
                for i, name in enumerate(self.class_names)
                if i < len(sv_per_class)
            },
            "feature_names": feature_names,
            "patient_values": [_safe_float(v) for v in patient_values],
        }

    @staticmethod
    def _normalize_shap_values(shap_values: Any) -> np.ndarray:
        """
        SHAP zwraca różne formaty zależnie od wersji i modelu:
            - list[np.ndarray] — jeden array per klasa
            - np.ndarray (n, f, c)
            - np.ndarray (n, f) — tylko jedna klasa

        Normalizujemy do shape (n_classes, n_features) dla pierwszego sampla.
        """
        if isinstance(shap_values, list):
            # list[(n_samples, n_features)]
            arr = np.array([sv[0] if sv.ndim == 2 else sv for sv in shap_values])
            return arr  # shape (n_classes, n_features)

        sv = np.asarray(shap_values)

        if sv.ndim == 3:
            # (n_samples, n_features, n_classes) lub (n_samples, n_classes, n_features)
            # Konwencja XGBoost: (n_samples, n_features, n_classes)
            if sv.shape[2] == 5:
                return sv[0].T  # (n_classes, n_features)
            elif sv.shape[1] == 5:
                return sv[0]  # już (n_classes, n_features)

        if sv.ndim == 2:
            # (n_samples, n_features) — binary lub jedna klasa
            return sv[0:1]  # (1, n_features)

        return sv


# ─────────────────────────────────────────
# Helpery do bezpiecznej konwersji
# ─────────────────────────────────────────
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
) -> Path | None:
    """
    Globalny summary plot (beeswarm).
    Pokazuje wpływ wszystkich cech we wszystkich próbkach.
    """
    try:
        import shap
    except ImportError:
        log.error("Pakiet `shap` nie zainstalowany.")
        return None

    import matplotlib.pyplot as plt

    if class_names is None:
        class_names = CLASS_NAMES

    fig = plt.figure(figsize=(10, max(8, max_display * 0.4)))

    try:
        shap.summary_plot(
            shap_values,
            X,
            class_names=class_names,
            max_display=max_display,
            show=False,
        )
    except Exception as e:
        log.warning(f"shap.summary_plot błąd: {e}")
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
    Waterfall plot dla pojedynczego pacjenta — pokazuje
    additive contribution każdej cechy do finalnej predykcji.
    """
    try:
        import shap
    except ImportError:
        log.error("Pakiet `shap` nie zainstalowany.")
        return None

    import matplotlib.pyplot as plt

    if not explainer.is_fitted:
        explainer.fit()

    if isinstance(patient, pd.Series):
        patient_df = patient.to_frame().T
    else:
        patient_df = patient.copy()

    # Auto-detekcja predicted class
    if class_idx is None:
        proba = explainer.model.predict_proba(patient_df)[0]
        class_idx = int(np.argmax(proba))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.explainer.shap_values(patient_df)

    sv = explainer._normalize_shap_values(shap_values)
    sv_class = sv[class_idx]

    expected_val = explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > class_idx:
        base_val = float(np.asarray(expected_val)[class_idx])
    else:
        base_val = float(np.asarray(expected_val))

    # Buduj obiekt Explanation
    explanation = shap.Explanation(
        values=sv_class,
        base_values=base_val,
        data=patient_df.iloc[0].values,
        feature_names=list(patient_df.columns),
    )

    fig = plt.figure(figsize=(10, max(6, max_display * 0.3)))
    try:
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
    except Exception as e:
        log.warning(f"shap.plots.waterfall błąd: {e}")
        plt.close(fig)
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

    return save_path
