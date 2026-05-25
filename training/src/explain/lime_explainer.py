"""
LIME — Local Interpretable Model-agnostic Explanations.

LIME tworzy lokalny model liniowy aproksymujący predykcję wokół danego pacjenta.
Wolniejszy niż TreeSHAP, ale agnostyczny co do typu modelu (działa też z ML
modelami nie-drzewowymi).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import CLASS_NAMES
from src.utils.logger import get_logger

log = get_logger(__name__)


class LIMETriageExplainer:
    """Wrapper LIME dla modeli triażowych."""

    def __init__(
        self,
        model: Any,
        training_data: pd.DataFrame,
        class_names: list[str] | None = None,
        categorical_features: list[int] | None = None,
        mode: str = "classification",
    ):
        """
        Parameters
        ----------
        model : BaseTriageModel | sklearn-like
            Model z metodą predict_proba.
        training_data : pd.DataFrame
            Dane treningowe (LIME potrzebuje ich do estymacji rozkładów).
        class_names : list[str], optional
        categorical_features : list[int], optional
            Indeksy kolumn kategorycznych (LIME musi je traktować inaczej).
        """
        self.model = model
        self.training_data = training_data
        self.class_names = class_names or CLASS_NAMES
        self.categorical_features = categorical_features or []
        self.mode = mode
        self.explainer = None
        self.feature_names = list(training_data.columns)

    def fit(self) -> "LIMETriageExplainer":
        """Inicjalizuje LimeTabularExplainer."""
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError as e:
            raise ImportError("Pakiet `lime` nie zainstalowany. Zainstaluj: `pip install lime`") from e

        log.info("Tworzenie LimeTabularExplainer…")
        self.explainer = LimeTabularExplainer(
            training_data=self.training_data.values,
            mode=self.mode,
            feature_names=self.feature_names,
            class_names=self.class_names,
            categorical_features=self.categorical_features,
            discretize_continuous=True,
            random_state=42,
        )
        return self

    def explain_patient(
        self,
        patient: pd.DataFrame | pd.Series,
        num_features: int = 10,
        num_samples: int = 5000,
    ) -> dict[str, Any]:
        """
        Lokalna interpretacja dla pojedynczego pacjenta.

        Parameters
        ----------
        patient : pd.DataFrame (1 row) | pd.Series
        num_features : int
            Ile najważniejszych cech zwrócić.
        num_samples : int
            Ile sampli LIME wygeneruje dookoła pacjenta (więcej = stabilniejsze).

        Returns
        -------
        dict z kluczami:
            - predicted_class
            - probabilities
            - top_features  (lista {feature, weight, condition})
            - lime_explanation  (oryginalny obiekt LIME — do .show_in_notebook())
        """
        if self.explainer is None:
            self.fit()

        # Normalizuj do np.array (LIME wymaga 1D)
        if isinstance(patient, pd.Series):
            instance = patient.values
            patient_df = patient.to_frame().T
        else:
            instance = patient.iloc[0].values
            patient_df = patient

        # Predict proba musi być na 2D arrayu
        def predict_fn(X: np.ndarray) -> np.ndarray:
            X_df = pd.DataFrame(X, columns=self.feature_names)
            return self.model.predict_proba(X_df)

        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples,
        )

        # Predykcja + topowe cechy
        proba = self.model.predict_proba(patient_df)[0]
        predicted_class_idx = int(np.argmax(proba))
        predicted_class = self.class_names[predicted_class_idx]

        # LIME zwraca listę (feature_condition, weight) dla predicted class
        feature_weights = explanation.as_list(label=predicted_class_idx)

        top_features = [
            {
                "condition": str(cond),
                "weight": float(weight),
            }
            for cond, weight in feature_weights
        ]

        return {
            "predicted_class": predicted_class,
            "predicted_class_idx": predicted_class_idx,
            "probabilities": {
                name: float(p) for name, p in zip(self.class_names, proba)
            },
            "top_features": top_features,
            "lime_explanation": explanation,
        }

    def save_html(
        self,
        explanation_dict: dict,
        save_path: Path | str,
    ) -> Path:
        """Zapisuje wyjaśnienie LIME jako HTML (do osadzenia w demo)."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        explanation = explanation_dict.get("lime_explanation")
        if explanation is None:
            raise ValueError("Brak 'lime_explanation' w słowniku.")

        explanation.save_to_file(str(save_path))
        log.info(f"Zapisano LIME HTML: {save_path}")
        return save_path
