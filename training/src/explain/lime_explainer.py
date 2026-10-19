"""
LIME — Local Interpretable Model-agnostic Explanations.

LIME tworzy lokalny model liniowy aproksymujący predykcję wokół danego pacjenta.
Wolniejszy niż TreeSHAP, ale agnostyczny co do typu modelu (działa z każdym
modelem z `predict_proba`, także z regresją logistyczną i EBM).

Braki danych: LIME nie potrafi losować wartości NaN (statystyki i odległości
liczone są na liczbach), dlatego braki w danych treningowych i w wierszu pacjenta
uzupełniamy medianą WYŁĄCZNIE na potrzeby sąsiedztwa LIME. Informacja o braku
pomiaru zostaje w cechach `*_missing`, a predykcja zwracana w wyniku pochodzi
z oryginalnego wiersza (z NaN).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import CLASS_NAMES, RANDOM_SEED
from src.utils.logger import get_logger

log = get_logger(__name__)


class LIMETriageExplainer:
    """Wrapper LIME dla modeli triażowych (klasy ESI 1–5)."""

    def __init__(
        self,
        model: Any,
        training_data: pd.DataFrame,
        class_names: list[str] | None = None,
        categorical_features: list[int] | None = None,
        mode: str = "classification",
        max_rows: int = 5000,
    ):
        """
        Parameters
        ----------
        model : BaseTriageModel | sklearn-like
            Model z metodą predict_proba.
        training_data : pd.DataFrame
            Dane treningowe (LIME estymuje z nich rozkłady cech). Kolumny w kolejności
            cech modelu; jeśli model ma `feature_names`, ramka jest do nich wyrównywana.
        class_names : list[str], optional — domyślnie config.CLASS_NAMES
        categorical_features : list[int], optional
            Indeksy kolumn kategorycznych (LIME traktuje je inaczej).
        max_rows : int
            Limit wierszy tła (statystyki LIME nie potrzebują całego zbioru).
        """
        self.model = model
        names = getattr(model, "feature_names", None) or list(training_data.columns)
        self.feature_names = list(names)
        data = training_data.reindex(columns=self.feature_names)
        if len(data) > max_rows:
            data = data.sample(n=max_rows, random_state=RANDOM_SEED)
        data = data.astype(np.float32)
        # Mediany do uzupełnienia braków (kolumna bez wartości → 0)
        self.fill_values = data.median().fillna(0.0)
        self.training_data = data.fillna(self.fill_values)
        self.class_names = list(class_names or CLASS_NAMES)
        self.categorical_features = categorical_features or []
        self.mode = mode
        self.explainer = None

    def fit(self) -> "LIMETriageExplainer":
        """Inicjalizuje LimeTabularExplainer."""
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError as e:
            raise ImportError("Pakiet `lime` nie zainstalowany. Zainstaluj: `pip install lime`") from e

        log.info("Tworzenie LimeTabularExplainer…")
        self.explainer = LimeTabularExplainer(
            training_data=self.training_data.to_numpy(),
            mode=self.mode,
            feature_names=self.feature_names,
            class_names=self.class_names,
            categorical_features=self.categorical_features,
            discretize_continuous=True,
            random_state=RANDOM_SEED,
        )
        return self

    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(X, columns=self.feature_names).astype(np.float32)
        return self.model.predict_proba(X_df)

    def explain_patient(
        self,
        patient: pd.DataFrame | pd.Series,
        num_features: int = 10,
        num_samples: int = 5000,
        class_idx: int | None = None,
    ) -> dict[str, Any]:
        """
        Lokalna interpretacja dla pojedynczego pacjenta.

        Parameters
        ----------
        patient : pd.DataFrame (1 wiersz) | pd.Series
        num_features : int — ile najważniejszych cech zwrócić
        num_samples : int — ile próbek LIME wygeneruje wokół pacjenta (więcej = stabilniej)
        class_idx : int, optional — klasa do wyjaśnienia (domyślnie argmax)

        Returns
        -------
        dict z kluczami:
            - predicted_class / predicted_class_idx / predicted_esi
            - explained_class
            - probabilities         {nazwa_klasy: p} (z oryginalnego wiersza, z NaN)
            - top_features          lista {condition, weight}
            - imputed_features      cechy pacjenta uzupełnione medianą na potrzeby LIME
            - lime_explanation      oryginalny obiekt LIME (do .show_in_notebook())
        """
        if self.explainer is None:
            self.fit()

        patient_df = patient.to_frame().T if isinstance(patient, pd.Series) else patient.iloc[[0]]
        patient_df = patient_df.reindex(columns=self.feature_names).astype(np.float32)
        imputed = [c for c in self.feature_names if pd.isna(patient_df.iloc[0][c])]
        instance = patient_df.fillna(self.fill_values).iloc[0].to_numpy()

        proba = np.asarray(self.model.predict_proba(patient_df), dtype=float)[0]
        predicted_idx = int(np.argmax(proba))
        explained_idx = predicted_idx if class_idx is None else int(class_idx)

        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self._predict_fn,
            labels=(explained_idx,),
            num_features=num_features,
            num_samples=num_samples,
        )

        # LIME zwraca listę (warunek_na_cechę, waga) dla wyjaśnianej klasy
        top_features = [
            {"condition": str(cond), "weight": float(weight)}
            for cond, weight in explanation.as_list(label=explained_idx)
        ]

        return {
            "predicted_class": self.class_names[predicted_idx],
            "predicted_class_idx": predicted_idx,
            "predicted_esi": predicted_idx + 1,
            "explained_class": self.class_names[explained_idx],
            "explained_class_idx": explained_idx,
            "probabilities": {name: float(p) for name, p in zip(self.class_names, proba)},
            "top_features": top_features,
            "imputed_features": imputed,
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
