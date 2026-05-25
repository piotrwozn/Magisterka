"""
DiCE — Diverse Counterfactual Explanations.

Counterfactuals odpowiadają na pytanie:
    "Co by się musiało zmienić, żeby pacjent był Orange zamiast Yellow?"

To kluczowa warstwa eksplanacyjna dla decyzji medycznych:
    - Lekarz widzi nie tylko DLACZEGO model wybrał Yellow,
      ale też CO musi się zmienić, by uzasadnić upgrade.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import CLASS_NAMES
from src.utils.logger import get_logger

log = get_logger(__name__)


class DiCEExplainer:
    """Wrapper DiCE dla modeli triażowych."""

    def __init__(
        self,
        model: Any,
        training_data: pd.DataFrame,
        outcome_name: str = "mts_numeric",
        continuous_features: list[str] | None = None,
        method: str = "random",
    ):
        """
        Parameters
        ----------
        model : BaseTriageModel | sklearn-like
        training_data : pd.DataFrame
            Z kolumną outcome (target). Używana do estymacji feasible ranges.
        outcome_name : str
        continuous_features : list[str], optional
            Lista cech ciągłych (np. vital signs). Auto-detekcja jeśli None.
        method : str
            'random' (domyślne, najszybszy) | 'genetic' | 'kdtree'
        """
        self.model = model
        self.training_data = training_data
        self.outcome_name = outcome_name
        self.method = method

        # Auto-detekcja cech ciągłych
        if continuous_features is None:
            continuous_features = []
            for col in training_data.columns:
                if col == outcome_name:
                    continue
                if pd.api.types.is_numeric_dtype(training_data[col]):
                    nunique = training_data[col].nunique()
                    # Heurystyka: jeśli więcej niż 20 unikalnych wartości — ciągłe
                    if nunique > 20:
                        continuous_features.append(col)
        self.continuous_features = continuous_features

        self.dice_data = None
        self.dice_model = None
        self.explainer = None

    def fit(self) -> "DiCEExplainer":
        """Inicjalizuje DiCE."""
        try:
            import dice_ml
        except ImportError as e:
            raise ImportError(
                "Pakiet `dice-ml` nie zainstalowany. Zainstaluj: `pip install dice-ml`"
            ) from e

        log.info("Tworzenie DiCE explainer…")

        self.dice_data = dice_ml.Data(
            dataframe=self.training_data,
            continuous_features=self.continuous_features,
            outcome_name=self.outcome_name,
        )

        # DiCE potrzebuje sklearn-like model
        sklearn_model = self.model.model if hasattr(self.model, "model") else self.model

        self.dice_model = dice_ml.Model(
            model=sklearn_model,
            backend="sklearn",
            model_type="classifier",
        )

        self.explainer = dice_ml.Dice(
            self.dice_data,
            self.dice_model,
            method=self.method,
        )
        return self

    def generate_counterfactuals(
        self,
        patient: pd.DataFrame,
        desired_class: int,
        total_cfs: int = 3,
        features_to_vary: list[str] | None = None,
        permitted_range: dict[str, list] | None = None,
    ) -> dict[str, Any]:
        """
        Generuje counterfactuals dla pacjenta.

        Parameters
        ----------
        patient : pd.DataFrame (1 row)
            Bez kolumny outcome.
        desired_class : int
            Pożądana klasa (np. 1 = Orange).
        total_cfs : int
            Liczba counterfactuals.
        features_to_vary : list[str], optional
            Które cechy mogą się zmieniać (None = wszystkie poza target).
            Dla SOR: vital signs są variable, ale demographics fixed.
        permitted_range : dict, optional
            Dozwolone zakresy: {feature: [min, max]}.

        Returns
        -------
        dict z kluczami:
            - original         (pd.DataFrame)
            - counterfactuals  (pd.DataFrame)
            - changes          (list[dict] — co się zmieniło)
        """
        if self.explainer is None:
            self.fit()

        # DiCE chce features_to_vary jako string "all" lub list
        if features_to_vary is None:
            features_to_vary_arg = "all"
        else:
            features_to_vary_arg = features_to_vary

        # Drop outcome jeśli jest
        query = patient.copy()
        if self.outcome_name in query.columns:
            query = query.drop(columns=[self.outcome_name])

        try:
            cfs = self.explainer.generate_counterfactuals(
                query_instances=query,
                total_CFs=total_cfs,
                desired_class=desired_class,
                features_to_vary=features_to_vary_arg,
                permitted_range=permitted_range,
                verbose=False,
            )
        except Exception as e:
            log.warning(f"DiCE nie znalazł counterfactuals: {e}")
            return {
                "original": patient,
                "counterfactuals": pd.DataFrame(),
                "changes": [],
                "error": str(e),
            }

        # Wyciągnij DataFrame counterfactuals
        cf_examples = cfs.cf_examples_list[0] if cfs.cf_examples_list else None
        if cf_examples is None or cf_examples.final_cfs_df is None:
            return {
                "original": patient,
                "counterfactuals": pd.DataFrame(),
                "changes": [],
            }

        cf_df = cf_examples.final_cfs_df.copy()

        # Wylicz zmiany
        original_values = query.iloc[0]
        changes_list = []
        for idx, cf_row in cf_df.iterrows():
            row_changes = {}
            for col in query.columns:
                if col not in cf_df.columns:
                    continue
                orig = original_values[col]
                new = cf_row[col]
                try:
                    if not pd.isna(orig) and not pd.isna(new) and not np.isclose(float(orig), float(new), rtol=1e-3):
                        row_changes[col] = {
                            "from": float(orig),
                            "to": float(new),
                            "delta": float(new) - float(orig),
                        }
                except (ValueError, TypeError):
                    if str(orig) != str(new):
                        row_changes[col] = {"from": str(orig), "to": str(new)}
            changes_list.append(row_changes)

        return {
            "original": patient,
            "counterfactuals": cf_df,
            "changes": changes_list,
            "desired_class": desired_class,
            "desired_class_name": (
                CLASS_NAMES[desired_class] if 0 <= desired_class < len(CLASS_NAMES) else str(desired_class)
            ),
        }

    def format_changes_text(
        self,
        cf_result: dict,
        original_class_name: str | None = None,
    ) -> str:
        """Formatuje zmiany counterfactual jako czytelny tekst po polsku."""
        if not cf_result.get("changes"):
            return "Nie udało się znaleźć counterfactuals."

        out = []
        target_name = cf_result.get("desired_class_name", "?")
        if original_class_name:
            out.append(f"Aby zmienić klasyfikację z {original_class_name} na {target_name}, należałoby:\n")
        else:
            out.append(f"Aby uzyskać klasyfikację {target_name}, należałoby:\n")

        for i, changes in enumerate(cf_result["changes"][:3], start=1):
            out.append(f"\nScenariusz {i}:")
            if not changes:
                out.append("  (brak zmian wymaganych)")
                continue
            for feat, change in changes.items():
                if "delta" in change:
                    direction = "↑" if change["delta"] > 0 else "↓"
                    out.append(
                        f"  • {feat}: {change['from']:.2f} {direction} {change['to']:.2f} "
                        f"(zmiana o {change['delta']:+.2f})"
                    )
                else:
                    out.append(f"  • {feat}: {change['from']} → {change['to']}")

        return "\n".join(out)
