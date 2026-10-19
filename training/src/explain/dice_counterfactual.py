"""
DiCE — Diverse Counterfactual Explanations.

Counterfactuals odpowiadają na pytanie:
    "Co musiałoby się zmienić, żeby pacjent był ESI 2 zamiast ESI 3?"

To kluczowa warstwa eksplanacyjna dla decyzji medycznych:
    - Lekarz widzi nie tylko DLACZEGO model wybrał ESI 3,
      ale też CO musiałoby się zmienić, by uzasadnić podniesienie pilności.

Ograniczenia:
    - DiCE nie obsługuje NaN — braki w danych tła i w wierszu pacjenta są
      uzupełniane medianą (informacja o braku zostaje w cechach `*_missing`).
    - DiCE zmienia cechy niezależnie: cechy pochodne (shock_index, news2_score …)
      NIE są przeliczane po zmianie parametrów życiowych — interpretuj ostrożnie.
    - Wszystkie cechy traktowane są jako ciągłe: cechy kategoryczne DiCE zamienia
      na napisy, czego modele numeryczne (XGBoost, LightGBM …) nie przyjmują.
      Dlatego domyślnie zmieniane są TYLKO parametry życiowe (`DEFAULT_FEATURES_TO_VARY`);
      `features_to_vary="all"` pozwala zmieniać wszystko (także flagi 0/1 → wartości ułamkowe).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import CLASS_NAMES, CLASS_NAMES_PL, TARGET_COL
from src.utils.logger import get_logger

log = get_logger(__name__)

# Parametry życiowe z triażu — domyślnie jedyne cechy zmieniane w counterfactuals
DEFAULT_FEATURES_TO_VARY: list[str] = [
    "triage_vital_hr",
    "triage_vital_sbp",
    "triage_vital_dbp",
    "triage_vital_rr",
    "triage_vital_o2",
    "triage_vital_temp",
]


class DiCEExplainer:
    """Wrapper DiCE dla modeli triażowych."""

    def __init__(
        self,
        model: Any,
        training_data: pd.DataFrame,
        outcome_name: str = TARGET_COL,
        continuous_features: list[str] | None = None,
        method: str = "random",
    ):
        """
        Parameters
        ----------
        model : BaseTriageModel | sklearn-like
        training_data : pd.DataFrame
            Cechy modelu + kolumna outcome (indeks klasy 0–4 = ESI 1–5).
            Używana do estymacji dopuszczalnych zakresów.
        outcome_name : str
            Kolumna z klasą (domyślnie config.TARGET_COL = 'target').
        continuous_features : list[str], optional
            Lista cech ciągłych. None = wszystkie cechy modelu (patrz uwagi w nagłówku modułu).
        method : str
            'random' (domyślne, najszybszy) | 'genetic' | 'kdtree'
        """
        self.model = model
        self.outcome_name = outcome_name
        self.method = method

        # Kolejność cech zgodna z modelem; braki → mediana (DiCE nie obsługuje NaN)
        names = getattr(model, "feature_names", None) or [
            c for c in training_data.columns if c != outcome_name
        ]
        self.feature_names = list(names)
        features = training_data.reindex(columns=self.feature_names).astype(np.float64)
        self.fill_values = features.median().fillna(0.0)
        self.training_data = features.fillna(self.fill_values)
        self.training_data[outcome_name] = training_data[outcome_name].astype(int).to_numpy()

        # Wszystkie cechy jako ciągłe (kategorie DiCE = napisy, nieakceptowane przez modele)
        self.continuous_features = list(continuous_features or self.feature_names)

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
            Pożądany indeks klasy 0–4 (np. 1 = ESI 2).
        total_cfs : int
            Liczba counterfactuals.
        features_to_vary : list[str] | 'all', optional
            Które cechy mogą się zmieniać. None = parametry życiowe obecne w modelu
            (`DEFAULT_FEATURES_TO_VARY`) — demografia, przybycie i skargi stałe.
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
            features_to_vary_arg = [c for c in DEFAULT_FEATURES_TO_VARY if c in self.feature_names]
        else:
            features_to_vary_arg = features_to_vary

        # Kolumny modelu (bez outcome), braki → mediana
        query = patient.reindex(columns=self.feature_names).astype(np.float64).fillna(self.fill_values)

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
            "desired_class_pl": (
                CLASS_NAMES_PL[desired_class] if 0 <= desired_class < len(CLASS_NAMES_PL) else str(desired_class)
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
        # Krótka etykieta: "ESI 2 · pomarańczowy" (bez czasu oczekiwania)
        target_name = str(cf_result.get("desired_class_pl") or cf_result.get("desired_class_name", "?")).split(" (")[0]
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
