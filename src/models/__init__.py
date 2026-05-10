"""Implementacje modeli ML do klasyfikacji triażu MTS."""

from src.models.base import BaseTriageModel, compute_sample_weights
from src.models.ebm_model import EBMTriageModel
from src.models.fusion_model import StackingTriageModel
from src.models.lightgbm_model import LightGBMTriageModel
from src.models.random_forest import RandomForestTriageModel
from src.models.xgboost_model import XGBoostTriageModel

__all__ = [
    "BaseTriageModel",
    "EBMTriageModel",
    "LightGBMTriageModel",
    "RandomForestTriageModel",
    "StackingTriageModel",
    "XGBoostTriageModel",
    "compute_sample_weights",
]


# Rejestr modeli — używany przez `train.py` do wyboru modelu z CLI
MODEL_REGISTRY: dict[str, type[BaseTriageModel]] = {
    "xgboost": XGBoostTriageModel,
    "lightgbm": LightGBMTriageModel,
    "random_forest": RandomForestTriageModel,
    "rf": RandomForestTriageModel,
    "ebm": EBMTriageModel,
    "stacking": StackingTriageModel,
    "fusion": StackingTriageModel,
}


def get_model(name: str, **kwargs) -> BaseTriageModel:
    """Fabryka — zwraca instancję modelu po nazwie."""
    name_lower = name.lower()
    if name_lower not in MODEL_REGISTRY:
        raise ValueError(
            f"Nieznany model: '{name}'. Dostępne: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name_lower](**kwargs)
