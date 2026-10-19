"""
Modele klasyfikacji ESI (5 klas: ESI 1–5).

Zestaw celowo zróżnicowany — pod ręczne łączenie modeli:
    xgboost        gradient boosting, drzewa poziomowe
    lightgbm       gradient boosting, drzewa liść-po-liściu
    catboost       gradient boosting, drzewa symetryczne
    random_forest  bagging (inny mechanizm niż boosting)
    ebm            Explainable Boosting Machine — interpretowalny GAM
    logreg         regresja logistyczna — liniowy punkt odniesienia

Usunięte: hist_gbt (powielał LightGBM), extra_trees (powielał RF),
stacking (łączenie modeli robione ręcznie na predykcjach z runs/).

Wspólny interfejs (src/models/base.py): fit_estimator() (cichy trening, tuning/CV),
fit() (trening z trackingiem), predict_proba() → (n, 5) float64, save()/load_model().
"""

from __future__ import annotations

from pathlib import Path

from src.models.base import BaseTriageModel, compute_sample_weights
from src.models.catboost_model import CatBoostTriageModel
from src.models.ebm_model import EBMTriageModel
from src.models.lightgbm_model import LightGBMTriageModel
from src.models.logreg_model import LogisticRegressionTriageModel
from src.models.random_forest import RandomForestTriageModel
from src.models.xgboost_model import XGBoostTriageModel

__all__ = [
    "BaseTriageModel",
    "CatBoostTriageModel",
    "EBMTriageModel",
    "LightGBMTriageModel",
    "LogisticRegressionTriageModel",
    "RandomForestTriageModel",
    "XGBoostTriageModel",
    "compute_sample_weights",
    "get_model",
    "load_model",
    "resolve_model_name",
]


# Rejestr modeli — kolejność = kolejność treningu w `--model all`
# (od najszybszych do najwolniejszych).
MODEL_REGISTRY: dict[str, type[BaseTriageModel]] = {
    "logreg": LogisticRegressionTriageModel,
    "lightgbm": LightGBMTriageModel,
    "xgboost": XGBoostTriageModel,
    "catboost": CatBoostTriageModel,
    "random_forest": RandomForestTriageModel,
    "ebm": EBMTriageModel,
}

MODEL_ALIASES: dict[str, str] = {
    "lr": "logreg",
    "lgbm": "lightgbm",
    "xgb": "xgboost",
    "cb": "catboost",
    "rf": "random_forest",
}


def resolve_model_name(name: str) -> str:
    key = MODEL_ALIASES.get(name.lower(), name.lower())
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Nieznany model: '{name}'. Dostępne: {list(MODEL_REGISTRY)}")
    return key


def get_model(name: str, **kwargs) -> BaseTriageModel:
    """Fabryka — zwraca instancję modelu po nazwie (lub aliasie)."""
    return MODEL_REGISTRY[resolve_model_name(name)](**kwargs)


def load_model(path: Path | str) -> BaseTriageModel:
    """
    Ładuje zapisany model (`model.save()`), wybierając klasę po nazwie modelu w artefakcie.

    Nie uruchamia wyboru urządzenia ani prób GPU — działa na maszynie bez karty.
    """
    return BaseTriageModel.load(path)
