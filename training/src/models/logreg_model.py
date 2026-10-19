"""
Regresja logistyczna (multinomialna) — liniowy punkt odniesienia.

Po co w pracy: pokazuje, ile zyskują modele nieliniowe względem prostego,
w pełni przejrzystego modelu (standard w artykułach o ML w medycynie ratunkowej).

Pipeline (dopasowany wyłącznie na zbiorze treningowym):
    imputacja medianą + wskaźniki braków → standaryzacja → LogisticRegression.
Wskaźniki braków (`add_indicator=True`) zachowują informację "nie zmierzono" także dla
cech bez własnej kolumny `*_missing` (np. setki kolumn historii w zestawie `full`).
Wątki BLAS ograniczone do config.N_JOBS (przy równoległych procesach nadsubskrypcja
spowalnia lbfgs kilkadziesiąt razy). Brak early stopping — zbiór ES ignorowany.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.models.base import BaseTriageModel
from src.utils import config as _config
from src.utils.config import LOGREG_DEFAULT_PARAMS
from src.utils.logger import get_logger

log = get_logger(__name__)


class LogisticRegressionTriageModel(BaseTriageModel):
    """Multinomialna regresja logistyczna dla 5-klasowej klasyfikacji ESI."""

    name = "logreg"
    display_name = "regresji logistycznej"

    def __init__(self, params: dict[str, Any] | None = None):
        merged = {**LOGREG_DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged)

    def _build_model(self) -> Pipeline:
        return Pipeline([
            # keep_empty_features: kolumna bez wartości w próbce nie znika (stała liczba cech);
            # add_indicator: kolumna 0/1 "brak wartości" dla każdej cechy z brakami w treningu
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(**self.params)),
        ])

    def _fit_impl(self, X, y, X_es, y_es, sample_weight) -> None:
        self.model = self._build_model()
        with threadpool_limits(limits=max(1, int(_config.N_JOBS)), user_api="blas"):
            self.model.fit(X, y, clf__sample_weight=sample_weight)

    def _log_extra_details(self) -> None:
        n_iter = int(np.max(self.model.named_steps["clf"].n_iter_))
        self.train_logger.info(f"Iteracje solvera: {n_iter} (limit {self.params.get('max_iter')})")
        self.tracker.log_metrics({"n_iter": n_iter})

    def transformed_feature_names(self) -> list[str]:
        """Nazwy wejść klasyfikatora: cechy + `missingindicator_<cecha>`."""
        if self.model is None:
            raise RuntimeError("Model nie wytrenowany.")
        return [str(n) for n in self.model[:-1].get_feature_names_out(self.feature_names)]

    def feature_importances(self) -> pd.DataFrame | None:
        """|współczynnik| na standaryzowanych wejściach (cechy i wskaźniki braków), średnio po klasach."""
        if self.model is None:
            return None
        coef = np.abs(self.model.named_steps["clf"].coef_).mean(axis=0)
        try:
            names = self.transformed_feature_names()
        except (AttributeError, ValueError):
            names = [f"f_{i}" for i in range(len(coef))]
        if len(names) != len(coef):
            names = [f"f_{i}" for i in range(len(coef))]
        df = pd.DataFrame({"feature": names, "importance": coef})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
