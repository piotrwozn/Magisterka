"""Testy modeli (smoke tests + sprawdzenie logowania)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


class TestXGBoostBasic:
    def test_can_train_and_predict(self, synthetic_X_y, tmp_path):
        """Smoke test: model trenuje się i robi predykcje."""
        from src.models.xgboost_model import XGBoostTriageModel

        X, y = synthetic_X_y

        # Mała konfiguracja by trening był szybki
        model = XGBoostTriageModel(params={
            "n_estimators": 10,
            "max_depth": 3,
            "early_stopping_rounds": None,
        })
        model.fit(
            X, y,
            X_val=X, y_val=y,
            sample_weight_strategy="custom",
            run_id="test_run",
        )

        assert model.is_fitted
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert set(preds.tolist()).issubset(set(range(5)))

        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 5)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)

    def test_logs_are_created(self, synthetic_X_y, tmp_path):
        """Trening musi zapisać log treningu i JSON eksperymentu."""
        from src.models.xgboost_model import XGBoostTriageModel

        X, y = synthetic_X_y
        model = XGBoostTriageModel(params={"n_estimators": 5, "early_stopping_rounds": None})
        model.fit(X, y, X_val=X, y_val=y, run_id="test_logging_xgb")

        # Plik logu musi istnieć
        assert model.train_log_path is not None
        assert Path(model.train_log_path).exists()
        log_content = Path(model.train_log_path).read_text(encoding="utf-8")
        assert "Trenowanie XGBoost" in log_content
        assert "Trening XGBoost zakończony" in log_content

        # JSON eksperymentu musi istnieć
        assert model.tracker is not None
        assert Path(model.tracker.json_path).exists()

        with open(model.tracker.json_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["model_name"] == "xgboost"
        assert data["run_id"] == "test_logging_xgb"
        assert "training_history" in data
        assert len(data["training_history"]) > 0
        assert "params" in data
        assert "duration_seconds" in data
        assert data["duration_seconds"] > 0


class TestLightGBM:
    def test_can_train(self, synthetic_X_y):
        from src.models.lightgbm_model import LightGBMTriageModel

        X, y = synthetic_X_y
        model = LightGBMTriageModel(params={
            "n_estimators": 10,
            "max_depth": 3,
        })
        model.fit(X, y, X_val=X, y_val=y, run_id="test_lgbm")

        assert model.is_fitted
        preds = model.predict(X)
        assert preds.shape == (len(X),)


class TestRandomForest:
    def test_can_train(self, synthetic_X_y):
        from src.models.random_forest import RandomForestTriageModel

        X, y = synthetic_X_y
        model = RandomForestTriageModel(params={"n_estimators": 10, "max_depth": 5})
        model.fit(X, y, X_val=X, y_val=y, run_id="test_rf")

        assert model.is_fitted


class TestSaveLoad:
    def test_roundtrip(self, synthetic_X_y, tmp_path):
        from src.models.xgboost_model import XGBoostTriageModel

        X, y = synthetic_X_y
        model = XGBoostTriageModel(params={"n_estimators": 10, "early_stopping_rounds": None})
        model.fit(X, y, X_val=X, y_val=y, run_id="test_save_load")

        save_path = tmp_path / "test_model.joblib"
        model.save(save_path)
        assert save_path.exists()

        # Load
        loaded = XGBoostTriageModel.load(save_path)
        assert loaded.is_fitted

        # Predykcje muszą się zgadzać
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


class TestModelRegistry:
    def test_get_model(self):
        from src.models import get_model

        for name in ["xgboost", "lightgbm", "random_forest", "rf", "ebm", "stacking"]:
            model = get_model(name)
            assert model is not None
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")

    def test_unknown_model_raises(self):
        from src.models import get_model

        with pytest.raises(ValueError, match="Nieznany model"):
            get_model("xxx-nonexistent-xxx")


class TestSampleWeights:
    def test_custom_weights(self):
        from src.models.base import compute_sample_weights

        y = np.array([0, 0, 1, 2, 3, 4])
        weights = compute_sample_weights(y, strategy="custom")
        assert len(weights) == len(y)
        # Red (0) ma wagę 10, Yellow (2) ma 1
        assert weights[0] > weights[3]  # Red > Yellow

    def test_balanced_weights(self):
        from src.models.base import compute_sample_weights

        y = np.array([0, 0, 1, 2, 3, 4])
        weights = compute_sample_weights(y, strategy="balanced")
        assert len(weights) == len(y)
        assert (weights > 0).all()

    def test_none_weights(self):
        from src.models.base import compute_sample_weights

        y = np.array([0, 1, 2, 3, 4])
        weights = compute_sample_weights(y, strategy="none")
        assert (weights == 1.0).all()
