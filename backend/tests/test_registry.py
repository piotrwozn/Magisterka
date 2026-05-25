"""Tests for the model registry."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models.registry import ModelRegistry


class TestEmptyRegistry:
    def test_empty_registry_has_no_models(self):
        reg = ModelRegistry()
        assert reg.models == {}
        assert reg.medians == {}
        assert reg.loaded_ids == []

    def test_loaded_ids_returns_sorted_list(self):
        reg = ModelRegistry()
        reg.models["zebra"] = {"model": None, "feature_names": [], "path": ""}
        reg.models["alpha"] = {"model": None, "feature_names": [], "path": ""}
        assert reg.loaded_ids == ["alpha", "zebra"]


class TestLoadFromEmptyDir:
    def test_no_models_in_empty_dir(self, tmp_path: Path):
        reg = ModelRegistry()
        # Need to mock medians otherwise it complains
        reg.load_from_dir(tmp_path)
        assert reg.models == {}

    def test_loads_medians_when_present(self, tmp_path: Path):
        medians_payload = {
            "feature_set": "triage_only",
            "n_features": 3,
            "medians": {"hr": 80.0, "sbp": 120.0, "temp": 98.0},
        }
        (tmp_path / "imputation_medians.json").write_text(json.dumps(medians_payload))

        reg = ModelRegistry()
        reg.load_from_dir(tmp_path)
        assert reg.medians == {"hr": 80.0, "sbp": 120.0, "temp": 98.0}
        assert reg.feature_set == "triage_only"


@pytest.mark.integration
class TestRealModelLoading:
    def test_loads_real_catboost(self, real_registry):
        assert "catboost" in real_registry.models
        assert len(real_registry.models["catboost"]["feature_names"]) > 200

    def test_loads_real_lightgbm(self, real_registry):
        assert "lightgbm" in real_registry.models
        assert len(real_registry.models["lightgbm"]["feature_names"]) > 200

    def test_medians_are_loaded(self, real_registry):
        assert len(real_registry.medians) > 100
        # Yale data uses Fahrenheit, median should be ~98
        assert 95 < real_registry.medians["triage_vital_temp"] < 102
