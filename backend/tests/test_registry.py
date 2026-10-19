"""Tests for the model registry (ESI artifacts from backend/models, models/ and training/runs)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from app.models.registry import TARGET_MODELS, ModelRegistry

CLASS_NAMES = ["ESI1", "ESI2", "ESI3", "ESI4", "ESI5"]
FEATURES = ["age", "triage_vital_hr", "cc_chestpain"]


def _artifact(name: str = "logreg", **overrides) -> dict:
    """Minimal artifact in the format of BaseTriageModel.save()."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, len(FEATURES))), columns=FEATURES)
    y = np.arange(50) % 5
    artifact = {
        "name": name,
        "run_id": "20260101_000000",
        "params": {},
        "model": LogisticRegression(max_iter=200).fit(X, y),
        "feature_names": FEATURES,
        "feature_set": "triage",
        "classes_": np.arange(5),
        "class_names": CLASS_NAMES,
        "label": "esi",
        "data_version": "esi-v1",
        "training_duration_s": 0.1,
    }
    artifact.update(overrides)
    return artifact


def _dump(path: Path, artifact: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def _make_run(runs_dir: Path, run_name: str, model: str = "logreg",
              train_sample: int | None = None, feature_set: str = "triage",
              with_info: bool = True, done: bool = True) -> Path:
    """training/runs/<run_name>/{model.joblib, run_info.json, DONE.json}."""
    run_dir = runs_dir / run_name
    _dump(run_dir / "model.joblib", _artifact(model, run_id=run_name[:15], feature_set=feature_set))
    if with_info:
        (run_dir / "run_info.json").write_text(json.dumps({
            "run_name": run_name, "model": model, "feature_set": feature_set,
            "train_sample": train_sample, "eval_sample": None,
        }), encoding="utf-8")
    if done:
        (run_dir / "DONE.json").write_text(json.dumps({"run_name": run_name, "model": model}),
                                           encoding="utf-8")
    return run_dir


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    """(models_dir, runs_dir) inside a fake project tree: <root>/backend/models, <root>/training/runs."""
    models_dir = tmp_path / "backend" / "models"
    runs_dir = tmp_path / "training" / "runs"
    models_dir.mkdir(parents=True)
    runs_dir.mkdir(parents=True)
    return models_dir, runs_dir


class TestEmptyRegistry:
    def test_empty_registry_has_no_models(self):
        reg = ModelRegistry()
        assert reg.models == {}
        assert reg.medians == {}
        assert reg.loaded_ids == []

    def test_feature_set_alias_resolved(self):
        assert ModelRegistry().feature_set == "triage"            # settings default
        assert ModelRegistry(feature_set="triage_only").feature_set == "triage"
        assert ModelRegistry(feature_set="full").feature_set == "full"

    def test_target_models(self):
        assert set(TARGET_MODELS) == {"xgboost", "lightgbm", "catboost", "random_forest", "ebm", "logreg"}

    def test_loaded_ids_returns_sorted_list(self):
        reg = ModelRegistry()
        reg.models["zebra"] = {"model": None, "feature_names": [], "path": ""}
        reg.models["alpha"] = {"model": None, "feature_names": [], "path": ""}
        assert reg.loaded_ids == ["alpha", "zebra"]


class TestLoadFromEmptyDir:
    def test_no_models_in_empty_dir(self, dirs):
        models_dir, runs_dir = dirs
        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=runs_dir)
        assert reg.models == {}
        assert reg.medians == {}

    def test_loads_medians_when_present(self, dirs):
        models_dir, runs_dir = dirs
        medians_payload = {
            "feature_set": "triage",
            "n_features": 3,
            "medians": {"triage_vital_hr": 84.0, "triage_vital_temp": 36.7, "cc_chestpain": 0.0},
        }
        (models_dir / "imputation_medians.json").write_text(json.dumps(medians_payload), encoding="utf-8")

        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=runs_dir)
        assert reg.medians == medians_payload["medians"]
        assert reg.feature_set == "triage"


class TestLoadArtifacts:
    def test_loads_artifact_metadata(self, dirs):
        models_dir, runs_dir = dirs
        _dump(models_dir / "logreg_20260101.joblib", _artifact("logreg"))
        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=runs_dir)

        bundle = reg.models["logreg"]
        assert bundle["feature_names"] == FEATURES
        assert bundle["classes"] == [0, 1, 2, 3, 4]
        assert bundle["class_names"] == CLASS_NAMES
        assert bundle["feature_set"] == "triage"
        X = pd.DataFrame([[50.0, 90.0, 1.0]], columns=FEATURES)
        assert bundle["model"].predict_proba(X).shape == (1, 5)

    def test_loads_newest_run_per_model(self, dirs):
        models_dir, runs_dir = dirs
        _make_run(runs_dir, "20260101_100000_logreg_triage")
        newest = _make_run(runs_dir, "20260102_100000_logreg_triage_tuned")
        _make_run(runs_dir, "20260101_110000_xgboost_triage", model="xgboost")

        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=runs_dir)
        assert reg.loaded_ids == ["logreg", "xgboost"]
        assert reg.models["logreg"]["path"] == str(newest / "model.joblib")
        assert reg.models["logreg"]["train_sample"] is False

    def test_full_run_preferred_over_newer_sample_run(self, dirs):
        models_dir, runs_dir = dirs
        full = _make_run(runs_dir, "20260101_100000_logreg_triage")
        _make_run(runs_dir, "20260105_100000_logreg_triage_smoke", train_sample=12000)

        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=runs_dir)
        assert reg.models["logreg"]["path"] == str(full / "model.joblib")

    def test_finished_run_preferred_over_newer_interrupted_run(self, dirs):
        models_dir, runs_dir = dirs
        finished = _make_run(runs_dir, "20260101_100000_logreg_triage_standard")
        _make_run(runs_dir, "20260102_100000_logreg_triage_standard", done=False)

        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=runs_dir)
        assert reg.models["logreg"]["path"] == str(finished / "model.joblib")

    def test_interrupted_full_run_beats_finished_sample_run(self, dirs):
        models_dir, runs_dir = dirs
        full = _make_run(runs_dir, "20260101_100000_logreg_triage_standard", done=False)
        _make_run(runs_dir, "20260102_100000_logreg_triage_smoke", train_sample=6000)

        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=runs_dir)
        assert reg.models["logreg"]["path"] == str(full / "model.joblib")

    def test_sample_run_used_as_fallback(self, dirs):
        models_dir, runs_dir = dirs
        smoke = _make_run(runs_dir, "20260105_100000_logreg_triage_smoke", train_sample=12000)

        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=runs_dir)
        assert reg.models["logreg"]["path"] == str(smoke / "model.joblib")
        assert reg.models["logreg"]["train_sample"] is True

    def test_deployment_dir_takes_precedence(self, dirs):
        models_dir, runs_dir = dirs
        deployed = _dump(models_dir / "logreg_deployed.joblib", _artifact("logreg"))
        _make_run(runs_dir, "20990101_000000_logreg_triage")

        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=runs_dir)
        assert reg.models["logreg"]["path"] == str(deployed)

    def test_newest_file_in_deployment_dir(self, dirs):
        models_dir, runs_dir = dirs
        old = _dump(models_dir / "catboost_old.joblib", _artifact("catboost"))
        new = _dump(models_dir / "catboost_new.joblib", _artifact("catboost"))
        past = time.time() - 3600
        os.utime(old, (past, past))

        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=runs_dir)
        assert reg.models["catboost"]["path"] == str(new)

    def test_repo_root_models_dir_searched(self, dirs):
        models_dir, runs_dir = dirs
        root_models = models_dir.parent.parent / "models"
        artifact = _dump(root_models / "lightgbm_20260101.joblib", _artifact("lightgbm"))

        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=runs_dir)
        assert reg.models["lightgbm"]["path"] == str(artifact)

    def test_default_runs_dir_is_training_runs(self, dirs):
        models_dir, runs_dir = dirs
        run = _make_run(runs_dir, "20260101_100000_ebm_triage", model="ebm")
        reg = ModelRegistry()
        reg.load_from_dir(models_dir)            # <root>/training/runs derived from models_dir
        assert reg.models["ebm"]["path"] == str(run / "model.joblib")

    def test_removed_models_not_loaded(self, dirs):
        models_dir, runs_dir = dirs
        _dump(models_dir / "extra_trees_1.joblib", _artifact("extra_trees"))
        _make_run(runs_dir, "20260101_100000_hist_gbt_triage", model="hist_gbt")

        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=runs_dir)
        assert reg.models == {}

    def test_legacy_artifact_skipped(self, dirs):
        models_dir, runs_dir = dirs
        legacy = _artifact("xgboost")
        del legacy["label"]                       # pre-ESI artifact (no native ESI label)
        _dump(models_dir / "xgboost_legacy.joblib", legacy)
        fallback = _make_run(runs_dir, "20260101_100000_xgboost_triage", model="xgboost")

        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=runs_dir)
        assert reg.models["xgboost"]["path"] == str(fallback / "model.joblib")

    def test_wrong_class_count_skipped(self, dirs):
        models_dir, runs_dir = dirs
        _dump(models_dir / "logreg_bad.joblib", _artifact("logreg", class_names=["a", "b", "c"]))
        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=runs_dir)
        assert "logreg" not in reg.models

    def test_other_feature_set_skipped(self, dirs):
        models_dir, runs_dir = dirs
        _make_run(runs_dir, "20260101_100000_logreg_full", feature_set="full")
        _dump(models_dir / "catboost_full.joblib", _artifact("catboost", feature_set="full"))

        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=runs_dir)
        assert reg.models == {}

    def test_incomplete_run_skipped(self, dirs):
        models_dir, runs_dir = dirs
        # model.joblib is saved before run_info.json — a crashed/in-progress run has no info
        _make_run(runs_dir, "20260101_100000_logreg_triage", with_info=False)
        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=runs_dir)
        assert reg.models == {}

    def test_corrupted_file_falls_back(self, dirs):
        models_dir, runs_dir = dirs
        (models_dir / "random_forest_broken.joblib").write_bytes(b"not a joblib file")
        run = _make_run(runs_dir, "20260101_100000_random_forest_triage", model="random_forest")

        reg = ModelRegistry()
        reg.load_from_dir(models_dir, runs_dir=runs_dir)
        assert reg.models["random_forest"]["path"] == str(run / "model.joblib")


@pytest.mark.integration
class TestRealModelLoading:
    def test_loads_real_catboost(self, real_registry):
        assert "catboost" in real_registry.models
        assert len(real_registry.models["catboost"]["feature_names"]) > 200

    def test_loads_real_lightgbm(self, real_registry):
        assert "lightgbm" in real_registry.models
        assert len(real_registry.models["lightgbm"]["feature_names"]) > 200

    def test_real_models_are_esi(self, real_registry):
        for bundle in real_registry.models.values():
            assert bundle["class_names"] == CLASS_NAMES

    def test_medians_are_loaded(self, real_registry):
        assert len(real_registry.medians) > 100
        # Temperature in °C (Yale °F is converted in training/src/data/prepare.py)
        assert 35 < real_registry.medians["triage_vital_temp"] < 39
