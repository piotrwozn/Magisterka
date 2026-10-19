"""
Test end-to-end: syntetyczny surowy Yale → prepare_dataset → run_training → runs/ → src.ensemble.

Wszystko w katalogu tymczasowym (fixture `isolated_paths`); modele w małych
konfiguracjach, żeby cały plik trwał kilkanaście sekund.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.prepare import prepare_dataset
from src.ensemble import aligned_probas, blend, latest_runs
from src.models import MODEL_REGISTRY
from src.models.train import LEADERBOARD_METRICS, PROBA_COLS, run_training
from src.utils.config import DATA_VERSION, N_CLASSES

FAST = {
    "logreg": {"max_iter": 300},
    "lightgbm": {"n_estimators": 60, "num_leaves": 15, "min_child_samples": 10},
}


@pytest.fixture
def pipeline(isolated_paths, make_raw):
    """Przygotowane splity (podział losowy) w katalogu tymczasowym."""
    prepare_dataset(strategy="random", raw=make_raw(n=1000, seed=11))
    return isolated_paths


def _check_run(result: dict, model_name: str, paths) -> dict:
    run_dir = paths.RUNS_DIR / result["run_name"]
    assert result["run_dir"] == str(run_dir)
    assert run_dir.parent == paths.RUNS_DIR
    for name in ["model.joblib", "run_info.json", "metrics.json", "predictions_val.parquet",
                 "predictions_test.parquet", "report.md"]:
        assert (run_dir / name).exists(), name

    metrics = json.loads((run_dir / "metrics.json").read_text())
    for split in ("val", "test"):
        for rule in ("argmax", "cost"):
            for m in ["quadratic_weighted_kappa", "undertriage_rate", "critical_miss_rate",
                      "auc_esi1", "log_loss", "brier", "expected_cost", "confusion_matrix"]:
                assert f"{split}_{rule}_{m}" in metrics, f"{split}_{rule}_{m}"
        assert f"{split}_admission_auc_model" in metrics
        assert f"{split}_admission_auc_nurse_esi" in metrics
    assert -1.0 <= metrics["val_argmax_quadratic_weighted_kappa"] <= 1.0

    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["model"] == model_name
    assert info["data_version"] == DATA_VERSION
    assert info["split_strategy"] == "random"
    assert info["proba_columns"] == PROBA_COLS
    assert info["n_fit"] > 0 and info["n_early_stopping"] > 0

    for split in ("val", "test"):
        pred = pd.read_parquet(run_dir / f"predictions_{split}.parquet")
        assert list(pred.columns) == ["row_id", "site", "esi", "target", *PROBA_COLS,
                                      "pred_argmax", "pred_cost"]
        assert PROBA_COLS == [f"p_esi{i}" for i in range(1, 6)]
        np.testing.assert_allclose(pred[PROBA_COLS].sum(axis=1), 1.0, atol=1e-5)
        assert (pred["target"] == pred["esi"] - 1).all()
        assert pred["pred_argmax"].between(0, N_CLASSES - 1).all()
        assert pred["row_id"].is_unique
        assert info[f"n_{split}"] == len(pred)

    artifact = joblib.load(run_dir / "model.joblib")
    assert artifact["name"] == model_name
    assert artifact["label"] == "esi"
    assert artifact["data_version"] == DATA_VERSION
    return metrics


class TestRunTraining:
    def test_logreg_and_lightgbm(self, pipeline):
        results = [
            run_training(name, feature_set="triage", custom_params=FAST[name])
            for name in ("logreg", "lightgbm")
        ]
        for name, result in zip(("logreg", "lightgbm"), results):
            assert result["model_name"] == name
            _check_run(result, name, pipeline)

        board = pd.read_csv(pipeline.LEADERBOARD_CSV)
        assert len(board) == 2
        assert list(board["model"]) == ["logreg", "lightgbm"]
        assert "val_argmax_quadratic_weighted_kappa" in board.columns
        assert "test_cost_undertriage_rate" in board.columns
        assert set(f"val_argmax_{m}" for m in LEADERBOARD_METRICS) <= set(board.columns)

        # Runy nadają się do ręcznego łączenia
        runs = latest_runs("triage", runs_dir=pipeline.RUNS_DIR)
        assert {r.model for r in runs} == {"logreg", "lightgbm"}
        y_val, probas = aligned_probas(runs, "val")
        assert set(probas) == {"logreg", "lightgbm"}
        mixed = blend(probas, {"logreg": 0.3, "lightgbm": 0.7})
        assert mixed.shape == (len(y_val), N_CLASSES)

    def test_saved_model_reproduces_predictions(self, pipeline):
        from src.data.dataset import get_xy, load_split

        result = run_training("logreg", custom_params=FAST["logreg"], tag="repro")
        assert result["run_name"].endswith("_logreg_triage_repro")
        artifact = joblib.load(result["model_path"])
        model = MODEL_REGISTRY[artifact["name"]].load(result["model_path"])

        X_val, _ = get_xy(load_split("val"), artifact["feature_set"])
        proba = model.predict_proba(X_val[model.feature_names])
        saved = pd.read_parquet(f"{result['run_dir']}/predictions_val.parquet")
        np.testing.assert_allclose(saved[PROBA_COLS].to_numpy(), proba, atol=1e-6)

    def test_full_feature_set_with_samples(self, pipeline):
        result = run_training("lightgbm", feature_set="full", custom_params=FAST["lightgbm"],
                              train_sample=400, eval_sample=60, tag="sample")
        info = json.loads((pipeline.RUNS_DIR / result["run_name"] / "run_info.json").read_text())
        assert info["feature_set"] == "full"
        assert info["n_features"] > len(json.loads(pipeline.FEATURE_SETS_JSON.read_text())
                                        ["feature_sets"]["triage"])
        assert info["n_fit"] + info["n_early_stopping"] == 400
        assert info["n_val"] == 60
        # Runy na próbkach są domyślnie pomijane przy łączeniu modeli
        assert latest_runs("full", runs_dir=pipeline.RUNS_DIR) == []
        assert len(latest_runs("full", include_samples=True, runs_dir=pipeline.RUNS_DIR)) == 1

    def test_model_alias_and_no_save(self, pipeline):
        result = run_training("lr", custom_params=FAST["logreg"], save_model=False, tag="nosave")
        assert result["model_name"] == "logreg"
        assert result["model_path"] is None
        assert not (pipeline.RUNS_DIR / result["run_name"] / "model.joblib").exists()

    def test_with_tuning(self, pipeline):
        result = run_training("logreg", tune=True, n_trials=2, cv_folds=2, tune_rows=300,
                              custom_params=FAST["logreg"], tag="tuned")
        info = json.loads((pipeline.RUNS_DIR / result["run_name"] / "run_info.json").read_text())
        assert info["tuned"] is True
        assert info["tuning"]["n_trials"] == 2
        assert "C" in info["tuning"]["best_params"]
        assert info["params"]["C"] == pytest.approx(info["tuning"]["best_params"]["C"])
        assert list(pipeline.OPTUNA_STORAGE_DIR.glob("optuna_logreg.db"))

    def test_never_loads_real_data(self, isolated_paths):
        """Bez przygotowanych splitów trening próbuje uruchomić prepare — w testach to błąd."""
        with pytest.raises(RuntimeError, match="pełnego zbioru Yale"):
            run_training("logreg", custom_params=FAST["logreg"])
