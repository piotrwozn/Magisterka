"""
Integracja modeli fundamentalnych (TFM) z pipeline'em treningu — atrapa estymatora (`SOR_AI_TFM_FAKE=1`)
na syntetycznych splitach (`prepared_paths`).

Sprawdzane: `run_training` z wyborem konfiguracji (profil smoke: tag, run_info / DONE.json `pretrained`
+ blok `tfm`, raport, wznawianie wyboru), strażnik CPU przed wczytaniem danych, ignorowane --filter / wagi,
`refit_from_run` dla TFM (nowy seed → nowe worki, "auto" przeliczane), `latest_runs(include_pretrained)`,
'all' = 6 modeli klasycznych (03_train, analizy), 04_evaluate z zapisanych predykcji, odmowy 05 / 06 oraz
pełny proces `scripts/03_train.py --model tfm --tune --profile smoke` (haki środowiskowe w podprocesie).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src.ensemble import latest_runs, list_runs
from src.models import tfm_fake, train
from src.models.tfm_base import TFMDeviceError, TFMPredictor
from src.models.train import DONE_JSON, PROBA_COLS, refit_from_run, run_training
from src.utils import config
from src.utils.config import CLASSICAL_MODELS, TFM_MODELS
from tests.test_ensemble import write_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAST_TFM = {"n_jobs": 1}


@pytest.fixture(autouse=True)
def fake_tfm(monkeypatch):
    monkeypatch.setenv("SOR_AI_TFM_FAKE", "1")
    monkeypatch.setenv("SOR_AI_DEVICE", "cpu")
    for var in ("SOR_AI_DEVICE_TABPFN", "SOR_AI_DEVICE_TABICL", "SOR_AI_TABPFN_MODEL_PATH",
                "SOR_AI_TABICL_MODEL_PATH", tfm_fake.FAKE_OOM_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(tfm_fake, "FAKE_OOM", None)
    for name in ("TABPFN_DEFAULT_PARAMS", "TABICL_DEFAULT_PARAMS"):
        monkeypatch.setitem(getattr(config, name), "n_jobs", 1)


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"tfm_script_{name}", PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tfm_run(prepared_paths) -> Path:
    """Zwykły run profilu smoke z wyborem konfiguracji (TabPFN, atrapa)."""
    return Path(run_training("tabpfn", tune=True, profile="smoke")["run_dir"])


# ─────────────────────────────────────────
# run_training
# ─────────────────────────────────────────
class TestRunTraining:
    def test_smoke_tuned_run(self, tfm_run, prepared_paths):
        for name in ("model.joblib", "run_info.json", "metrics.json", "predictions_val.parquet",
                     "predictions_test.parquet", "report.md", "leaderboard_row.json", DONE_JSON):
            assert (tfm_run / name).exists(), name
        assert not (tfm_run / "feature_importance.csv").exists()
        info = _read(tfm_run / "run_info.json")
        assert info["pretrained"] is True and info["tag"] == "smoke" and info["custom"] is False
        assert info["tuned"] is True and info["early_stopping"] is False and info["n_early_stopping"] == 0
        assert info["device"] == {"requested": "cpu", "actual": "cpu", "fallback_reason": None}
        tuning = info["tuning"]
        assert tuning["kind"] == "tfm_config_selection" and tuning["metric"] == "holdout_logloss"
        assert tuning["n_fold_train"] is None and tuning["n_complete"] == 1
        assert tuning["holdout"]["rows"] == info["n_fit"] // 2           # smoke: 500 → najwyżej połowa X_fit
        tfm = info["tfm"]
        assert tfm["library"] == "tabpfn" and tfm["weights_override"] == "SOR_AI_TFM_FAKE=1"
        assert tfm["licence"] == "non-commercial (TabPFN)" and tfm["n_bags"] == 1
        assert tfm["context_sizes"] == [info["n_fit"]]                  # n_ctx 1000 > X_fit → cały TRAIN
        assert tfm["predict_s"] is not None and tfm["final_budget"]["reduce"] is False
        assert info["params"]["n_ctx"] == 1000 and info["params"]["n_estimators"] == 1
        done = _read(tfm_run / DONE_JSON)
        assert done["pretrained"] is True and done["tfm"]["weights_override"] == "SOR_AI_TFM_FAKE=1"
        assert done["study_name"] == tuning["study_name"] and done["best_trial"] == tuning["best_trial"]
        assert done["actual_device"] == "cpu" and done["tag"] == "smoke"
        report = (tfm_run / "report.md").read_text(encoding="utf-8")
        assert "wybór konfiguracji modelu fundamentalnego" in report and "## Model fundamentalny (TFM)" in report
        assert "PRZEKIEROWANIE WAG" in report
        row = _read(tfm_run / "leaderboard_row.json")
        assert row["cv_metric"] == "holdout_logloss" and row["cv_score"] == tuning["best_value"]
        pred = pd.read_parquet(tfm_run / "predictions_val.parquet")
        np.testing.assert_allclose(pred[PROBA_COLS].sum(axis=1), 1.0, atol=1e-5)
        # Wznowienie: ten sam wybór z plików kandydatów
        again = _read(Path(run_training("tabpfn", tune=True, profile="smoke")["run_dir"]) / "run_info.json")
        assert again["tuning"]["n_cached"] == 1 and again["tuning"]["study_name"] == tuning["study_name"]

    def test_artifact_loads_and_matches_saved_predictions(self, tfm_run):
        from src.data.dataset import get_xy, load_splits
        from src.models import load_model

        model = load_model(tfm_run / "model.joblib")
        assert model.pretrained and isinstance(model.model, TFMPredictor)
        val = load_splits(("val",), feature_set="triage")["val"]
        X, _ = get_xy(val, "triage")
        saved = pd.read_parquet(tfm_run / "predictions_val.parquet").set_index("row_id")
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba, saved.loc[val["row_id"], PROBA_COLS].to_numpy(), atol=1e-6)

    def test_classical_run_info_unchanged(self, prepared_paths):
        result = run_training("logreg", custom_params={"max_iter": 300}, profile="smoke")
        info = _read(Path(result["run_dir"]) / "run_info.json")
        done = _read(Path(result["run_dir"]) / DONE_JSON)
        assert "pretrained" not in info and "tfm" not in info
        assert "pretrained" not in done and "tfm" not in done

    def test_cpu_guard_before_data_and_run_dir(self, prepared_paths):
        with pytest.raises(TFMDeviceError, match="TFM_CPU_MAX_CONTEXT"):
            run_training("tabicl", profile="smoke")                         # domyślne n_ctx 50 000 (→ 6000)
        assert not list(prepared_paths.RUNS_DIR.glob("*_tabicl_*"))
        result = run_training("tabicl", profile="smoke", custom_params={"n_ctx": 300})
        info = _read(Path(result["run_dir"]) / "run_info.json")
        assert info["tag"] == "smoke-custom" and info["tfm"]["context_sizes"] == [300]

    def test_filter_and_weights_ignored(self, prepared_paths):
        result = run_training("tabpfn", tune=True, profile="smoke", do_filter=True, sample_weight_strategy="balanced")
        info = _read(Path(result["run_dir"]) / "run_info.json")
        assert info["feature_filter"] is False and info["overrides"] == ["sample_weight_strategy"]
        assert info["n_features"] == info["n_features_before_filter"]

    def test_final_budget_reduces_bags(self, prepared_paths, monkeypatch):
        from src.models import tfm_tuning

        original = tfm_tuning.select_config

        def slow_selection(*args, **kwargs):
            result = original(*args, **kwargs)
            result["best_params"] = {"n_ctx": 200, "n_estimators": 1, "n_bags": 3}
            result["throughput"] = {"rows_per_s": 0.25, "fit_s_per_bag": 0.0}   # 200 wierszy → 800 s na worek
            return result

        monkeypatch.setattr(tfm_tuning, "select_config", slow_selection)
        result = run_training("tabpfn", tune=True, profile="smoke")
        info = _read(Path(result["run_dir"]) / "run_info.json")
        budget = info["tfm"]["final_budget"]
        assert budget["reduce"] and info["tfm"]["n_bags"] == budget["n_bags"] == 2      # 3 × 800 s > 1800 s
        assert any(e["kind"] == "bags_reduced" for e in info["tfm"]["events"])
        assert info["params"]["n_bags"] == 3                               # parametry bez zmian (refit)


# ─────────────────────────────────────────
# refit_from_run
# ─────────────────────────────────────────
def test_refit_from_run_tfm(prepared_paths):
    source = Path(run_training("tabicl", profile="smoke",
                               custom_params={"n_ctx": 300, "n_bags": "auto", "max_bags": 2})["run_dir"])
    src_info = _read(source / "run_info.json")
    assert src_info["tfm"]["n_bags"] == 2
    refit = refit_from_run(source, seed=3, tag="smoke-seed3", eval_splits=("val",))
    info = _read(Path(refit["run_dir"]) / "run_info.json")
    assert info["refit_of"] == source.name and info["pretrained"] is True
    assert info["params"]["random_state"] == 3 and info["params"]["n_bags"] == "auto"
    assert info["tfm"]["n_bags"] == 2 and info["tfm"]["context_sizes"] == [300, 300]
    assert _read(Path(refit["run_dir"]) / DONE_JSON)["pretrained"] is True
    a = pd.read_parquet(source / "predictions_val.parquet")[PROBA_COLS].to_numpy()
    b = pd.read_parquet(Path(refit["predictions"]["val"]))[PROBA_COLS].to_numpy()
    assert a.shape == b.shape and not np.allclose(a, b)                 # nowy seed → nowe worki
    small = refit_from_run(source, seed=4, tag="smoke-seed4", eval_splits=("val",), fit_fraction=0.2)
    small_info = _read(Path(small["run_dir"]) / "run_info.json")
    assert small_info["tfm"]["n_bags"] == 1                             # "auto" przeliczone dla mniejszego N
    assert "Model fundamentalny" in (Path(small["run_dir"]) / "report.md").read_text(encoding="utf-8")


# ─────────────────────────────────────────
# latest_runs / list_runs
# ─────────────────────────────────────────
def test_latest_runs_include_pretrained(tmp_path):
    y = np.resize(np.arange(5), 50)
    proba = np.full((50, 5), 0.2)
    write_run(tmp_path, "20260101_000001_lightgbm_triage_standard", "lightgbm", {"val": proba}, y)
    tfm_dir = write_run(tmp_path, "20260101_000002_tabpfn_triage_standard", "tabpfn", {"val": proba}, y)
    info = _read(tfm_dir / "run_info.json")
    (tfm_dir / "run_info.json").write_text(json.dumps({**info, "pretrained": True}), encoding="utf-8")
    write_run(tmp_path, "20260101_000003_tabicl_triage_standard", "tabicl", {"val": proba}, y)  # bez flagi
    table = list_runs(tmp_path)
    assert table.set_index("model")["pretrained"].to_dict() == {"lightgbm": False, "tabpfn": True, "tabicl": True}
    names = lambda runs: sorted(r.model for r in runs)                 # noqa: E731
    assert names(latest_runs(runs_dir=tmp_path, tag="standard", data_version=None)) == ["lightgbm"]
    assert names(latest_runs(runs_dir=tmp_path, tag="standard", data_version=None, include_pretrained=True)) == \
        ["lightgbm", "tabicl", "tabpfn"]
    assert names(latest_runs(runs_dir=tmp_path, tag="standard", data_version=None,
                             models=["lightgbm", "tabpfn"])) == ["lightgbm", "tabpfn"]


# ─────────────────────────────────────────
# 'all' = 6 modeli klasycznych
# ─────────────────────────────────────────
@pytest.mark.parametrize("spec, expected", [
    ("all", list(CLASSICAL_MODELS)),
    ("tfm", list(TFM_MODELS)),
    ("all,tfm", list(CLASSICAL_MODELS) + list(TFM_MODELS)),
    ("pfn,lgbm", ["tabpfn", "lightgbm"]),
])
def test_train_cli_model_groups(monkeypatch, spec, expected):
    seen: list[str] = []

    def fake_run_training(model_name, **kwargs):
        seen.append(model_name)
        return {"model_name": model_name, "run_dir": "x", "metrics": {
            "val_argmax_quadratic_weighted_kappa": 0.5, "val_cost_undertriage_rate": 0.1}}

    monkeypatch.setattr(train, "run_training", fake_run_training)
    assert train.main(["--model", spec]) == 0
    assert seen == expected


def test_analysis_cli_all_is_classical():
    from src.analysis.cli import parse_model_list

    assert parse_model_list("all") == list(CLASSICAL_MODELS)
    assert parse_model_list("") == list(CLASSICAL_MODELS)
    assert parse_model_list("tfm") == list(TFM_MODELS)
    assert parse_model_list("lgbm, icl,lgbm") == ["lightgbm", "tabicl"]
    with pytest.raises(ValueError, match="Nieznany model"):
        parse_model_list("svm")


# ─────────────────────────────────────────
# Skrypty 04 / 05 / 06
# ─────────────────────────────────────────
def test_evaluate_uses_saved_predictions(tfm_run, monkeypatch):
    def no_inference(self, frames):
        raise AssertionError("04_evaluate nie może ponownie liczyć predykcji TFM")

    monkeypatch.setattr(TFMPredictor, "predict_proba_many", no_inference)
    script = _load_script("04_evaluate")
    monkeypatch.setattr(sys, "argv", ["04_evaluate.py", "--run", str(tfm_run), "--split", "val",
                                      "--bootstrap", "0", "--no-plots"])
    script.main()
    payload = _read(tfm_run / "figures" / "metrics_val.json")
    assert payload["from_saved_predictions"] is True
    trained = _read(tfm_run / "metrics.json")
    assert payload["metrics"]["val_argmax_log_loss"] == pytest.approx(trained["val_argmax_log_loss"], abs=1e-5)
    assert payload["metrics"]["val_argmax_quadratic_weighted_kappa"] == pytest.approx(
        trained["val_argmax_quadratic_weighted_kappa"], abs=1e-9)


def test_explain_refuses_tfm(tfm_run, monkeypatch, capsys):
    script = _load_script("05_explain")
    with pytest.raises(SystemExit) as err:
        script.main(["--run", str(tfm_run)])
    assert err.value.code == script.TFM_EXIT_CODE == 2
    assert script.latest_run_with_model("triage", include_samples=True) is None    # najnowszy run: bez TFM
    assert script.latest_run_with_model("triage", "tabpfn", include_samples=True).path == tfm_run


def test_cross_validate_refuses_tfm(monkeypatch, capsys):
    script = _load_script("06_cross_validate")
    monkeypatch.setattr(sys, "argv", ["06_cross_validate.py", "--model", "icl"])
    with pytest.raises(SystemExit) as err:
        script.main()
    assert err.value.code == 2
    assert "model fundamentalny" in capsys.readouterr().err


# ─────────────────────────────────────────
# Pełny proces: scripts/03_train.py w podprocesie (haki środowiskowe dziedziczone)
# ─────────────────────────────────────────
def test_train_script_tfm_group_subprocess(prepared_esi_dir, tmp_path):
    data_dir = tmp_path / "esi"
    shutil.copytree(prepared_esi_dir, data_dir)
    env = {**os.environ, "SOR_AI_TFM_FAKE": "1", "SOR_AI_DEVICE": "cpu", "SOR_AI_N_JOBS": "1",
           "SOR_AI_ESI_DATA_DIR": str(data_dir), "SOR_AI_RUNS_DIR": str(tmp_path / "runs"),
           "SOR_AI_LOGS_DIR": str(tmp_path / "logs"), "SOR_AI_NO_AUTO_PREPARE": "1", "PYTHONUTF8": "1"}
    proc = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "03_train.py"), "--model", "tfm",
                           "--tune", "--profile", "smoke"], env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    done = sorted((tmp_path / "runs").glob(f"*/{DONE_JSON}"))
    assert [_read(p)["model"] for p in done] == ["tabpfn", "tabicl"]
    for path in done:
        payload = _read(path)
        assert payload["pretrained"] is True and payload["tag"] == "smoke"
        assert payload["tfm"]["weights_override"] == "SOR_AI_TFM_FAKE=1"
    assert list((tmp_path / "logs" / "optuna").glob("tfm_tabpfn_*.json"))
