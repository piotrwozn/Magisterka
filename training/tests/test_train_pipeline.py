"""
Test end-to-end: syntetyczny surowy Yale → prepare_dataset → run_training → runs/ → src.ensemble.

Wszystko w katalogu tymczasowym (fixture `isolated_paths`); modele w małych
konfiguracjach, żeby cały plik trwał kilkanaście sekund.
"""

from __future__ import annotations

import importlib.util
import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.prepare import prepare_dataset
from src.ensemble import aligned_probas, blend, latest_runs
from src.models import MODEL_REGISTRY, train
from src.models.train import (
    DONE_JSON,
    LEADERBOARD_METRICS,
    LEADERBOARD_ROW_JSON,
    PROBA_COLS,
    rebuild_leaderboard,
    resolve_budget,
    run_training,
)
from src.utils.config import (
    CPU_FALLBACK_BUDGETS,
    DATA_VERSION,
    N_CLASSES,
    PROJECT_ROOT,
    TUNING_PROFILES,
)

FAST = {
    "logreg": {"max_iter": 300},
    "lightgbm": {"n_estimators": 60, "num_leaves": 15, "min_child_samples": 10},
    "random_forest": {"n_estimators": 20},
}
PREDICTION_COLUMNS = ["row_id", "site", "esi", "target", "vitals_complete", *PROBA_COLS,
                      "pred_argmax", "pred_cost"]


@pytest.fixture
def pipeline(isolated_paths, make_raw):
    """Przygotowane splity (podział losowy) w katalogu tymczasowym."""
    prepare_dataset(strategy="random", raw=make_raw(n=1000, seed=11))
    return isolated_paths


def _read_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_run(result: dict, model_name: str, paths) -> dict:
    run_dir = paths.RUNS_DIR / result["run_name"]
    assert result["run_dir"] == str(run_dir)
    assert run_dir.parent == paths.RUNS_DIR
    for name in ["model.joblib", "run_info.json", "metrics.json", "predictions_val.parquet",
                 "predictions_test.parquet", "report.md", LEADERBOARD_ROW_JSON, DONE_JSON]:
        assert (run_dir / name).exists(), name

    metrics = _read_json(run_dir / "metrics.json")
    for split in ("val", "test"):
        for rule in ("argmax", "cost"):
            for m in ["quadratic_weighted_kappa", "undertriage_rate", "critical_miss_rate",
                      "auc_esi1", "log_loss", "brier", "expected_cost", "confusion_matrix"]:
                assert f"{split}_{rule}_{m}" in metrics, f"{split}_{rule}_{m}"
                # Podgrupa z kompletem parametrów życiowych (HR, SBP, RR)
                if m != "auc_esi1":
                    assert f"{split}_vc_{rule}_{m}" in metrics, f"{split}_vc_{rule}_{m}"
        assert f"{split}_admission_auc_model" in metrics
        assert f"{split}_admission_auc_nurse_esi" in metrics
        assert 0 < metrics[f"{split}_vc_n"] < sum(sum(r) for r in metrics[f"{split}_argmax_confusion_matrix"])
    assert -1.0 <= metrics["val_argmax_quadratic_weighted_kappa"] <= 1.0

    info = _read_json(run_dir / "run_info.json")
    assert info["model"] == model_name
    assert info["data_version"] == DATA_VERSION
    assert info["split_strategy"] == "random"
    assert info["proba_columns"] == PROBA_COLS
    assert info["n_fit"] > 0
    uses_es = MODEL_REGISTRY[model_name].supports_early_stopping
    assert info["early_stopping"] is uses_es
    assert (info["n_early_stopping"] > 0) is uses_es
    assert info["device"] == {"requested": "cpu", "actual": "cpu", "fallback_reason": None}
    assert info["weights_strategy"] == "none" and info["class_weights"] is None
    assert info["hardware"]["physical_cores"] >= 1

    for split in ("val", "test"):
        pred = pd.read_parquet(run_dir / f"predictions_{split}.parquet")
        assert list(pred.columns) == PREDICTION_COLUMNS
        np.testing.assert_allclose(pred[PROBA_COLS].sum(axis=1), 1.0, atol=1e-5)
        assert (pred["target"] == pred["esi"] - 1).all()
        assert pred["pred_argmax"].between(0, N_CLASSES - 1).all()
        assert set(pred["vitals_complete"].unique()) <= {0, 1}
        assert pred["row_id"].is_unique
        assert info[f"n_{split}"] == len(pred)

    done = _read_json(run_dir / DONE_JSON)
    assert done["run_name"] == result["run_name"] and done["model"] == model_name
    assert done["data_version"] == DATA_VERSION
    assert done["data_fingerprint"] == info["data_fingerprint"] and len(done["data_fingerprint"]) == 10
    for key in ("feature_set", "tag", "profile", "study_name", "best_trial", "actual_device", "finished_at",
                "custom", "overrides", "tuned", "budget", "device"):
        assert key in done
    assert done["device"] == info["device"] and done["budget"] == info["budget"]
    assert done["custom"] is info["custom"] is bool(info["overrides"]) and done["tuned"] is info["tuned"]
    # DONE.json zapisany jako ostatni artefakt runa
    assert all(p.stat().st_mtime <= (run_dir / DONE_JSON).stat().st_mtime for p in run_dir.iterdir())

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Tuning" in report
    assert report.index("## Metryki — VAL") < report.index("## Test — tylko do raportu końcowego")

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
        for name, result in zip(("logreg", "lightgbm"), results, strict=True):
            assert result["model_name"] == name
            # custom_params → run niestandardowy, nie zwykły run profilu
            assert result["run_name"].endswith(f"_{name}_triage_standard-custom")
            _check_run(result, name, pipeline)
            done = _read_json(pipeline.RUNS_DIR / result["run_name"] / DONE_JSON)
            assert done["custom"] is True and done["overrides"] == ["custom_params"] and done["tuned"] is False

        board = pd.read_csv(pipeline.LEADERBOARD_CSV)
        assert len(board) == 2
        assert list(board["model"]) == ["logreg", "lightgbm"]
        assert set(f"val_argmax_{m}" for m in LEADERBOARD_METRICS) <= set(board.columns)
        assert "val_vc_cost_undertriage_rate" in board.columns
        assert not [c for c in board.columns if c.startswith("test_")]      # leaderboard = tylko VAL
        assert set(board["profile"]) == {"standard"}

        # Runy nadają się do ręcznego łączenia (bez tuningu — domyślnie pomijane)
        assert latest_runs("triage", runs_dir=pipeline.RUNS_DIR) == []
        runs = latest_runs("triage", runs_dir=pipeline.RUNS_DIR, require_tuned=False)
        assert {r.model for r in runs} == {"logreg", "lightgbm"}
        y_val, probas = aligned_probas(runs, "val")
        assert set(probas) == {"logreg", "lightgbm"}
        mixed = blend(probas, {"logreg": 0.3, "lightgbm": 0.7})
        assert mixed.shape == (len(y_val), N_CLASSES)

    def test_rebuild_leaderboard_from_rows(self, pipeline):
        result = run_training("logreg", custom_params=FAST["logreg"], tag="lb")
        pipeline.LEADERBOARD_CSV.unlink()
        # Run przerwany przed DONE.json nie trafia do leaderboardu
        broken = pipeline.RUNS_DIR / "20990101_000000_logreg_triage_crashed"
        broken.mkdir()
        (broken / LEADERBOARD_ROW_JSON).write_text(json.dumps({"run_name": broken.name}), encoding="utf-8")

        path = rebuild_leaderboard()
        assert path == pipeline.LEADERBOARD_CSV
        board = pd.read_csv(path)
        assert list(board["run_name"]) == [result["run_name"]]
        assert rebuild_leaderboard(pipeline.RUNS_DIR) == pipeline.RUNS_DIR / "leaderboard.csv"

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
        info = _read_json(pipeline.RUNS_DIR / result["run_name"] / "run_info.json")
        assert info["feature_set"] == "full"
        assert info["n_features"] > len(_read_json(pipeline.FEATURE_SETS_JSON)["feature_sets"]["triage"])
        assert info["n_fit"] + info["n_early_stopping"] == 400
        assert info["n_val"] == 60
        # Runy na próbkach są domyślnie pomijane przy łączeniu modeli
        assert latest_runs("full", runs_dir=pipeline.RUNS_DIR, require_tuned=False) == []
        assert len(latest_runs("full", include_samples=True, runs_dir=pipeline.RUNS_DIR, require_tuned=False)) == 1

    def test_model_without_early_stopping_fits_whole_train(self, pipeline):
        result = run_training("random_forest", custom_params=FAST["random_forest"], tag="")
        assert result["run_name"].endswith("_random_forest_triage")              # tag '' = bez dopisku
        info = _read_json(pipeline.RUNS_DIR / result["run_name"] / "run_info.json")
        n_train = len(pd.read_parquet(pipeline.TRAIN_PARQUET, columns=["row_id"]))
        assert info["n_fit"] == n_train and info["n_early_stopping"] == 0
        assert info["early_stopping"] is False

    def test_model_alias_and_no_save(self, pipeline):
        result = run_training("lr", custom_params=FAST["logreg"], save_model=False, tag="nosave")
        assert result["model_name"] == "logreg"
        assert result["model_path"] is None
        assert not (pipeline.RUNS_DIR / result["run_name"] / "model.joblib").exists()

    def test_with_tuning(self, pipeline):
        result = run_training("logreg", tune=True, n_trials=2, cv_folds=2, tune_rows=300,
                              custom_params=FAST["logreg"], tag="tuned")
        run_dir = pipeline.RUNS_DIR / result["run_name"]
        info = _read_json(run_dir / "run_info.json")
        tuning = info["tuning"]
        assert info["tuned"] is True
        assert tuning["n_trials"] == 2 and tuning["n_complete"] + tuning["n_pruned"] == 2
        assert tuning["n_rows"] == 300 and tuning["n_folds"] == 2
        assert tuning["config"]["stratified_by"] == "esi_site"           # foldy po ESI × oddział
        assert tuning["config"]["data"] == info["data_fingerprint"]
        assert tuning["config"]["profile"] == "standard"
        assert "C" in tuning["best_params"]
        assert info["params"]["C"] == pytest.approx(tuning["best_params"]["C"])
        assert list(pipeline.OPTUNA_STORAGE_DIR.glob("optuna_logreg.db"))
        assert os.path.exists(tuning["best_json"])

        done = _read_json(run_dir / DONE_JSON)
        assert done["study_name"] == tuning["study_name"] and done["best_trial"] == tuning["best_trial"]
        assert done["tuned"] is True and done["custom"] is True
        assert set(done["overrides"]) == {"n_trials", "n_folds", "tune_rows", "custom_params"}
        assert done["budget"]["n_trials"] == 2 and done["budget"]["n_folds"] == 2
        report = (run_dir / "report.md").read_text(encoding="utf-8")
        assert tuning["study_name"] in report and "| `C` |" in report and "CV logloss" in report
        row = pd.read_csv(pipeline.LEADERBOARD_CSV).iloc[-1]
        assert row["study_name"] == tuning["study_name"] and row["cv_score"] == pytest.approx(tuning["best_value"])

        # Ponowne uruchomienie: badanie ukończone → zero nowych prób, ten sam najlepszy wynik
        again = run_training("logreg", tune=True, n_trials=2, cv_folds=2, tune_rows=300,
                             custom_params=FAST["logreg"], tag="tuned")
        info2 = _read_json(pipeline.RUNS_DIR / again["run_name"] / "run_info.json")
        assert info2["tuning"]["study_name"] == tuning["study_name"]
        assert info2["tuning"]["n_trials_session"] == 0

    def test_qwk_tuning_reports_optuna_class_weights(self, pipeline):
        result = run_training("logreg", tune=True, tune_metric="qwk", n_trials=1, cv_folds=2,
                              custom_params=FAST["logreg"], tag="qwk")
        info = _read_json(pipeline.RUNS_DIR / result["run_name"] / "run_info.json")
        assert info["weights_strategy"] == "optuna_qwk"
        assert set(info["class_weights"]) == {"0", "1", "2", "3", "4"}
        assert info["class_weights"]["2"] == 1.0

    def test_balanced_weights_reported(self, pipeline):
        result = run_training("logreg", custom_params=FAST["logreg"], sample_weight_strategy="balanced",
                              tag="balanced")
        info = _read_json(pipeline.RUNS_DIR / result["run_name"] / "run_info.json")
        assert info["weights_strategy"] == "balanced"
        assert info["class_weights"]["0"] > info["class_weights"]["2"]        # ESI 1 rzadsze niż ESI 3

    def test_filter_without_tune_is_a_noop(self, pipeline, caplog):
        result = run_training("logreg", custom_params=FAST["logreg"], do_filter=True, tag="filter")
        info = _read_json(pipeline.RUNS_DIR / result["run_name"] / "run_info.json")
        assert info["feature_filter"] is False
        assert info["n_features"] == info["n_features_before_filter"]
        assert "--filter działa tylko razem z --tune" in caplog.text

    def test_profile_samples_and_notune_tag(self, pipeline):
        result = run_training("logreg", profile="smoke")
        assert result["run_name"].endswith("_logreg_triage_smoke-notune")      # bez tuningu
        info = _read_json(pipeline.RUNS_DIR / result["run_name"] / "run_info.json")
        assert info["train_sample"] == TUNING_PROFILES["smoke"]["train_sample"]
        assert info["eval_sample"] == TUNING_PROFILES["smoke"]["eval_sample"]
        assert info["budget"]["n_trials"] == TUNING_PROFILES["smoke"]["logreg"]["n_trials"]
        done = _read_json(pipeline.RUNS_DIR / result["run_name"] / DONE_JSON)
        assert done["tag"] == "smoke-notune" and done["custom"] is False and done["tuned"] is False

    def test_plain_tuned_profile_run_gets_profile_tag(self, pipeline):
        result = run_training("logreg", tune=True, profile="smoke")
        assert result["run_name"].endswith("_logreg_triage_smoke")
        done = _read_json(pipeline.RUNS_DIR / result["run_name"] / DONE_JSON)
        assert done["tag"] == "smoke" and done["profile"] == "smoke"
        assert done["custom"] is False and done["overrides"] == [] and done["tuned"] is True
        assert done["budget"]["n_trials"] == TUNING_PROFILES["smoke"]["logreg"]["n_trials"]
        assert done["device"] == {"requested": "cpu", "actual": "cpu", "fallback_reason": None}
        # Run po tuningu = kandydat do porównania (smoke ma próbki → include_samples)
        assert [r.name for r in latest_runs("triage", include_samples=True, runs_dir=pipeline.RUNS_DIR)] == \
            [result["run_name"]]

    def test_explicit_args_override_profile(self, pipeline, monkeypatch):
        from src.models import tuning

        seen = {}

        def fake_tune_model(**kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop po sprawdzeniu argumentów")

        monkeypatch.setattr(tuning, "tune_model", fake_tune_model)
        with pytest.raises(RuntimeError, match="stop po sprawdzeniu"):
            run_training("lightgbm", tune=True, profile="quick", n_trials=7, tune_timeout=99, cv_folds=4,
                         custom_params=FAST["lightgbm"])
        quick = TUNING_PROFILES["quick"]["lightgbm"]
        assert seen["n_trials"] == 7 and seen["timeout"] == 99 and seen["n_folds"] == 4
        assert seen["max_rows"] == quick["tune_rows"] and seen["profile"] == "quick"
        assert seen["fixed_params"] == FAST["lightgbm"]
        assert len(seen["strata"]) == len(seen["X_train"])

    def test_never_loads_real_data(self, isolated_paths):
        """Bez przygotowanych splitów trening próbuje uruchomić prepare — w testach to błąd."""
        with pytest.raises(RuntimeError, match="pełnego zbioru Yale"):
            run_training("logreg", custom_params=FAST["logreg"])


# ─────────────────────────────────────────
# Budżety, skalowanie, predykcje (bez treningu)
# ─────────────────────────────────────────
def test_resolve_budget_profiles_and_cpu_fallback():
    for profile, spec in TUNING_PROFILES.items():
        assert {"train_sample", "eval_sample"} <= set(spec)
        for model in MODEL_REGISTRY:
            budget = resolve_budget(model, profile, device="gpu")
            assert set(budget) >= {"n_trials", "tune_rows", "n_folds", "timeout_s", "train_sample", "eval_sample"}
            assert budget["n_trials"] >= 1 and budget["n_folds"] >= 2 and budget["timeout_s"] > 0
    assert set(TUNING_PROFILES) == {"smoke", "quick", "standard", "thorough"}

    gpu = resolve_budget("xgboost", "standard", device="gpu")
    cpu = resolve_budget("xgboost", "standard", device="cpu")
    assert gpu["n_trials"] == TUNING_PROFILES["standard"]["xgboost"]["n_trials"] and not gpu["cpu_fallback_budget"]
    assert cpu["n_trials"] == CPU_FALLBACK_BUDGETS["standard"]["xgboost"]["n_trials"] and cpu["cpu_fallback_budget"]
    assert cpu["tune_rows"] > 0
    # LightGBM nie ma GPU — profil bez zmian; smoke nie ma budżetu zastępczego
    assert resolve_budget("lightgbm", "standard", device="cpu")["n_trials"] == 120
    assert resolve_budget("catboost", "smoke", device="cpu")["n_trials"] == TUNING_PROFILES["smoke"]["catboost"]["n_trials"]
    # Bez jawnego urządzenia: testy wymuszają CPU (SOR_AI_DEVICE=cpu)
    assert resolve_budget("catboost", "standard")["cpu_fallback_budget"] is True
    with pytest.raises(ValueError, match="Nieznany profil"):
        resolve_budget("logreg", "turbo")


def test_default_run_tag_and_custom_overrides():
    assert train.default_run_tag("standard", tune=True, custom=False) == "standard"
    assert train.default_run_tag("standard", tune=False, custom=False) == "standard-notune"
    assert train.default_run_tag("quick", tune=True, custom=True) == "quick-custom"
    assert train.default_run_tag("quick", tune=False, custom=True) == "quick-custom"

    budget = resolve_budget("logreg", "quick", device="cpu")
    none = dict.fromkeys(("n_trials", "tune_rows", "n_folds", "timeout_s", "train_sample", "eval_sample"))
    plain = {"tune_metric": "logloss", "sample_weight_strategy": "none", "do_filter": False,
             "custom_params": None, "save_model": True}

    def overrides(tune=True, **changes):
        explicit = {**none, **{k: v for k, v in changes.items() if k in none}}
        flags = {**plain, **{k: v for k, v in changes.items() if k in plain}}
        return train._custom_overrides(budget, explicit, tune=tune, **flags)

    assert overrides() == []
    assert overrides(n_trials=budget["n_trials"]) == []                  # jawnie, ale jak w profilu
    assert overrides(n_trials=budget["n_trials"] + 1, tune_timeout=None) == ["n_trials"]
    assert overrides(tune=False, n_trials=99, timeout_s=5) == []         # bez tuningu bez znaczenia
    assert overrides(tune=False, train_sample=123) == ["train_sample"]
    assert overrides(tune=False, tune_metric="qwk", do_filter=True) == []
    assert overrides(tune_metric="qwk", do_filter=True) == ["tune_metric", "do_filter"]
    assert overrides(tune=False, sample_weight_strategy="balanced", custom_params={"C": 2.0},
                     save_model=False) == ["sample_weight_strategy", "custom_params", "no_save"]


def test_scale_count_params_sqrt():
    params = {"min_child_samples": 20, "learning_rate": 0.05}
    info = train._scale_count_params("lightgbm", params, n_fit=40_000, n_fold_train=10_000)
    assert params["min_child_samples"] == 40 and info["factor"] == pytest.approx(2.0)
    params = {"min_child_weight": 1.5}
    train._scale_count_params("xgboost", params, n_fit=90_000, n_fold_train=10_000)
    assert params["min_child_weight"] == pytest.approx(4.5)
    params = {"min_samples_leaf": 3}                     # RF: optimum bezwzględne — bez skalowania
    assert train._scale_count_params("random_forest", params, 40_000, 10_000) == {}
    assert params["min_samples_leaf"] == 3


def test_predictions_frame_without_vitals_complete():
    meta = pd.DataFrame({"row_id": [1, 2], "site": ["A", "B"], "esi": [1, 3], "target": [0, 2]})
    proba = np.array([[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]])
    frame = train._predictions_frame(meta, proba)
    assert list(frame.columns) == PREDICTION_COLUMNS
    assert frame["vitals_complete"].isna().all()
    assert list(frame["pred_argmax"]) == [0, 2]


# ─────────────────────────────────────────
# CLI: kody wyjścia
# ─────────────────────────────────────────
def test_main_exit_codes(monkeypatch):
    calls = []

    def fake_run_training(model_name, **kwargs):
        calls.append(model_name)
        if model_name == "lightgbm":
            raise RuntimeError("symulowany błąd modelu")
        return {"model_name": model_name, "run_dir": "x", "metrics": {
            "val_argmax_quadratic_weighted_kappa": 0.5, "val_argmax_log_loss": 0.8,
            "val_cost_undertriage_rate": 0.1}}

    monkeypatch.setattr(train, "run_training", fake_run_training)
    assert train.main(["--model", "logreg,lightgbm,ebm"]) == 1
    assert calls == ["logreg", "lightgbm", "ebm"]                   # błąd nie przerywa kolejnych
    assert train.main(["--model", "logreg"]) == 0

    def interrupted(model_name, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(train, "run_training", interrupted)
    assert train.main(["--model", "logreg"]) == 130


def test_cli_passes_profile_and_overrides(monkeypatch):
    seen = {}

    def fake_run_training(model_name, **kwargs):
        seen[model_name] = kwargs
        return {"model_name": model_name, "run_dir": "x", "metrics": {
            "val_argmax_quadratic_weighted_kappa": 0.5, "val_cost_undertriage_rate": 0.1}}

    monkeypatch.setattr(train, "run_training", fake_run_training)
    code = train.main(["--model", "xgb", "--tune", "--profile", "quick", "--tune-timeout", "60",
                       "--tune-rows", "0", "--filter"])
    assert code == 0
    kwargs = seen["xgboost"]
    assert kwargs["profile"] == "quick" and kwargs["tune_timeout"] == 60 and kwargs["tune_rows"] == 0
    assert kwargs["n_trials"] is None and kwargs["cv_folds"] is None and kwargs["train_sample"] is None
    assert kwargs["do_filter"] is True and kwargs["tag"] is None


def test_train_script_sets_n_jobs_before_import(monkeypatch):
    spec = importlib.util.spec_from_file_location("train_script", PROJECT_ROOT / "scripts" / "03_train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for var in ("SOR_AI_N_JOBS", "OMP_NUM_THREADS"):
        monkeypatch.setenv(var, "1")          # zapamiętaj stan sprzed testu (przywracany po teście)
        monkeypatch.delenv(var)
    assert module.apply_n_jobs(["--model", "all", "--n-jobs", "6"]) == 6
    assert os.environ["SOR_AI_N_JOBS"] == "6" and os.environ["OMP_NUM_THREADS"] == "6"
    assert module.apply_n_jobs(["--model", "all"]) is None


def test_train_script_disables_power_throttling_first(monkeypatch):
    """Windows (EcoQoS): 03_train.py wyłącza dławienie energii przed treningiem."""
    from src.utils import hardware

    spec = importlib.util.spec_from_file_location("train_script", PROJECT_ROOT / "scripts" / "03_train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    order: list[str] = []
    monkeypatch.setattr(hardware, "disable_power_throttling", lambda: order.append("qos") or False)
    monkeypatch.setattr(train, "main", lambda argv: order.append("main") or 0)
    assert module.run(["--model", "logreg"]) == 0
    assert order == ["qos", "main"]
