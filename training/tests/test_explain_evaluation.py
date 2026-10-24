"""
Testy warstwy wyjaśnień i ewaluacji: kolory TOPSOR z config, etykiety przybycia,
helpery demo Streamlit, podgrupy w 04_evaluate (komplet parametrów, oddziały),
stratyfikacja CV po ESI × oddział, 05_explain na syntetycznych splitach.

Wszystko na syntetycznych danych (`prepared_paths` z conftest) i w katalogach tymczasowych.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src.data.prepare import SITE_COL, VITALS_COMPLETE_COL
from src.data.yale_schema import ARRIVAL_CATEGORIES, TRAUMA_ACTIVATION_COL
from src.utils import config
from src.utils.config import CLASS_NAMES, DATA_VERSION, TARGET_COL, TOPSOR_HEX

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str) -> ModuleType:
    """Import skryptu z scripts/ (nazwa zaczyna się od cyfry — przez importlib)."""
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evaluate_script() -> ModuleType:
    return _load_script("04_evaluate")


def _fitted_logreg(prepared_paths, feature_set: str = "triage"):
    """Mała regresja logistyczna na syntetycznym TRAIN (zapisana w katalogu tymczasowym)."""
    from src.data.dataset import get_xy, load_splits
    from src.models import get_model

    train = load_splits(("train",), feature_set=feature_set)["train"]
    X, y = get_xy(train, feature_set)
    model = get_model("logreg", params={"max_iter": 300})
    model.fit_estimator(X, y)
    model.feature_set = feature_set
    return model


# ─────────────────────────────────────────
# Kolory TOPSOR — jedna tabela w config
# ─────────────────────────────────────────
class TestTopsorColors:
    def test_hex_keyed_by_class_names(self):
        assert list(TOPSOR_HEX) == CLASS_NAMES
        assert all(v.startswith("#") and len(v) == 7 for v in TOPSOR_HEX.values())

    def test_visualizations_use_config_table(self):
        from src.evaluation import visualizations

        assert visualizations.ESI_COLORS == [TOPSOR_HEX[name] for name in CLASS_NAMES]
        assert visualizations.TOPSOR_HEX is config.TOPSOR_HEX

    def test_shap_explainer_uses_config_table(self):
        from src.explain import shap_explainer

        assert shap_explainer.TOPSOR_HEX is config.TOPSOR_HEX

    def test_only_topsor_wait_table(self):
        """Jedna tabela czasów oczekiwania (TOPSOR) — bez aliasów pod innymi nazwami."""
        wait_tables = [n for n in dir(config) if n.endswith("MAX_WAIT_MINUTES")]
        assert wait_tables == ["TOPSOR_MAX_WAIT_MINUTES"]


# ─────────────────────────────────────────
# Sposób przybycia (ambulance / self / other) w promptach i demo
# ─────────────────────────────────────────
class TestArrivalLabels:
    def test_ollama_labels_match_categories(self):
        from src.explain.ollama_medical import ARRIVAL_LABELS_PL, _arrival_label

        assert list(ARRIVAL_LABELS_PL) == [f"arrival_{c}" for c in ARRIVAL_CATEGORIES]
        other = {"arrival_ambulance": 0, "arrival_self": 0, "arrival_other": 1}
        assert _arrival_label(other).startswith("inny")
        assert _arrival_label({"arrival_mode": "ZRM"}) == "ZRM"


@pytest.fixture(scope="module")
def app() -> ModuleType:
    """Helpery demo Streamlit (import działa bez pakietu streamlit)."""
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from app import streamlit_app

    return streamlit_app


class TestStreamlitHelpers:
    def test_arrival_labels_cover_categories(self, app):
        assert list(app.ARRIVAL_LABELS_PL) == ARRIVAL_CATEGORIES

    def test_trauma_activation_in_curated_complaints(self, app):
        assert TRAUMA_ACTIVATION_COL in app.COMPLAINT_LABELS_PL

    @pytest.mark.parametrize("mode", ARRIVAL_CATEGORIES)
    def test_build_patient_row_one_hot(self, app, mode):
        from datetime import datetime

        row = app.build_patient_row(
            age=70, is_male=True, arrival_mode=mode,
            vitals={"triage_vital_hr": 120.0, "triage_vital_sbp": None},
            o2_device=False, complaints=[TRAUMA_ACTIVATION_COL],
            arrival_time=datetime(2026, 10, 19, 23, 30),
            all_complaints=["cc_chestpain", TRAUMA_ACTIVATION_COL],
        )
        arrival = {f"arrival_{c}": float(c == mode) for c in ARRIVAL_CATEGORIES}
        assert row.iloc[0][list(arrival)].to_dict() == arrival
        assert row.iloc[0][TRAUMA_ACTIVATION_COL] == 1.0
        assert np.isnan(row.iloc[0]["triage_vital_sbp"])

    def test_build_patient_row_rejects_old_categories(self, app):
        from datetime import datetime

        for old in ("police", "unknown"):
            with pytest.raises(ValueError):
                app.build_patient_row(70, True, old, {}, False, [], datetime(2026, 1, 1))

    def test_model_input_has_no_removed_features(self, app):
        from datetime import datetime

        row = app.build_patient_row(50, False, "ambulance", {"triage_vital_hr": 130.0}, False, [],
                                    datetime(2026, 1, 1, 12))
        features = ["tachycardia", "esi_danger_zone", "arrival_ambulance", "esi_danger_hr"]
        X = app.prepare_model_input(row, features)
        assert X.iloc[0]["tachycardia"] == 1.0
        assert X.iloc[0]["arrival_ambulance"] == 1.0
        assert np.isnan(X.iloc[0]["esi_danger_hr"])        # usunięta cecha — tylko NaN z reindex

    def test_level_badge_uses_topsor_hex(self, app):
        for idx, name in enumerate(CLASS_NAMES):
            assert app.level_info(idx)["hex"] == TOPSOR_HEX[name]


# ─────────────────────────────────────────
# 04_evaluate — podgrupy
# ─────────────────────────────────────────
def _toy_frame(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.resize(np.arange(5), n)
    proba = rng.dirichlet(np.ones(5), size=n)
    proba[np.arange(n), y] += 1.0
    proba /= proba.sum(axis=1, keepdims=True)
    df = pd.DataFrame({
        TARGET_COL: y,
        SITE_COL: np.resize(["A", "B", "C"], n),
        VITALS_COMPLETE_COL: (np.arange(n) % 3 != 0).astype(np.int8),
    })
    return df, proba


class TestEvaluateSubgroups:
    def test_vitals_complete_and_sites(self, evaluate_script):
        df, proba = _toy_frame()
        vc_metrics, info = evaluate_script.evaluate_subgroups(df, proba, "val")

        mask = df[VITALS_COMPLETE_COL].to_numpy() == 1
        assert info["vitals_complete"] == {
            "n": int(mask.sum()),
            "n_esi1": int((df[TARGET_COL].to_numpy()[mask] == 0).sum()),
            "n_incomplete": int((~mask).sum()),
        }
        for rule in ("argmax", "cost"):
            assert f"val_vc_{rule}_quadratic_weighted_kappa" in vc_metrics
            assert f"val_vc_{rule}_undertriage_rate" in vc_metrics
        # Metryki vc liczone tylko na podgrupie
        from src.evaluation.metrics import decide, quadratic_weighted_kappa

        y = df[TARGET_COL].to_numpy()
        expected = quadratic_weighted_kappa(y[mask], decide(proba[mask], "argmax"))
        assert vc_metrics["val_vc_argmax_quadratic_weighted_kappa"] == pytest.approx(expected)

        assert set(info["sites"]) == {"A", "B", "C"}
        site_a = info["sites"]["A"]
        assert site_a["n"] == int((df[SITE_COL] == "A").sum())
        assert "cost_critical_miss_rate" in site_a["metrics"]

    def test_missing_columns(self, evaluate_script):
        df, proba = _toy_frame()
        vc_metrics, info = evaluate_script.evaluate_subgroups(
            df.drop(columns=[VITALS_COMPLETE_COL, SITE_COL]), proba, "test"
        )
        assert vc_metrics == {}
        assert info == {"vitals_complete": None, "sites": {}}

    def test_no_complete_rows(self, evaluate_script):
        df, proba = _toy_frame()
        df[VITALS_COMPLETE_COL] = 0
        vc_metrics, info = evaluate_script.evaluate_subgroups(df, proba, "val")
        assert vc_metrics == {}
        assert info["vitals_complete"]["n"] == 0

    def test_summary_prints_subgroups(self, evaluate_script, capsys):
        df, proba = _toy_frame()
        vc_metrics, info = evaluate_script.evaluate_subgroups(df, proba, "val")
        evaluate_script._print_summary(vc_metrics, "val", info)
        out = capsys.readouterr().out
        assert "vc argmax" in out and "oddział" in out
        for site in ("A", "B", "C"):
            assert f"\n{site} " in out


class TestEvaluateScript:
    def test_find_latest_run_skips_samples_and_missing_models(self, evaluate_script, tmp_path):
        runs = tmp_path / "runs"
        for name, sample, has_model in [
            ("20260101_000000_logreg_triage_a", None, True),
            ("20260102_000000_logreg_triage_b", None, False),      # bez model.joblib
            ("20260103_000000_logreg_triage_c", 5000, True),       # run na próbce
        ]:
            (runs / name).mkdir(parents=True)
            info = {"model": "logreg", "feature_set": "triage",
                    "train_sample": sample, "eval_sample": None}
            (runs / name / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
            if has_model:
                (runs / name / "model.joblib").write_bytes(b"")
        run = evaluate_script.find_latest_run("logreg", "triage", include_samples=False, runs_dir=runs)
        assert run.name == "20260101_000000_logreg_triage_a"
        run = evaluate_script.find_latest_run("logreg", "triage", include_samples=True, runs_dir=runs)
        assert run.name == "20260103_000000_logreg_triage_c"

    def test_main_on_model_file(self, prepared_paths, evaluate_script, tmp_path, monkeypatch, capsys):
        """Pełna ewaluacja artefaktu spoza runs/: metryki całości, podgrupy vc i oddziałów w JSON."""
        model = _fitted_logreg(prepared_paths)
        model_path = model.save(tmp_path / "logreg.joblib")
        figures = tmp_path / "figures"
        monkeypatch.setattr(evaluate_script, "FIGURES_DIR", figures)
        monkeypatch.setattr(sys, "argv", ["04_evaluate.py", "--model", str(model_path), "--no-plots"])

        evaluate_script.main()

        files = list(figures.glob("*/metrics_val.json"))
        assert len(files) == 1
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert payload["data_version"] == DATA_VERSION
        metrics = payload["metrics"]
        assert "val_argmax_quadratic_weighted_kappa" in metrics
        assert "val_vc_cost_undertriage_rate" in metrics
        assert payload["subgroups"]["vitals_complete"]["n"] > 0
        assert set(payload["subgroups"]["sites"]) <= {"A", "B", "C"}
        assert payload["subgroups"]["sites"]
        # Kalibracja zawsze, przedziały ufności domyślnie (bootstrap 1000)
        cal = payload["calibration"]["all"]
        assert 0 <= cal["ece_top_label"] <= 1 and "mean" in cal["ece_classwise"]
        assert cal["ece_reference"]["n_sim"] > 0
        assert payload["calibration"]["vc"] is not None
        ci = payload["ci"]
        assert ci["n_boot"] == 1000 and ci["stratified"] is True
        qwk = ci["all"]["cost"]["quadratic_weighted_kappa"]
        assert qwk["ci_low"] <= qwk["value"] <= qwk["ci_high"]
        assert qwk["value"] == pytest.approx(metrics["val_cost_quadratic_weighted_kappa"])
        assert set(ci["vc"]) == {"argmax", "cost"}
        out = capsys.readouterr().out
        assert "vc koszt" in out and "95% przedziały ufności" in out and "Kalibracja (całość)" in out

    def test_main_bootstrap_disabled(self, prepared_paths, evaluate_script, tmp_path, monkeypatch, capsys):
        model = _fitted_logreg(prepared_paths)
        model_path = model.save(tmp_path / "logreg.joblib")
        figures = tmp_path / "figures"
        monkeypatch.setattr(evaluate_script, "FIGURES_DIR", figures)
        monkeypatch.setattr(sys, "argv", ["04_evaluate.py", "--model", str(model_path), "--bootstrap", "0"])

        evaluate_script.main()

        out_dir = next(figures.glob("*"))
        payload = json.loads((out_dir / "metrics_val.json").read_text(encoding="utf-8"))
        assert payload["ci"] is None and payload["calibration"]["all"]["ece_top_label"] >= 0
        assert (out_dir / "reliability_val.png").exists()
        assert "przedziały ufności" not in capsys.readouterr().out

    def test_bootstrap_section_and_table(self, evaluate_script, capsys):
        df, proba = _toy_frame()
        y = df[TARGET_COL].to_numpy()
        vc = df[VITALS_COMPLETE_COL].to_numpy() == 1
        ci = evaluate_script.bootstrap_section(y, proba, vc, n_boot=50)
        cal = evaluate_script.calibration_section(y, proba, vc)
        assert ci["all"]["argmax"]["undertriage_rate"]["n_valid"] == 50
        assert ci["vc"]["cost"]["auc_macro"]["value"] is not None
        assert evaluate_script.bootstrap_section(y, proba, vc, n_boot=0) is None
        evaluate_script.print_ci_table(ci, cal, "val")
        out = capsys.readouterr().out
        assert "vc argmax" in out and "bez reguły" in out and "quadratic_weighted_kappa" in out


# ─────────────────────────────────────────
# Cross-validation — warstwy ESI × oddział
# ─────────────────────────────────────────
class TestCrossValidation:
    def test_make_strata_combines_esi_and_site(self):
        from src.evaluation.cross_validation import make_strata

        y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
        site = np.array(["A", "A", "B", "A", "A", "A", "B", "B", "B"])
        strata = make_strata(y, site, n_splits=2)
        # ESI 1 na oddziale B ma 1 wizytę < 2 foldy → warstwa samego ESI
        assert list(strata) == ["0_A", "0_A", "0", "1_A", "1_A", "1_A", "1_B", "1_B", "1_B"]

    def test_make_strata_without_site(self):
        from src.evaluation.cross_validation import make_strata

        assert list(make_strata(pd.Series([2, 3, 4]))) == ["2", "3", "4"]

    def test_make_strata_missing_site(self):
        from src.evaluation.cross_validation import make_strata

        strata = make_strata([1, 1, 1], pd.Series(["A", None, "A"]), n_splits=1)
        assert list(strata) == ["1_A", "1_?", "1_A"]

    def test_folds_balanced_by_site(self, synthetic_X_y, tmp_path, monkeypatch):
        """CV ze strata: każdy fold ma zbliżony udział oddziałów; trening foldów bez logów treningu."""
        from src.evaluation import cross_validation as cv
        from src.models import get_model

        monkeypatch.setattr(cv, "EVAL_LOGS_DIR", tmp_path / "eval_logs")
        X, y = synthetic_X_y
        site = np.where(np.arange(len(y)) % 4 == 0, "B", "A")
        strata = cv.make_strata(y, site, n_splits=2)

        seen: list[np.ndarray] = []
        original_split = cv.StratifiedKFold.split

        def spy(self, X_, y_=None, groups=None):
            for train_idx, val_idx in original_split(self, X_, y_, groups):
                seen.append(val_idx)
                yield train_idx, val_idx

        monkeypatch.setattr(cv.StratifiedKFold, "split", spy)
        result = cv.cross_validate_model(
            lambda: get_model("logreg", params={"max_iter": 200}),
            X, y, n_splits=2, strata=strata, model_name="logreg",
            save_path=tmp_path / "cv_logreg.csv",
        )
        assert len(result) == 2
        assert {"best_iteration", "device", "argmax_quadratic_weighted_kappa"} <= set(result.columns)
        for val_idx in seen:
            assert np.mean(site[val_idx] == "B") == pytest.approx(0.25, abs=0.02)
        assert not list(Path(config.TRAINING_LOGS_DIR).glob("**/*.log"))
        assert len(list(tmp_path.glob("cv_logreg_*_summary.csv"))) == 1

    def test_strata_length_mismatch(self, synthetic_X_y, tmp_path, monkeypatch):
        from src.evaluation import cross_validation as cv
        from src.models import get_model

        monkeypatch.setattr(cv, "EVAL_LOGS_DIR", tmp_path / "eval_logs")
        X, y = synthetic_X_y
        with pytest.raises(ValueError):
            cv.cross_validate_model(lambda: get_model("logreg"), X, y, n_splits=2, strata=np.zeros(3),
                                    model_name="logreg")

    def test_cross_validate_script_uses_site_strata(self, prepared_paths, tmp_path, monkeypatch, capsys):
        """06_cross_validate: foldy po ESI × oddział z kolumny `site` splitu TRAIN, wyniki w CSV."""
        from src.evaluation import cross_validation as cv

        script = _load_script("06_cross_validate")
        monkeypatch.setattr(script, "TABLES_DIR", tmp_path / "tables")
        monkeypatch.setattr(cv, "EVAL_LOGS_DIR", tmp_path / "eval_logs")
        captured: dict = {}
        original = script.make_strata

        def spy(y, site=None, n_splits=5):
            captured["site"] = site
            captured["strata"] = original(y, site, n_splits=n_splits)
            return captured["strata"]

        monkeypatch.setattr(script, "make_strata", spy)
        monkeypatch.setattr(sys, "argv", ["06_cross_validate.py", "--model", "logreg", "--k", "2"])

        script.main()

        assert captured["site"] is not None
        assert any("_" in s for s in captured["strata"])          # warstwy ESI × oddział
        assert len(list((tmp_path / "tables").glob("cv_logreg_triage_*_per_fold.csv"))) == 1
        assert "CV logreg / triage" in capsys.readouterr().out

    def test_config_from_run_reads_weights_strategy(self, tmp_path):
        script = _load_script("06_cross_validate")
        run_dir = tmp_path / "20260101_000000_xgboost_triage_standard"
        run_dir.mkdir()
        info = {
            "model": "xgboost", "feature_set": "triage",
            "params": {"max_depth": 6, "device": "cuda"},
            "class_weights": {"0": 4.0, "1": 2.0},
            "weights_strategy": "optuna_qwk",
        }
        (run_dir / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
        cfg = script._config_from_run(str(run_dir))
        assert cfg["params"] == {"max_depth": 6}               # parametry sprzętowe pominięte
        assert cfg["class_weights"] == {0: 4.0, 1: 2.0}
        assert cfg["weights"] is None                           # wagi z Optuny, nie strategia

        info.pop("weights_strategy")
        info["sample_weight_strategy"] = "balanced"
        (run_dir / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
        assert script._config_from_run(str(run_dir))["weights"] == "balanced"


# ─────────────────────────────────────────
# Uncertainty — wybór runów z runs/ (runy bez tuningu)
# ─────────────────────────────────────────
class TestUncertaintyRunSelection:
    @staticmethod
    def _untuned_run(runs_dir: Path, name: str, model: str, seed: int) -> None:
        """Run jak z `03_train.py` bez --tune (tuned=False): run_info, predykcje val, model, DONE.json."""
        from src.ensemble.runs import PROBA_COLS

        rng = np.random.default_rng(seed)
        y = np.arange(200) % len(CLASS_NAMES)
        proba = rng.dirichlet(np.ones(len(CLASS_NAMES)), size=len(y))
        run_dir = runs_dir / name
        run_dir.mkdir(parents=True)
        info = {"run_name": name, "model": model, "feature_set": "triage", "tuned": False,
                "tag": "standard-notune", "data_version": DATA_VERSION, "train_sample": None, "eval_sample": None}
        (run_dir / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
        frame = pd.DataFrame({"row_id": np.arange(len(y)), "target": y})
        for i, col in enumerate(PROBA_COLS):
            frame[col] = proba[:, i]
        frame.to_parquet(run_dir / "predictions_val.parquet", index=False)
        (run_dir / "model.joblib").write_bytes(b"fake")
        (run_dir / "DONE.json").write_text(json.dumps({"run_name": name}), encoding="utf-8")

    def test_untuned_runs_need_flag(self, tmp_path, monkeypatch):
        from src.evaluation import uncertainty

        runs_dir = tmp_path / "runs"
        self._untuned_run(runs_dir, "20260101_000000_logreg_triage_standard-notune", "logreg", 1)
        self._untuned_run(runs_dir, "20260101_000100_lightgbm_triage_standard-notune", "lightgbm", 2)
        monkeypatch.setattr(config, "RUNS_DIR", runs_dir)

        monkeypatch.setattr(sys, "argv", ["uncertainty"])
        with pytest.raises(SystemExit, match="--include-untuned"):
            uncertainty.main()

        report = tmp_path / "uncertainty_val.txt"
        monkeypatch.setattr(sys, "argv", ["uncertainty", "--include-untuned", "--output", str(report)])
        uncertainty.main()
        text = report.read_text(encoding="utf-8")
        assert "logreg" in text and "lightgbm" in text


# ─────────────────────────────────────────
# 05_explain — reguły ESI + SHAP na syntetycznym splicie
# ─────────────────────────────────────────
def test_explain_script_end_to_end(prepared_paths, tmp_path, monkeypatch):
    script = _load_script("05_explain")
    model = _fitted_logreg(prepared_paths)
    model_path = model.save(tmp_path / "logreg.joblib")
    out_dir = tmp_path / "explanations"
    monkeypatch.setattr(sys, "argv", [
        "05_explain.py", "--model", str(model_path), "--n-patients", "2",
        "--summary-size", "0", "--background", "50", "--out-dir", str(out_dir),
    ])

    script.main()

    payload = json.loads((out_dir / "explanations.json").read_text(encoding="utf-8"))
    assert payload["model"] == "logreg"
    assert len(payload["patients"]) == 2
    for patient in payload["patients"]:
        assert patient[VITALS_COMPLETE_COL] in (0, 1)
        assert patient["rule_check"]["label_pl"]
        assert set(patient["consistency"]) == {"argmax", "cost"}
        assert patient["true"]["topsor_color"] in config.TOPSOR_COLORS
