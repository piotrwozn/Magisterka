"""
Testy narzędzi do ręcznego łączenia modeli (`src.ensemble`) i porównania runów (07_compare_runs.py).

Runy są fałszywe (run_info.json + predictions_{val,test}.parquet + DONE.json + model.joblib
w tmp_path), w formacie zapisywanym przez `src.models.train.run_training`.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ensemble import (
    aligned_probas,
    apply_temperature,
    blend,
    diversity,
    evaluate,
    fit_temperature,
    latest_runs,
    list_runs,
    load_run,
    optimize_weights,
)
from src.ensemble.runs import PROBA_COLS
from src.evaluation.metrics import decide
from src.utils import config
from src.utils.config import CLASS_NAMES, DATA_VERSION, N_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[1]

N_ROWS = 300


def _true_labels(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(N_CLASSES, size=N_ROWS, p=[0.05, 0.3, 0.4, 0.2, 0.05])


def _make_proba(y: np.ndarray, quality: float, seed: int) -> np.ndarray:
    """Prawdopodobieństwa skupione wokół prawdziwej klasy (quality ↑ → model lepszy)."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1, size=(len(y), N_CLASSES))
    logits[np.arange(len(y)), y] += quality
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def vitals_complete_flags(n: int = N_ROWS) -> np.ndarray:
    """Co trzeci wiersz bez kompletu parametrów życiowych."""
    return (np.arange(n) % 3 != 0).astype(np.int8)


def write_run(
    runs_dir: Path,
    name: str,
    model: str,
    probas: dict[str, np.ndarray],
    y: np.ndarray,
    feature_set: str = "triage",
    train_sample: int | None = None,
    shuffle: bool = False,
    row_ids: np.ndarray | None = None,
    *,
    tag: str | None = "standard",
    weights: str = "none",
    data_version: str = DATA_VERSION,
    done: bool = True,
    with_model: bool = True,
    vitals: bool = True,
) -> Path:
    """Katalog runa w formacie `run_training` (run_info + predykcje val/test + DONE.json + model)."""
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True)
    info = {
        "run_name": name, "model": model, "feature_set": feature_set, "tuned": False,
        "tag": tag, "profile": tag, "sample_weight_strategy": weights, "weights_strategy": weights,
        "data_version": data_version, "data_fingerprint": "abc123", "n_fit": 1000, "n_val": len(y),
        "train_sample": train_sample, "eval_sample": None, "note": "zażółć gęślą jaźń",
        "class_names": CLASS_NAMES, "proba_columns": PROBA_COLS,
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    ids = np.arange(len(y)) if row_ids is None else row_ids
    for split, proba in probas.items():
        df = pd.DataFrame({"row_id": ids, "site": "A", "esi": y + 1, "target": y})
        if vitals:
            df["vitals_complete"] = vitals_complete_flags(len(y))
        for i, col in enumerate(PROBA_COLS):
            df[col] = proba[:, i].astype(np.float32)
        df["pred_argmax"] = decide(proba, "argmax")
        df["pred_cost"] = decide(proba, "cost")
        if shuffle:
            df = df.sample(frac=1.0, random_state=0)
        df.to_parquet(run_dir / f"predictions_{split}.parquet", index=False)
    if with_model:
        (run_dir / "model.joblib").write_bytes(b"fake")
    if done:
        (run_dir / "DONE.json").write_text(json.dumps({"run_name": name, "model": model}), encoding="utf-8")
    return run_dir


@pytest.fixture
def runs_dir(tmp_path):
    """Trzy modele + starszy run LightGBM, run na próbce i run z zestawem 'full'."""
    y = _true_labels()
    root = tmp_path / "runs"
    good = {s: _make_proba(y, 3.0, seed=1 + i) for i, s in enumerate(["val", "test"])}
    weak = {s: _make_proba(y, 1.0, seed=11 + i) for i, s in enumerate(["val", "test"])}
    noise = {s: _make_proba(y, 0.0, seed=21 + i) for i, s in enumerate(["val", "test"])}
    write_run(root, "20260101_000000_lightgbm_triage", "lightgbm", noise, y)        # starszy
    write_run(root, "20260102_000000_lightgbm_triage", "lightgbm", good, y, shuffle=True)
    write_run(root, "20260102_000100_xgboost_triage", "xgboost", weak, y)
    write_run(root, "20260102_000200_logreg_triage_sample", "logreg", noise, y, train_sample=5000)
    write_run(root, "20260102_000300_catboost_full", "catboost", good, y, feature_set="full")
    return root


# ─────────────────────────────────────────
# Runy
# ─────────────────────────────────────────
class TestRuns:
    def test_list_runs(self, runs_dir):
        table = list_runs(runs_dir)
        assert len(table) == 5
        assert table["run"].is_monotonic_increasing      # najnowsze na końcu

    def test_latest_runs_picks_newest_per_model(self, runs_dir):
        runs = latest_runs("triage", runs_dir=runs_dir)
        names = {r.model: r.name for r in runs}
        assert set(names) == {"lightgbm", "xgboost"}     # bez próbek i bez 'full'
        assert names["lightgbm"] == "20260102_000000_lightgbm_triage"

    def test_latest_runs_include_samples(self, runs_dir):
        runs = latest_runs("triage", include_samples=True, runs_dir=runs_dir)
        assert {r.model for r in runs} == {"lightgbm", "xgboost", "logreg"}

    def test_latest_runs_filters(self, runs_dir):
        assert [r.model for r in latest_runs("full", runs_dir=runs_dir)] == ["catboost"]
        runs = latest_runs("triage", models=["xgboost"], runs_dir=runs_dir)
        assert [r.model for r in runs] == ["xgboost"]

    def test_latest_runs_empty_dir(self, tmp_path):
        assert latest_runs(runs_dir=tmp_path) == []

    def test_load_run_by_name_and_path(self, runs_dir):
        by_name = load_run("20260102_000100_xgboost_triage", runs_dir=runs_dir)
        by_path = load_run(runs_dir / "20260102_000100_xgboost_triage")
        assert by_name.model == by_path.model == "xgboost"
        assert by_name.info["feature_set"] == "triage"

    def test_load_run_not_a_run(self, tmp_path):
        (tmp_path / "pusty").mkdir()
        with pytest.raises(FileNotFoundError, match="run_info.json"):
            load_run(tmp_path / "pusty")

    def test_missing_predictions(self, runs_dir):
        run = load_run(runs_dir / "20260102_000100_xgboost_triage")
        with pytest.raises(FileNotFoundError):
            run.predictions("holdout")

    def test_default_runs_dir_redirected_in_tests(self, isolated_paths, runs_dir):
        """`isolated_paths` przekierowuje też domyślne argumenty (nie czytamy prawdziwego runs/)."""
        assert isolated_paths.RUNS_DIR == runs_dir
        assert len(list_runs()) == 5
        assert [r.name for r in latest_runs()] == [r.name for r in latest_runs(runs_dir=runs_dir)]
        assert load_run("20260102_000100_xgboost_triage").path == runs_dir / "20260102_000100_xgboost_triage"

    def test_default_runs_dir_resolved_at_call_time(self, runs_dir, monkeypatch):
        """Domyślny katalog czytany przy wywołaniu z config.RUNS_DIR (np. SOR_AI_RUNS_DIR), nie przy imporcie."""
        monkeypatch.setattr(config, "RUNS_DIR", runs_dir)
        assert len(list_runs()) == 5
        assert {r.model for r in latest_runs()} == {"lightgbm", "xgboost"}

    def test_list_runs_columns(self, runs_dir):
        table = list_runs(runs_dir)
        for col in ("run", "model", "feature_set", "tag", "weights", "data_version", "done", "has_model",
                    "train_sample", "eval_sample"):
            assert col in table.columns
        assert table["done"].all() and table["has_model"].all()

    def test_latest_runs_skips_unfinished_and_modelless(self, runs_dir):
        y = _true_labels()
        p = {"val": _make_proba(y, 3.0, 50), "test": _make_proba(y, 3.0, 51)}
        write_run(runs_dir, "20260105_000000_lightgbm_triage", "lightgbm", p, y, done=False)
        write_run(runs_dir, "20260105_000100_xgboost_triage", "xgboost", p, y, with_model=False)
        names = {r.model: r.name for r in latest_runs("triage", runs_dir=runs_dir)}
        assert names == {"lightgbm": "20260102_000000_lightgbm_triage", "xgboost": "20260102_000100_xgboost_triage"}
        loose = {r.model: r.name for r in latest_runs("triage", runs_dir=runs_dir, require_done=False,
                                                      require_model=False)}
        assert loose == {"lightgbm": "20260105_000000_lightgbm_triage", "xgboost": "20260105_000100_xgboost_triage"}

    def test_latest_runs_tag_filter(self, runs_dir):
        y = _true_labels()
        p = {"val": _make_proba(y, 2.0, 60)}
        write_run(runs_dir, "20260106_000000_lightgbm_triage_smoke", "lightgbm", p, y, tag="smoke")
        assert {r.name for r in latest_runs("triage", runs_dir=runs_dir, tag="smoke")} == {
            "20260106_000000_lightgbm_triage_smoke"}
        standard = {r.model: r.name for r in latest_runs("triage", runs_dir=runs_dir, tag="standard")}
        assert standard["lightgbm"] == "20260102_000000_lightgbm_triage"

    def test_latest_runs_excludes_weighted_with_note(self, runs_dir, caplog):
        y = _true_labels()
        p = {"val": _make_proba(y, 2.0, 70)}
        write_run(runs_dir, "20260107_000000_xgboost_triage_bal", "xgboost", p, y, weights="balanced")
        with caplog.at_level("INFO"):
            names = {r.model: r.name for r in latest_runs("triage", runs_dir=runs_dir)}
        assert names["xgboost"] == "20260102_000100_xgboost_triage"     # run z wagami pominięty
        weighted = {r.model: r.name for r in latest_runs("triage", runs_dir=runs_dir, include_weighted=True)}
        assert weighted["xgboost"] == "20260107_000000_xgboost_triage_bal"

    def test_latest_runs_other_data_version_skipped(self, runs_dir):
        y = _true_labels()
        p = {"val": _make_proba(y, 2.0, 80)}
        write_run(runs_dir, "20260108_000000_xgboost_triage", "xgboost", p, y, data_version="esi-v1")
        assert {r.model: r.name for r in latest_runs("triage", runs_dir=runs_dir)}["xgboost"] == \
            "20260102_000100_xgboost_triage"
        any_version = latest_runs("triage", runs_dir=runs_dir, data_version=None)
        assert {r.model: r.name for r in any_version}["xgboost"] == "20260108_000000_xgboost_triage"

    def test_feature_set_alias(self, runs_dir):
        assert {r.model for r in latest_runs("triage_only", runs_dir=runs_dir)} == {"lightgbm", "xgboost"}

    def test_run_info_read_as_utf8(self, runs_dir):
        run = load_run("20260102_000100_xgboost_triage", runs_dir=runs_dir)
        assert run.info["note"] == "zażółć gęślą jaźń"
        assert run.done and run.has_model


# ─────────────────────────────────────────
# Wyrównanie predykcji
# ─────────────────────────────────────────
class TestAlignedProbas:
    def test_aligned_by_row_id(self, runs_dir):
        runs = latest_runs("triage", runs_dir=runs_dir)
        y, probas = aligned_probas(runs, "val")
        assert y.shape == (N_ROWS,)
        np.testing.assert_array_equal(y, _true_labels())   # mimo przetasowanego parquet
        assert set(probas) == {"lightgbm", "xgboost"}
        for p in probas.values():
            assert p.shape == (N_ROWS, N_CLASSES)
            np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)

    def test_values_follow_row_ids(self, runs_dir):
        runs = latest_runs("triage", runs_dir=runs_dir)
        _, probas = aligned_probas(runs, "test")
        raw = runs[0].predictions("test").set_index("row_id").sort_index()
        key = runs[0].model
        np.testing.assert_allclose(probas[key], raw[PROBA_COLS].to_numpy(), atol=1e-6)

    def test_duplicate_models_keyed_by_run_name(self, runs_dir):
        runs = [load_run(p, runs_dir) for p in ("20260101_000000_lightgbm_triage",
                                                "20260102_000000_lightgbm_triage")]
        _, probas = aligned_probas(runs, "val")
        assert set(probas) == {r.name for r in runs}

    def test_mismatched_row_sets_raise(self, runs_dir):
        y = _true_labels()
        p = {"val": _make_proba(y, 2.0, seed=99)}
        write_run(runs_dir, "20260103_000000_ebm_triage", "ebm", p, y, row_ids=np.arange(N_ROWS) + 1000)
        runs = latest_runs("triage", runs_dir=runs_dir)
        with pytest.raises(ValueError, match="różne zbiory wierszy"):
            aligned_probas(runs, "val")

    def test_no_runs_raise(self):
        with pytest.raises(ValueError):
            aligned_probas([], "val")

    def test_return_vc_mask(self, runs_dir):
        runs = latest_runs("triage", runs_dir=runs_dir)
        y, probas, vc = aligned_probas(runs, "val", return_vc=True)
        np.testing.assert_array_equal(vc, vitals_complete_flags() == 1)   # po sortowaniu wg row_id
        assert vc.dtype == bool and len(vc) == len(y)

    def test_return_vc_none_without_column(self, tmp_path):
        y = _true_labels()
        write_run(tmp_path, "20260101_000000_ebm_triage", "ebm", {"val": _make_proba(y, 1.0, 3)}, y, vitals=False)
        _, _, vc = aligned_probas([load_run(tmp_path / "20260101_000000_ebm_triage")], "val", return_vc=True)
        assert vc is None

    def test_different_targets_raise(self, runs_dir):
        y = _true_labels()
        other = y.copy()
        other[:5] = (other[:5] + 1) % N_CLASSES
        write_run(runs_dir, "20260109_000000_ebm_triage", "ebm", {"val": _make_proba(other, 2.0, 9)}, other)
        runs = latest_runs("triage", runs_dir=runs_dir)
        with pytest.raises(ValueError, match="inne etykiety"):
            aligned_probas(runs, "val")

    def test_different_data_versions_raise(self, runs_dir):
        y = _true_labels()
        old = write_run(runs_dir, "20260109_000100_ebm_triage", "ebm", {"val": _make_proba(y, 2.0, 9)}, y,
                        data_version="esi-v1")
        runs = latest_runs("triage", runs_dir=runs_dir) + [load_run(old)]
        with pytest.raises(ValueError, match="wersji danych"):
            aligned_probas(runs, "val")


# ─────────────────────────────────────────
# Łączenie
# ─────────────────────────────────────────
class TestBlend:
    @pytest.fixture
    def probas(self):
        y = _true_labels()
        return {"a": _make_proba(y, 2.0, 1), "b": _make_proba(y, 1.0, 2), "c": _make_proba(y, 0.5, 3)}

    def test_equal_weights_is_mean(self, probas):
        out = blend(probas)
        expected = (probas["a"] + probas["b"] + probas["c"]) / 3
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_weights_normalized(self, probas):
        np.testing.assert_allclose(
            blend(probas, {"a": 2, "b": 2, "c": 4}),
            blend(probas, {"a": 0.25, "b": 0.25, "c": 0.5}),
        )

    def test_single_model_weight(self, probas):
        np.testing.assert_allclose(blend(probas, {"a": 1.0}), probas["a"], atol=1e-12)

    def test_geometric(self, probas):
        out = blend(probas, {"a": 1, "b": 1}, method="geometric")
        assert out.shape == probas["a"].shape
        np.testing.assert_allclose(out.sum(axis=1), 1.0)
        geo = np.sqrt(probas["a"] * probas["b"])
        np.testing.assert_allclose(out, geo / geo.sum(axis=1, keepdims=True), atol=1e-12)

    def test_geometric_identical_inputs(self, probas):
        same = {"x": probas["a"], "y": probas["a"]}
        np.testing.assert_allclose(blend(same, method="geometric"), probas["a"], atol=1e-12)

    def test_rows_sum_to_one(self, probas):
        np.testing.assert_allclose(blend(probas, {"a": 0.7, "b": 0.2, "c": 0.1}).sum(axis=1), 1.0)

    @pytest.mark.parametrize("weights", [{"a": -1.0, "b": 2.0}, {"a": 0.0, "b": 0.0, "c": 0.0}])
    def test_invalid_weights(self, probas, weights):
        with pytest.raises(ValueError, match="Wagi"):
            blend(probas, weights)

    def test_unknown_method(self, probas):
        with pytest.raises(ValueError, match="Nieznana metoda"):
            blend(probas, method="median")

    def test_evaluate(self, probas):
        y = _true_labels()
        res_argmax = evaluate(y, probas["a"], "argmax")
        res_cost = evaluate(y, probas["a"], "cost")
        assert res_argmax["quadratic_weighted_kappa"] > 0.3
        assert "auc_esi1" in res_cost
        assert res_argmax["auc_macro"] == pytest.approx(res_cost["auc_macro"])   # AUC nie zależy od reguły


class TestOptimizeWeights:
    def test_weights_sum_to_one_and_prefer_better_model(self):
        y = _true_labels()
        probas = {"good": _make_proba(y, 3.0, 1), "noise": _make_proba(y, 0.0, 2)}
        weights = optimize_weights(probas, y)
        assert set(weights) == {"good", "noise"}
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
        assert all(w >= 0 for w in weights.values())
        assert weights["good"] > weights["noise"]

    def test_geometric_method(self):
        y = _true_labels()
        probas = {"a": _make_proba(y, 2.0, 1), "b": _make_proba(y, 1.0, 2)}
        weights = optimize_weights(probas, y, method="geometric")
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)


# ─────────────────────────────────────────
# Kalibracja
# ─────────────────────────────────────────
class TestTemperature:
    def test_overconfident_gets_t_above_one(self):
        y = _true_labels()
        calibrated = _make_proba(y, 1.5, 5)
        overconfident = apply_temperature(calibrated, 0.3)       # wyostrzone
        t = fit_temperature(overconfident, y)
        assert t > 1.0

    def test_underconfident_gets_t_below_one(self):
        y = _true_labels()
        flat = apply_temperature(_make_proba(y, 2.0, 6), 4.0)   # spłaszczone
        assert fit_temperature(flat, y) < 1.0

    def test_apply_identity_and_normalization(self):
        p = _make_proba(_true_labels(), 1.0, 7)
        np.testing.assert_allclose(apply_temperature(p, 1.0), p, atol=1e-10)
        hot = apply_temperature(p, 1000.0)
        np.testing.assert_allclose(hot, 1.0 / N_CLASSES, atol=1e-2)
        np.testing.assert_allclose(apply_temperature(p, 0.5).sum(axis=1), 1.0)

    def test_temperature_preserves_argmax(self):
        p = _make_proba(_true_labels(), 1.0, 8)
        np.testing.assert_array_equal(apply_temperature(p, 2.5).argmax(axis=1), p.argmax(axis=1))


# ─────────────────────────────────────────
# Różnorodność
# ─────────────────────────────────────────
class TestDiversity:
    def test_columns_and_pairs(self):
        y = _true_labels()
        probas = {"a": _make_proba(y, 2.0, 1), "b": _make_proba(y, 1.0, 2), "c": _make_proba(y, 0.5, 3)}
        df = diversity(probas, y)
        assert list(df.columns) == ["model_a", "model_b", "disagreement", "corr_expected", "corr_errors"]
        assert len(df) == 3
        assert df["disagreement"].between(0, 1).all()
        assert df["corr_expected"].between(-1, 1).all()

    def test_identical_models(self):
        y = _true_labels()
        p = _make_proba(y, 1.0, 1)
        row = diversity({"a": p, "b": p.copy()}, y).iloc[0]
        assert row["disagreement"] == 0.0
        assert row["corr_expected"] == pytest.approx(1.0)
        assert row["corr_errors"] == pytest.approx(1.0)

    def test_single_model_empty_table(self):
        y = _true_labels()
        df = diversity({"a": _make_proba(y, 1.0, 1)}, y)
        assert df.empty
        assert list(df.columns) == ["model_a", "model_b", "disagreement", "corr_expected", "corr_errors"]
        assert df.sort_values("corr_errors").empty                      # 07_compare_runs sortuje


# ─────────────────────────────────────────
# scripts/07_compare_runs.py
# ─────────────────────────────────────────
def _load_compare_script():
    spec = importlib.util.spec_from_file_location("compare_runs_script", PROJECT_ROOT / "scripts" / "07_compare_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCompareScript:
    @pytest.fixture
    def compare_runs(self, isolated_paths):
        y = _true_labels()
        root = isolated_paths.RUNS_DIR
        for i, (name, model) in enumerate([("20260102_000000_lightgbm_triage_standard", "lightgbm"),
                                           ("20260102_000100_xgboost_triage_standard", "xgboost")]):
            write_run(root, name, model, {s: _make_proba(y, 2.0 + i, 10 * i + j) for j, s in
                                          enumerate(["val", "test"])}, y)
        return root

    def test_report_val_only_with_vc_columns(self, compare_runs, capsys):
        module = _load_compare_script()
        out = compare_runs / "cmp.md"
        assert module.main(["--feature-set", "triage", "--tag", "standard", "--out", str(out)]) == 0
        text = out.read_text(encoding="utf-8")
        assert "QWK vc [argmax]" in text and "undertriage vc [cost]" in text
        assert "Różnorodność na VAL" in text
        assert "| test |" not in text                         # zbiór testowy nie jest pokazywany
        assert "lightgbm" in text and "xgboost" in text
        assert (compare_runs / "leaderboard.csv").exists()    # przebudowany przy porównaniu

    def test_explicit_runs_single_model(self, compare_runs):
        module = _load_compare_script()
        out = compare_runs / "one.md"
        assert module.main(["--runs", "20260102_000000_lightgbm_triage_standard", "--out", str(out)]) == 0
        text = out.read_text(encoding="utf-8")
        assert "Różnorodność" not in text                     # jeden model — bez tabeli par
        assert "średnia równowagowa" in text

    def test_no_runs_exit_1(self, isolated_paths):
        module = _load_compare_script()
        assert module.main(["--feature-set", "triage"]) == 1

    def test_metrics_row_vc_keys(self):
        module = _load_compare_script()
        y = _true_labels()
        p = _make_proba(y, 2.0, 1)
        row = module.metrics_row("m", y, p, vitals_complete_flags() == 1)
        assert {"QWK [argmax]", "QWK [cost]", "QWK vc [argmax]", "undertriage vc [cost]"} <= set(row)
        assert "QWK vc [argmax]" not in module.metrics_row("m", y, p, None)
