"""
Testy analiz do pracy (src/analysis, scripts/09–11) i refitu `src.models.train.refit_from_run`.

Runy źródłowe: `run_training` na syntetycznym Yale (fixture `prepared_paths`, katalog tymczasowy)
z profilem 'smoke' i tagiem 'smoke'; flaga `tuned` dopisywana do run_info.json / DONE.json
(analizy biorą tylko runy z tuningiem, a prawdziwy tuning trwałby za długo).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import common, learning_curve, priors, seeds, site_out
from src.analysis.common import NoSourceRuns, UnitStore, analyses_status, execute_units, find_refit_run
from src.models import train
from src.models.train import PROBA_COLS, REFIT_VERSION, refit_from_run, refit_params, run_training
from src.utils import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAG = "smoke"


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mark_tuned(run_dir: Path) -> None:
    for name in ("run_info.json", "DONE.json"):
        data = _read(run_dir / name)
        data["tuned"] = True
        (run_dir / name).write_text(json.dumps(data), encoding="utf-8")


def make_source(model: str) -> Path:
    """Zwykły run profilu smoke (nie custom) z udawanym tuningiem."""
    result = run_training(model, profile="smoke", tag=TAG)
    run_dir = Path(result["run_dir"])
    assert _read(run_dir / "DONE.json")["custom"] is False
    _mark_tuned(run_dir)
    return run_dir


def refit_dirs(runs_dir: Path) -> list[Path]:
    return sorted(p.parent for p in runs_dir.glob("*/DONE.json") if _read(p).get("refit_of"))


@pytest.fixture
def lgbm_source(prepared_paths) -> Path:
    return make_source("lightgbm")


# ─────────────────────────────────────────
# Korekta rozkładu klas (EM, Saerens 2002)
# ─────────────────────────────────────────
class TestPriors:
    def test_class_prior_and_normalize(self):
        np.testing.assert_allclose(priors.class_prior(np.array([0, 1, 1, 4])), [0.25, 0.5, 0, 0, 0.25])
        with pytest.raises(ValueError):
            priors.normalize_prior([0, 0, 0])
        with pytest.raises(ValueError):
            priors.normalize_prior([0.5, -0.1, 0.6])
        with pytest.raises(ValueError):
            priors.class_prior(np.array([], dtype=int))

    def test_shift_identity_rows_and_absent_class(self):
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(5), size=50)
        prior = np.array([0.1, 0.2, 0.3, 0.3, 0.1])
        np.testing.assert_allclose(priors.apply_prior_shift(p, prior, prior), p)
        shifted = priors.apply_prior_shift(p, [0.3, 0.3, 0.2, 0.1, 0.1], prior)
        np.testing.assert_allclose(shifted.sum(axis=1), 1.0)
        assert (shifted[:, 0] > p[:, 0]).all()                      # ESI 1 częstsze → wyższe P(ESI 1)
        no_esi1 = priors.apply_prior_shift(p, [0.2, 0.2, 0.2, 0.2, 0.2], [0.0, 0.25, 0.25, 0.25, 0.25])
        assert (no_esi1[:, 0] == 0).all()

    def test_em_recovers_prior_and_known_prior_helps(self):
        """Dokładne posteriory przy π_trening; nowa populacja z innym π — EM odzyskuje π bez etykiet."""
        from scipy.stats import norm

        rng = np.random.default_rng(1)
        train_prior = np.array([0.05, 0.2, 0.4, 0.25, 0.1])
        new_prior = np.array([0.2, 0.3, 0.3, 0.15, 0.05])
        y = rng.choice(5, size=20_000, p=new_prior)
        x = rng.normal(loc=y.astype(float), scale=1.0)
        lik = norm.pdf(x[:, None], loc=np.arange(5)[None, :], scale=1.0)
        post = lik * train_prior
        post /= post.sum(axis=1, keepdims=True)

        est, info = priors.em_prior(post, train_prior)
        assert info["converged"] and info["iterations"] > 1 and not info["suspect"]
        assert info["max_ratio"] == pytest.approx(4.0, rel=0.2)                    # ESI 1: 0,20 / 0,05
        # Model bez informacji (stałe prawdopodobieństwa) — EM nie rusza się z π_trening
        flat, flat_info = priors.em_prior(np.tile(train_prior, (100, 1)), train_prior)
        np.testing.assert_allclose(flat, train_prior)
        assert not flat_info["suspect"]
        np.testing.assert_allclose(est, new_prior, atol=0.03)

        def logloss(p: np.ndarray) -> float:
            return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean())

        assert logloss(priors.apply_prior_shift(post, new_prior, train_prior)) < logloss(post) - 0.02
        assert logloss(priors.apply_prior_shift(post, est, train_prior)) < logloss(post) - 0.02


# ─────────────────────────────────────────
# Wspólne: jednostki, ETA, metryki
# ─────────────────────────────────────────
class TestCommon:
    def test_execute_units_isolates_failures_and_resumes(self, tmp_path):
        store = UnitStore(tmp_path / "a")
        units = [{"unit": f"m/{k}", "k": k} for k in ("a", "b", "c")]
        calls: list[str] = []

        def compute(u):
            calls.append(u["unit"])
            if u["k"] == "b" and len(calls) < 4:
                raise RuntimeError("awaria jednostki")
            return {"value": u["k"]}

        def expect(u):
            return {"source_run": "src", "metrics_version": 1}

        failures = execute_units(store, units, expect, compute, "test")
        assert [f["unit"] for f in failures] == ["m/b"] and "awaria" in failures[0]["error"]
        assert store.valid("m/a", expect({})) and store.valid("m/c", expect({}))
        assert store.valid("m/b", expect({})) is None
        assert store.path("m/a").name == "m__a.json"
        # Wznowienie: tylko nieudana jednostka
        assert execute_units(store, units, expect, compute, "test") == []
        assert calls == ["m/a", "m/b", "m/c", "m/b"]
        # Inny run źródłowy → jednostki nieważne; --force liczy wszystko
        assert store.valid("m/a", {"source_run": "inny"}) is None
        execute_units(store, units, expect, compute, "test", force=True)
        assert calls[-3:] == ["m/a", "m/b", "m/c"]

    def test_evaluate_rules_vc_and_admission(self):
        y = np.array([0, 1, 2, 3, 4, 1, 2, 0])
        p = np.eye(5)[np.array([0, 1, 2, 3, 4, 2, 2, 3])] * 0.9 + 0.02
        vc = np.array([1, 1, 1, 1, 0, 1, 0, 1], dtype=bool)
        adm = np.array([1, 1, 0, 0, 0, 1, np.nan, 1])
        out = common.evaluate(y, p, {"argmax": p.argmax(1)}, vc, adm)
        assert out["n"] == 8 and out["vc_n"] == 6 and out["n_esi1"] == 2
        assert out["undertriage_argmax"] == pytest.approx(2 / 4)       # ESI1→ESI4, ESI2→ESI3
        assert out["critical_miss_argmax"] == pytest.approx(1 / 2)
        assert out["recall_esi1_argmax"] == pytest.approx(1 / 2)
        assert 0 < out["log_loss"] < 3 and 0.5 <= out["admission_auc"] <= 1
        assert "vc_qwk_argmax" in out and "vc_log_loss" in out

    def test_merge_units_cumulative(self):
        prev = [{"unit": "logreg/seed1", "model": "logreg", "source_run": "r_lr"},
                {"unit": "lightgbm/seed1", "model": "lightgbm", "source_run": "r_lgbm"},
                {"unit": "lightgbm/seed2", "model": "lightgbm", "source_run": "r_lgbm"},
                {"unit": "xgboost/seed1", "model": "xgboost", "source_run": "r_xgb_old"}]
        cur = [{"unit": "lightgbm/seed1", "model": "lightgbm", "source_run": "r_lgbm"}]
        merged = common.merge_units(prev, cur, lambda u: u["model"],
                                    {"logreg": "r_lr", "lightgbm": "r_lgbm", "xgboost": "r_xgb_new"})
        # lightgbm liczony teraz (1 seed) zastępuje swoje 2 stare; xgboost ze starego runa odpada
        assert [u["unit"] for u in merged] == ["logreg/seed1", "lightgbm/seed1"]
        store = UnitStore(Path("/nonexistent"))
        assert store.planned_units() == []

    def test_mean_sd_and_clean_json(self):
        s = common.mean_sd([0.5, 0.7, None, float("nan")])
        assert s["mean"] == pytest.approx(0.6) and s["n"] == 2 and s["values"] == [0.5, 0.7, None, None]
        assert s["sd"] == pytest.approx(np.std([0.5, 0.7], ddof=1))
        assert common.mean_sd([0.4])["sd"] is None
        cleaned = common.clean_json({"a": np.float32(1.5), "b": [np.int64(2), float("inf")], "c": np.bool_(True)})
        assert json.dumps(cleaned) == '{"a": 1.5, "b": [2, null], "c": true}'

    def test_no_source_runs(self, prepared_paths):
        with pytest.raises(NoSourceRuns, match="run_all.py --profile smoke"):
            seeds.run_seed_refits(tag=TAG, models=["logreg"])
        assert analyses_status("triage", TAG) == {k: None for k in common.ANALYSIS_KINDS}


# ─────────────────────────────────────────
# Refit z parametrami runa
# ─────────────────────────────────────────
class TestRefit:
    def test_refit_params_machine_keys_and_seed(self):
        params, key = refit_params("cb", {"iterations": 50, "random_seed": 42, "thread_count": 24,
                                          "task_type": "GPU", "devices": "0", "gpu_ram_part": 0.85}, seed=7)
        assert key == "random_seed" and params == {"iterations": 50, "random_seed": 7}
        params, key = refit_params("xgboost", {"max_depth": 6, "random_state": 42, "n_jobs": 24, "device": "cuda"}, 3)
        assert key == "random_state" and params == {"max_depth": 6, "random_state": 3}

    def test_count_scaling_up_down_and_fixed(self):
        info = {"tuned": True, "tuning": {"n_fold_train": 1000, "best_params": {"min_child_samples": 20}},
                "tuning_adjustments": {"factor": 2.0, "n_fold_train": 1000,
                                       "params": {"min_child_samples": {"tuned": 20, "final": 40}}}}
        for n_fit, expected in ((4000, 40), (250, 10), (1000, 20)):
            params = {"min_child_samples": 40, "learning_rate": 0.1}
            train._refit_count_scaling("lightgbm", params, info, n_fit)
            assert params == {"min_child_samples": expected, "learning_rate": 0.1}
        fixed = {"tuned": True, "tuning": {"n_fold_train": 1000, "best_params": {"learning_rate": 0.1}}}
        params = {"min_child_samples": 50, "learning_rate": 0.1}       # stała (custom_params), nie strojona
        assert train._refit_count_scaling("lightgbm", params, fixed, 4000) == {}
        assert params["min_child_samples"] == 50
        assert train._refit_count_scaling("lightgbm", {"min_child_samples": 5}, {"tuned": False}, 10) == {}

    def test_same_seed_reproduces_source_and_markers(self, lgbm_source, prepared_paths):
        board_before = pd.read_csv(prepared_paths.LEADERBOARD_CSV)
        result = refit_from_run(lgbm_source, seed=42, tag=f"{TAG}-seed42", analysis="seeds", unit="lightgbm/x")
        run_dir = Path(result["run_dir"])
        assert run_dir.name.endswith("_lightgbm_triage_smoke-seed42") and result["refit_of"] == lgbm_source.name
        for split in ("val", "test"):
            new = pd.read_parquet(run_dir / f"predictions_{split}.parquet")
            old = pd.read_parquet(lgbm_source / f"predictions_{split}.parquet")
            assert list(new["row_id"]) == list(old["row_id"])
            np.testing.assert_allclose(new[PROBA_COLS].to_numpy(), old[PROBA_COLS].to_numpy(), atol=1e-6)

        done = _read(run_dir / "DONE.json")
        assert done["custom"] is True and done["refit_of"] == lgbm_source.name and done["tuned"] is True
        assert done["refit_version"] == REFIT_VERSION and done["unit"] == "lightgbm/x" and done["seed"] == 42
        info = _read(run_dir / "run_info.json")
        assert info["refit"]["seed_param"] == "random_state" and info["params"]["random_state"] == 42
        assert info["n_fit"] == _read(lgbm_source / "run_info.json")["n_fit"]
        assert sum(info["class_counts_train"]) == info["n_fit"] + info["n_early_stopping"]
        assert (run_dir / "report.md").read_text(encoding="utf-8").startswith(f"# {run_dir.name} — refit")
        # Bez pliku modelu i wiersza leaderboardu: łączenie / orkiestrator / leaderboard go nie biorą
        assert not (run_dir / "model.joblib").exists() and not (run_dir / "leaderboard_row.json").exists()
        train.rebuild_leaderboard()
        assert len(pd.read_csv(prepared_paths.LEADERBOARD_CSV)) == len(board_before)
        from src.ensemble import latest_runs
        from src.pipeline.orchestrator import find_done_run

        assert [r.name for r in latest_runs("triage", include_samples=True, runs_dir=prepared_paths.RUNS_DIR)] == \
            [lgbm_source.name]
        assert find_done_run(prepared_paths.RUNS_DIR, "lightgbm", "triage", TAG, config.DATA_VERSION,
                             done["data_fingerprint"], tuned=True) == lgbm_source
        assert find_refit_run(lgbm_source.name, f"{TAG}-seed42", "lightgbm/x") == run_dir
        assert find_refit_run(lgbm_source.name, f"{TAG}-seed42", "inna") is None

        other = refit_from_run(lgbm_source, seed=5, tag=f"{TAG}-seed5", save_model=True)
        assert _read(Path(other["run_dir"]) / "run_info.json")["params"]["random_state"] == 5
        assert (Path(other["run_dir"]) / "model.joblib").exists()

    def test_mask_fraction_and_eval_frames(self, lgbm_source, prepared_paths):
        from src.data.dataset import load_split

        val = load_split("val")
        frame = val[val["site"].astype(str) == "B"].reset_index(drop=True)
        result = refit_from_run(lgbm_source, seed=1, tag=f"{TAG}-x",
                                train_rows_mask=lambda df: df["site"].astype(str).to_numpy() != "B",
                                train_rows_label="bez B", fit_fraction=0.5, eval_frames={"site_b": frame})
        run_dir = Path(result["run_dir"])
        info = _read(run_dir / "run_info.json")
        n_not_b = int((load_split("train")["site"].astype(str) != "B").sum())
        assert info["refit"]["n_train_rows"] == n_not_b and info["refit"]["train_rows"] == "bez B"
        n_fit_full = n_not_b - info["n_early_stopping"]
        assert info["n_fit"] == round(0.5 * n_fit_full) and info["refit"]["fit_fraction"] == 0.5
        assert info["n_site_b"] == len(frame) and "n_val" not in info
        assert sorted(p.name for p in run_dir.glob("predictions_*.parquet")) == ["predictions_site_b.parquet"]
        assert "site_b_argmax_quadratic_weighted_kappa" in result["metrics"]
        pred = pd.read_parquet(run_dir / "predictions_site_b.parquet")
        assert set(pred["site"].astype(str)) == {"B"} and len(pred) == len(frame)

        only_val = refit_from_run(lgbm_source, seed=1, tag=f"{TAG}-v", eval_splits=("val",))
        assert list(only_val["predictions"]) == ["val"]

    def test_refit_errors(self, lgbm_source, prepared_paths):
        with pytest.raises(ValueError, match="własnego tagu"):
            refit_from_run(lgbm_source, seed=1, tag=TAG)
        with pytest.raises(ValueError, match="fit_fraction"):
            refit_from_run(lgbm_source, seed=1, tag="x", fit_fraction=0.0)
        with pytest.raises(ValueError, match="Nazwa ramki"):
            refit_from_run(lgbm_source, seed=1, tag="x", eval_frames={"Zła nazwa": pd.DataFrame()})
        with pytest.raises(ValueError, match="Nazwa ramki"):
            refit_from_run(lgbm_source, seed=1, tag="x", eval_frames={"fit": pd.DataFrame()})
        with pytest.raises(ValueError, match="eval_splits"):
            refit_from_run(lgbm_source, seed=1, tag="x", eval_splits=("train",))
        with pytest.raises(ValueError, match="nie zostawiła"):
            refit_from_run(lgbm_source, seed=1, tag="x", train_rows_mask=lambda df: np.zeros(len(df), bool))
        refit = Path(refit_from_run(lgbm_source, seed=1, tag="x", eval_splits=("val",))["run_dir"])
        with pytest.raises(ValueError, match="jest refitem"):
            refit_from_run(refit, seed=2, tag="y")
        unfinished = prepared_paths.RUNS_DIR / "20990101_000000_lightgbm_triage_smoke"
        unfinished.mkdir()
        with pytest.raises(FileNotFoundError):
            refit_from_run(unfinished, seed=1, tag="x")
        info = _read(lgbm_source / "run_info.json")
        info["data_fingerprint"] = "inne-dane"
        (lgbm_source / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
        with pytest.raises(RuntimeError, match="innych danych"):
            refit_from_run(lgbm_source, seed=1, tag="x")


# ─────────────────────────────────────────
# Analizy end-to-end (syntetyczne dane)
# ─────────────────────────────────────────
class TestAnalyses:
    def test_seeds_resume_and_summary(self, prepared_paths):
        sources = {m: make_source(m) for m in ("logreg", "lightgbm")}
        res = seeds.run_seed_refits(tag=TAG, models=["logreg", "lightgbm"], n_seeds=2)
        assert res["complete"] and res["n_done"] == res["n_units"] == 6 and res["failures"] == []
        assert res["source_runs"] == {m: d.name for m, d in sources.items()}
        out = prepared_paths.RUNS_DIR / "analyses" / "seeds__triage__smoke"
        on_disk = _read(out / "results.json")
        assert set(on_disk["models"]) == {"logreg", "lightgbm"} and "qwk_argmax" in on_disk["metrics"]
        lgbm = on_disk["models"]["lightgbm"]
        assert [r["seed"] for r in lgbm["runs"]] == [42, 1, 2] and lgbm["runs"][0]["source"] is True
        stats = lgbm["val"]["log_loss"]
        assert stats["n"] == 3 and stats["mean"] == pytest.approx(np.mean(stats["values"]))
        assert {"mean", "sd", "values"} <= set(lgbm["test"]["qwk_argmax"])
        assert on_disk["models"]["logreg"]["val"]["log_loss"]["sd"] == pytest.approx(0.0, abs=1e-9)
        assert (out / "summary.md").exists() and len(list((out / "units").glob("*.json"))) == 6
        refits = refit_dirs(prepared_paths.RUNS_DIR)
        assert sorted(d.name.split("_triage_")[1] for d in refits) == ["smoke-seed1", "smoke-seed1",
                                                                       "smoke-seed2", "smoke-seed2"]
        progress = analyses_status("triage", TAG)["seeds"]
        assert progress["done"] == progress["total"] == 6 and progress["complete"]

        # Wznowienie: nic nowego; utracony plik jednostki → metryki z gotowego refitu, bez treningu
        assert seeds.run_seed_refits(tag=TAG, models=["logreg", "lightgbm"], n_seeds=2)["complete"]
        (out / "units" / "lightgbm__seed1.json").unlink()
        assert seeds.run_seed_refits(tag=TAG, models=["logreg", "lightgbm"], n_seeds=2)["n_done"] == 6
        assert refit_dirs(prepared_paths.RUNS_DIR) == refits
        # Nowy run źródłowy (nowszy) → jednostki tego modelu liczone od nowa; wyniki kumulatywne:
        # lightgbm (niewywołany teraz, ten sam run źródłowy) zostaje w results.json i w postępie
        newer = make_source("logreg")
        res = seeds.run_seed_refits(tag=TAG, models=["logreg"], n_seeds=1)
        assert res["source_runs"] == {"logreg": newer.name, "lightgbm": sources["lightgbm"].name}
        assert len(refit_dirs(prepared_paths.RUNS_DIR)) == 5
        assert res["models"]["logreg"]["n_runs"] == 2 and res["models"]["lightgbm"]["n_runs"] == 3
        assert res["complete"] and res["n_done"] == res["n_units"] == 5
        assert list(res["models"]) == ["logreg", "lightgbm"]                       # kolejność rejestru
        progress = analyses_status("triage", TAG)["seeds"]
        assert progress["done"] == progress["total"] == 5

    def test_site_out(self, prepared_paths):
        from src.data.dataset import load_split

        source = make_source("logreg")
        res = site_out.run_leave_site_out(tag=TAG, models=["logreg"], sites=["a"])
        assert res["complete"] and res["sites_requested"] == ["A"]
        entry = res["sites"]["A"]["logreg"]
        train_df, val_df, test_df = (load_split(s) for s in ("train", "val", "test"))
        n_site = int((train_df["site"].astype(str) == "A").sum() + (val_df["site"].astype(str) == "A").sum())
        assert entry["n"] == n_site and entry["n_esi1"] >= 1
        for variant in site_out.VARIANTS:
            assert {"qwk_argmax", "undertriage_argmax", "log_loss", "admission_auc", "oe_esi1"} <= set(entry[variant])
        for key in ("train", "site", "em"):
            assert sum(entry["priors"][key]) == pytest.approx(1.0)
        assert entry["priors"]["site"] != entry["priors"]["train"]
        assert "in_distribution" in entry and "transported_same_rows" in entry

        run_dir = prepared_paths.RUNS_DIR / entry["run"]
        info = _read(run_dir / "run_info.json")
        assert info["refit"]["n_train_rows"] == int((train_df["site"].astype(str) != "A").sum())
        held = pd.read_parquet(run_dir / "predictions_heldout.parquet")
        assert set(held["site"].astype(str)) == {"A"}
        assert not set(held["row_id"]) & set(test_df["row_id"])                      # nigdy test
        assert set(held["row_id"]) <= set(train_df["row_id"]) | set(val_df["row_id"])
        dev = pd.read_parquet(run_dir / "predictions_dev_val.parquet")
        assert "A" not in set(dev["site"].astype(str)) and set(dev["row_id"]) <= set(val_df["row_id"])
        assert _read(run_dir / "DONE.json")["refit_of"] == source.name
        # Wznowienie bez nowych refitów; nieznany oddział → błąd argumentów PRZED planem i treningiem
        assert site_out.run_leave_site_out(tag=TAG, models=["logreg"], sites=["A"])["n_done"] == 1
        plan_before = (prepared_paths.RUNS_DIR / "analyses" / "site_out__triage__smoke" / "plan.json").read_text(
            encoding="utf-8")
        with pytest.raises(common.AnalysisArgumentError, match="Z"):
            site_out.run_leave_site_out(tag=TAG, models=["logreg"], sites=["A", "Z"])
        assert (prepared_paths.RUNS_DIR / "analyses" / "site_out__triage__smoke" / "plan.json").read_text(
            encoding="utf-8") == plan_before
        assert len(refit_dirs(prepared_paths.RUNS_DIR)) == 1
        # Wyniki kumulatywne: wywołanie dla B dopisuje oddział, A zostaje
        res = site_out.run_leave_site_out(tag=TAG, models=["logreg"], sites=["B"])
        assert set(res["sites"]) == {"A", "B"} and res["sites_requested"] == ["A", "B"] and res["complete"]
        assert analyses_status("triage", TAG)["site_out"]["total"] == 2

    def test_oe_is_observed_over_expected(self):
        y = np.array([0, 0, 0, 0, 1, 2, 2, 2, 3, 4])
        p = np.full((10, 5), 0.2)                       # Σ P(ESI 1) = 2, obserwowane 4
        assert site_out._oe(y, p, [0]) == pytest.approx(2.0)          # > 1 = model zaniża ESI 1
        assert site_out._oe(y, p, [0, 1]) == pytest.approx(5 / 4)
        assert np.isnan(site_out._oe(y, np.zeros((10, 5)), [0]))

    def test_effective_train_prior_uses_class_weights(self):
        counts = [10, 90, 400, 400, 100]
        plain, note = site_out.effective_train_prior(counts, {"weights_strategy": "none"})
        np.testing.assert_allclose(plain, np.array(counts) / 1000) and note is None
        optuna, note = site_out.effective_train_prior(
            counts, {"weights_strategy": "optuna_qwk", "class_weights": {"0": 30.0, "1": 2.0}})
        expected = np.array([300.0, 180, 400, 400, 100])
        np.testing.assert_allclose(optuna, expected / expected.sum())
        assert note == "optuna_qwk" and optuna[0] > 20 * plain[0]
        balanced, _ = site_out.effective_train_prior(
            counts, {"weights_strategy": "balanced", "sample_weight_strategy": "balanced"})
        np.testing.assert_allclose(balanced, np.full(5, 0.2))

    def test_learning_curve(self, lgbm_source, prepared_paths):
        res = learning_curve.run_learning_curve(tag=TAG, models=["lightgbm"], fractions=[0.25, 0.5, 1.0])
        assert res["complete"] and res["fractions"] == [0.25, 0.5, 1.0]
        entry = res["models"]["lightgbm"]
        n_fit = [p["n_fit"] for p in entry["points"]]
        assert n_fit == sorted(n_fit) and n_fit[-1] == _read(lgbm_source / "run_info.json")["n_fit"]
        assert set(entry["power_law"]) == set(learning_curve.FIT_METRICS)
        assert entry["n_fit_full_train"] > 0
        out = prepared_paths.RUNS_DIR / "analyses" / "learning_curve__triage__smoke"
        assert (out / "learning_curve.png").exists() and (out / "summary.md").exists()
        # Punkt 100% (ten sam seed) odtwarza run źródłowy; refity tylko z predykcjami val
        full = prepared_paths.RUNS_DIR / entry["points"][-1]["run"]
        np.testing.assert_allclose(pd.read_parquet(full / "predictions_val.parquet")[PROBA_COLS].to_numpy(),
                                   pd.read_parquet(lgbm_source / "predictions_val.parquet")[PROBA_COLS].to_numpy(),
                                   atol=1e-6)
        assert not (full / "predictions_test.parquet").exists()

    def test_power_law_reliability_checks(self):
        """Ekstrapolacja gorsza od ostatniego punktu, zły kierunek, daleki cel → niewiarygodna z powodem."""
        n = [270, 540, 1350, 2700, 5400]
        smoke = learning_curve.fit_power_law(n, [0.150, 0.281, 0.309, 0.311, 0.374], [401_779], True)
        assert not smoke["reliable"] and smoke["predicted"]["401779"] < 0.374
        assert any("gorsza od ostatniego punktu" in r for r in smoke["reasons"])
        assert any("ekstrapolacja > 10" in r for r in smoke["reasons"])
        x = np.array([20_000, 40_000, 100_000, 200_000, 400_000], dtype=float)
        qwk = 0.70 - 3 * x ** -0.5 + np.array([0.001, -0.001, 0.0005, 0.0, 0.0008])
        good = learning_curve.fit_power_law(x, qwk, [400_000, 800_000], True)
        assert good["reliable"] and good["reasons"] == []
        wrong_way = learning_curve.fit_power_law(x, qwk, [800_000], False)        # QWK jako „mniej = lepiej”
        assert not wrong_way["reliable"] and "krzywa pogarsza się z liczbą danych" in wrong_way["reasons"]
        assert learning_curve.HIGHER_IS_BETTER["log_loss"] is False

    def test_power_law_fit(self):
        n = np.array([1_000, 2_000, 5_000, 10_000, 50_000], dtype=float)
        y = 0.75 + 2.0 * n ** -0.4
        fit = learning_curve.fit_power_law(n, y, extrapolate_to=[100_000])
        assert fit["a"] == pytest.approx(0.75, abs=0.01) and fit["c"] == pytest.approx(0.4, abs=0.05)
        assert fit["predicted"]["100000"] == pytest.approx(0.75 + 2.0 * 100_000 ** -0.4, abs=0.005)
        assert fit["reliable"] and not fit["c_fixed"] and not fit["at_bound"]
        three = learning_curve.fit_power_law(n[:3], y[:3], extrapolate_to=[100_000])
        assert three["c_fixed"] and three["c"] == learning_curve.C_FIXED and not three["reliable"]
        linear = learning_curve.fit_power_law(n, 1.0 - 0.01 * np.log(n))          # bez asymptoty → c na granicy
        assert linear["at_bound"] and not linear["reliable"] and "c na granicy przedziału" in linear["reasons"]
        assert fit["reasons"] == []
        assert learning_curve.fit_power_law(n[:2], y[:2]) is None
        assert learning_curve.parse_fractions("1,0.1,0.1") == [0.1, 1.0]
        with pytest.raises(ValueError):
            learning_curve.parse_fractions("0,0.5")


# ─────────────────────────────────────────
# Skrypty 09–11 (CLI)
# ─────────────────────────────────────────
def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), PROJECT_ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", ["09_seed_refits.py", "10_leave_site_out.py", "11_learning_curve.py"])
def test_scripts_exit_2_without_source_runs(prepared_paths, script):
    assert _load_script(script).run(["--tag", "brak-takiego-tagu", "--models", "logreg"]) == 2


def test_scripts_bad_arguments(prepared_paths):
    assert _load_script("11_learning_curve.py").run(["--tag", TAG, "--fractions", "0,2"]) == 2
    assert _load_script("10_leave_site_out.py").run(["--tag", TAG, "--sites", " , "]) == 2
    make_source("logreg")
    assert _load_script("10_leave_site_out.py").run(["--tag", TAG, "--sites", "Z", "--models", "lr"]) == 2
    assert not (prepared_paths.RUNS_DIR / "analyses" / "site_out__triage__smoke" / "plan.json").exists()
    assert refit_dirs(prepared_paths.RUNS_DIR) == []
    with pytest.raises(SystemExit):
        _load_script("09_seed_refits.py").run(["--seeds", "0"])


def test_seed_script_runs_and_resumes(prepared_paths, capsys):
    make_source("logreg")
    script = _load_script("09_seed_refits.py")
    assert script.run(["--tag", TAG, "--models", "lr", "--seeds", "1"]) == 0
    assert script.run(["--tag", TAG, "--models", "lr", "--seeds", "1"]) == 0
    assert len(refit_dirs(prepared_paths.RUNS_DIR)) == 1
