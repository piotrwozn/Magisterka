"""
Testy eksperymentu A (src/analysis/small_data.py, scripts/14_small_data_curve.py): losowanie podprób
(sparowane row_id, ≥ 2 wiersze każdej klasy, czystość site_local, obcy kontekst), wznawianie, statystyka
(znane odpowiedzi: Holm, przedział t, dwupoziomowy bootstrap, przecięcie krzywych), raport i CLI.

Modele TFM: atrapa estymatora (`SOR_AI_TFM_FAKE=1`) na syntetycznym Yale (fixture `prepared_paths`);
test z PRAWDZIWYMI bibliotekami (maleńkie losowe checkpointy z tests/tfm_testkit.py) — pomijany bez torch.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import small_data as sd
from src.analysis.common import AnalysisArgumentError, analysis_progress
from src.utils import config
from tests.tfm_testkit import HAS_TABICL, HAS_TABPFN, write_tiny_tabicl, write_tiny_tabpfn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAG = "smoke"


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_script():
    spec = importlib.util.spec_from_file_location("small_data_script", PROJECT_ROOT / "scripts" / "14_small_data_curve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_tfm(monkeypatch):
    monkeypatch.setenv("SOR_AI_TFM_FAKE", "1")
    monkeypatch.setenv("SOR_AI_DEVICE", "cpu")
    for var in ("SOR_AI_TABPFN_MODEL_PATH", "SOR_AI_TABICL_MODEL_PATH", "TABPFN_MODEL_CACHE_SIZE"):
        monkeypatch.delenv(var, raising=False)


def small_cfg(**overrides) -> sd.SmallDataConfig:
    base = dict(feature_set="triage", tag=TAG, models=["tabpfn", "tabicl"], scenarios=["pooled", "site_local"],
                sizes=[60, 120], repeats=[2], eval_rows=0, local_eval_rows=0, local_site="B", foreign_rows=50,
                transport_rows=300, lgbm_trials=2, n_boot=40, device="auto", seed=42)
    base.update(overrides)
    return sd.SmallDataConfig(**base)


def _root(paths) -> Path:
    return paths.RUNS_DIR / "analyses" / "small_data__triage__smoke"


def _ids(root: Path, name: str) -> np.ndarray:
    return np.load(root / "samples" / f"{name}.npy")


def _split(paths, name: str) -> pd.DataFrame:
    return pd.read_parquet(Path(paths.ESI_DATA_DIR) / f"{name}.parquet", columns=["row_id", "site", "target"])


# ─────────────────────────────────────────
# Argumenty
# ─────────────────────────────────────────
class TestArguments:
    def test_sizes_repeats(self):
        assert sd.parse_sizes_repeats("1000,500", "3") == ([500, 1000], [3, 3])
        assert sd.parse_sizes_repeats("500,1_000", "10,5") == ([500, 1000], [10, 5])
        assert sd.parse_sizes_repeats([2000, 500], [1, 4]) == ([500, 2000], [4, 1])     # pary zostają razem
        for sizes, repeats in (("500,1000", "1,2,3"), ("10", "1"), ("500,500", "1"), ("500", "0"), ("a", "1")):
            with pytest.raises(AnalysisArgumentError):
                sd.parse_sizes_repeats(sizes, repeats)

    def test_models_and_scenarios(self):
        assert sd.parse_tfm_models("icl,pfn") == ["tabpfn", "tabicl"]
        assert sd.parse_tfm_models("tfm") == ["tabpfn", "tabicl"]
        assert sd.parse_tfm_models("none") == [] and sd.parse_tfm_models("") == []
        # Orkiestrator przekazuje modele kroku z lightgbm — akceptowane, LightGBM liczony zawsze
        assert sd.parse_tfm_models("lightgbm,tabpfn,tabicl") == ["tabpfn", "tabicl"]
        assert sd.parse_tfm_models("lgbm") == []
        for bad in ("xgboost", "xyz", "all"):
            with pytest.raises(AnalysisArgumentError):
                sd.parse_tfm_models(bad)
        assert sd.parse_scenarios("site_local,pooled") == ["pooled", "site_local"]
        with pytest.raises(AnalysisArgumentError):
            sd.parse_scenarios("pooled,other")

    def test_validate_config(self):
        cfg = sd.validate_config(small_cfg(models="pfn", scenarios="pooled", local_site=" b ", device="GPU"))
        assert cfg.models == ["tabpfn"] and cfg.scenarios == ["pooled"] and cfg.local_site == "B" and cfg.device == "gpu"
        for bad in (dict(device="tpu"), dict(eval_rows=-1), dict(lgbm_trials=0), dict(feature_set="nope")):
            with pytest.raises(AnalysisArgumentError):
                sd.validate_config(small_cfg(**bad))


# ─────────────────────────────────────────
# Losowanie
# ─────────────────────────────────────────
class TestSampling:
    def test_derive_seed(self):
        assert sd.derive_seed(42, "pooled", 500, 0) == sd.derive_seed(42, "pooled", 500, 0)
        assert sd.derive_seed(42, "pooled", 500, 0) != sd.derive_seed(42, "pooled", 500, 1)
        assert sd.derive_seed(42, "a") != sd.derive_seed(43, "a")
        assert 0 <= sd.derive_seed(7, "x") < 2**31 - 1

    def test_stratified_sample_is_proportional(self):
        rng = np.random.default_rng(0)
        labels = np.array(["a"] * 600 + ["b"] * 300 + ["c"] * 100)
        pool = np.arange(1000) + 5000
        chosen = sd.stratified_sample(pool, labels, 100, rng)
        assert len(chosen) == 100 and len(np.unique(chosen)) == 100 and np.isin(chosen, pool).all()
        lab = labels[chosen - 5000]
        assert [(lab == k).sum() for k in "abc"] == [60, 30, 10]
        np.testing.assert_array_equal(sd.stratified_sample(pool, labels, 5000, rng), pool)
        assert len(sd.stratified_sample(pool, labels, 0, rng)) == 0

    def test_min_per_class_swap_keeps_size(self):
        y = np.array([0] * 3 + [1] * 200 + [2] * 500 + [3] * 250 + [4] * 47)
        pool = np.arange(len(y))
        for seed in range(20):
            chosen = sd.draw_sample(pool, y.astype(str), y, 50, seed)
            assert len(chosen) == 50 and len(np.unique(chosen)) == 50
            counts = np.bincount(y[chosen], minlength=5)
            assert (counts >= 2).all(), counts                         # ESI 1 dobrane w miejsce ESI 3
        # Klasa z 1 wierszem w puli — wystarczy ten jeden
        y1 = np.array([0] + [2] * 99)
        chosen = sd.draw_sample(np.arange(100), y1.astype(str), y1, 10, 3)
        assert (y1[chosen] == 0).sum() == 1 and len(chosen) == 10
        # Deterministycznie
        np.testing.assert_array_equal(sd.draw_sample(pool, y.astype(str), y, 50, 5),
                                      sd.draw_sample(pool, y.astype(str), y, 50, 5))

    def test_sample_bank_reuses_and_redraws(self, tmp_path):
        row_ids = np.arange(100) * 10 + 7
        calls = []

        def draw():
            calls.append(1)
            return np.array([5, 1, 3])

        bank = sd.SampleBank(tmp_path, {"seed": 1})
        np.testing.assert_array_equal(bank.get("s", row_ids, draw, 3), [1, 3, 5])
        bank.write_manifest()
        np.testing.assert_array_equal(np.load(tmp_path / "samples" / "s.npy"), row_ids[[1, 3, 5]])
        again = sd.SampleBank(tmp_path, {"seed": 1})
        np.testing.assert_array_equal(again.get("s", row_ids, draw, 3), [1, 3, 5])
        assert len(calls) == 1 and again.reused == 1
        # Inny seed (manifest) albo inna liczność → losowanie od nowa
        sd.SampleBank(tmp_path, {"seed": 2}).get("s", row_ids, draw, 3)
        assert len(calls) == 2
        sd.SampleBank(tmp_path, {"seed": 2}).get("s", row_ids, draw, 4)
        assert len(calls) == 3


# ─────────────────────────────────────────
# Statystyka — znane odpowiedzi
# ─────────────────────────────────────────
class TestStatistics:
    def test_holm_known_answer(self):
        adjusted = sd.holm([0.01, 0.04, 0.03, 0.005])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02])
        assert sd.holm([0.02, None, float("nan"), 0.04]) == [0.04, None, None, 0.04]
        assert sd.holm([0.6, 0.9]) == [1.0, 1.0]
        assert sd.holm([]) == []

    def test_t_interval_known_answer(self):
        t = sd.t_interval([1, 2, 3, 4, 5])
        assert t["mean"] == pytest.approx(3.0) and t["sd"] == pytest.approx(math.sqrt(2.5)) and t["df"] == 4
        # t_{0.975, 4} = 2.776445; se = √0.5
        assert t["ci_low"] == pytest.approx(3 - 2.776445 * math.sqrt(0.5), abs=1e-5)
        assert t["ci_high"] == pytest.approx(3 + 2.776445 * math.sqrt(0.5), abs=1e-5)
        assert t["p"] == pytest.approx(0.013236, abs=1e-5)                 # t = 4.2426, df = 4
        one = sd.t_interval([0.3, None])
        assert one["mean"] == 0.3 and one["ci_low"] is None and one["p"] is None
        flat = sd.t_interval([0.2, 0.2, 0.2])
        assert flat["ci_low"] == flat["ci_high"] == pytest.approx(0.2) and flat["p"] == 0.0

    def test_two_level_bootstrap(self):
        rng = np.random.default_rng(0)
        np.testing.assert_allclose(sd.two_level_draws(np.full((3, 50), 0.25), rng), 0.25)
        row = np.linspace(-1, 1, 50)
        np.testing.assert_allclose(sd.two_level_draws(row[None, :], rng), row)     # R = 1 → tylko poziom 2
        diffs = np.vstack([np.full(400, -1.0), np.full(400, 1.0)])
        draws = sd.two_level_draws(diffs, np.random.default_rng(1))
        assert set(np.unique(draws)) <= {-1.0, 0.0, 1.0}
        assert 0.1 < (draws == 0).mean() < 0.9                             # losowanie powtórzeń działa
        with pytest.raises(ValueError):
            sd.two_level_draws(np.ones(5), rng)

    def test_boot_summary_tails(self):
        s = sd.boot_summary(np.arange(1, 1001, dtype=float))
        assert s["ci_low"] == 25 and s["ci_high"] == 976 and s["p"] == pytest.approx(0.001)
        s0 = sd.boot_summary(np.concatenate([np.full(500, -1.0), np.full(500, 1.0)]))
        assert s0["p"] == 1.0 and s0["ci_low"] < 0 < s0["ci_high"]
        assert sd.boot_summary(np.array([np.nan]))["p"] is None

    def test_crossover_known_answer(self):
        tfm = {"a": 0.70, "b": 2.0, "c": 0.5}                 # lepszy przy małym n
        lgbm = {"a": 0.60, "b": 10.0, "c": 0.5}               # 0.70 + 2/√n = 0.60 + 10/√n → n* = 6400
        res = sd.crossover(tfm, lgbm, 100, 1e6)
        assert res["status"] == "arm_better_below" and res["n_star"] == pytest.approx(6400, rel=1e-6)
        assert sd.crossover(lgbm, tfm, 100, 1e6)["status"] == "arm_better_above"
        assert sd.crossover(tfm, lgbm, 100, 1000)["status"] == "arm_better_everywhere"
        assert sd.crossover(tfm, lgbm, 1e5, 1e6)["status"] == "base_better_everywhere"
        assert sd.crossover(None, lgbm, 100, 1000)["status"] == "no_fit"
        # QWK (im więcej, tym lepiej): odwrócony kierunek
        res_q = sd.crossover({"a": 0.3, "b": -2.0, "c": 0.5}, {"a": 0.4, "b": -10.0, "c": 0.5}, 100, 1e6, True)
        assert res_q["status"] == "arm_better_below" and res_q["n_star"] == pytest.approx(6400, rel=1e-6)

    def test_crossover_bootstrap_over_repeats(self):
        rng = np.random.default_rng(3)
        sizes = [500, 1000, 2000, 5000, 10000, 50000]
        per_size = {}
        for n in sizes:
            arm = 0.70 + 2.0 / math.sqrt(n) + rng.normal(0, 0.0005, 5)
            base = 0.60 + 10.0 / math.sqrt(n) + rng.normal(0, 0.0005, 5)
            per_size[n] = (arm.tolist(), base.tolist())
        res = sd.crossover_bootstrap(per_size, 200, seed=1)
        assert res["status"] == "arm_better_below" and res["in_range"] is True
        assert res["n_star"] == pytest.approx(6400, rel=0.1)
        assert res["ci_low"] <= res["n_star"] <= res["ci_high"] and res["share"]["arm_better_below"] > 0.9
        few = sd.crossover_bootstrap({500: ([0.9], [1.0]), 1000: ([0.8], [0.9])}, 50, seed=1)
        assert few["status"] == "no_fit" and "za mało" in few["reason"]

    def test_verdict_rule(self):
        assert sd.verdict(-0.02, 0.01, "log_loss")[0] == "win"
        assert sd.verdict(0.02, 0.01, "log_loss")[0] == "loss"
        assert sd.verdict(-0.003, 0.001, "log_loss")[0] == "tie_small"          # |Δ| ≤ MID 0.005
        assert sd.verdict(-0.02, 0.2, "log_loss")[0] == "tie"
        assert sd.verdict(0.03, 0.01, "qwk_argmax")[0] == "win"                 # QWK: więcej = lepiej
        assert sd.verdict(0.001, 0.01, "brier") == ("loss", "odniesienie lepsze (bez MID)")
        assert sd.verdict(None, 0.01, "log_loss")[0] == "na"

    def test_collect_statistics_pipeline(self, tmp_path):
        """Model lepszy przy małym n, gorszy przy dużym (różnica ≫ MID) → wygrana / przegrana, Holm ≥ p."""
        rng = np.random.default_rng(0)
        n_eval = 600
        y = rng.choice(5, n_eval, p=[0.1, 0.25, 0.35, 0.2, 0.1])
        onehot = np.eye(5)[y]

        def proba(sharp: float, noise: float, seed: int) -> np.ndarray:
            r = np.random.default_rng(seed)
            logits = sharp * onehot + r.normal(0, noise, (n_eval, 5))
            e = np.exp(logits)
            return e / e.sum(axis=1, keepdims=True)

        done = []
        quality = {100: (2.5, 1.0), 1000: (2.0, 1.5), 10000: (1.0, 2.0)}      # (TFM, LightGBM) „ostrość”
        for n, (s_tfm, s_lgbm) in quality.items():
            for r in range(3):
                for arm, sharp in (("tabpfn", s_tfm + 0.05 * r), (sd.BASELINE, s_lgbm + 0.05 * r),
                                   ("lightgbm_default", s_lgbm)):
                    p = proba(sharp, 0.8, n * 10 + r)
                    rel = f"preds/x_{arm}_{n}_{r}.npy"
                    (tmp_path / "preds").mkdir(exist_ok=True)
                    np.save(tmp_path / rel, p.astype(np.float32))
                    done.append({"arm": arm, "n": n, "repeat": r, "preds": rel, "metrics": sd.unit_metrics(y, p)})
        block = sd.collect_statistics(tmp_path, "pooled", done, y, n_boot=200, seed=1)
        prim = {c["n"]: c for c in block["comparisons"] if c["arm"] == "tabpfn" and c["metric"] == "log_loss"}
        assert prim[100]["verdict"] == "win" and prim[10000]["verdict"] == "loss"
        for c in prim.values():
            assert c["R"] == 3 and c["p_holm"] >= c["p_boot"] - 1e-12
            assert c["boot_ci_low"] <= c["delta"] <= c["boot_ci_high"]
            assert c["t_ci_low"] <= c["delta"] <= c["t_ci_high"]
        # Δ paired = różnica metryk z bootstrapu na całym zbiorze, średnio po powtórzeniach
        assert prim[100]["delta"] < -sd._mid("log_loss")
        pts = block["points"]["tabpfn"]["100"]
        assert pts["R"] == 3 and pts["metrics"]["log_loss"]["n"] == 3
        cross = block["crossover"][f"tabpfn|{sd.BASELINE}"]
        assert cross["status"] in ("arm_better_below", "multiple") and cross["n_star"] is not None
        assert block["power_law"]["tabpfn"]["c_fixed"] is True                # 3 rozmiary → c = 0.5
        # bez bootstrapu: tylko przedziały t, Holm z p testu t
        plain = sd.collect_statistics(tmp_path, "pooled", done, y, n_boot=0, seed=1)
        row = next(c for c in plain["comparisons"] if c["arm"] == "tabpfn" and c["n"] == 100 and c["metric"] == "log_loss")
        assert row["boot_ci_low"] is None and row["p_holm"] is not None and row["delta"] == pytest.approx(prim[100]["delta"])

    def test_unit_metrics_keys(self):
        rng = np.random.default_rng(0)
        y = np.r_[np.zeros(5, int), rng.integers(1, 5, 95)]
        p = rng.dirichlet(np.ones(5), 100)
        m = sd.unit_metrics(y, p, vc=np.arange(100) % 2 == 0)
        for key in ("log_loss", "brier", "ece", "ece_esi1", "ece_esi5", "auc_macro", "qwk_argmax", "qwk_cost",
                    "undertriage_cost", "recall_esi1_cost", "vc_log_loss", "vc_brier"):
            assert m[key] is not None, key
        assert m["esi1_counts"]["n"] == 5 and 0 <= m["esi1_counts"]["k_cost"] <= 5


# ─────────────────────────────────────────
# Plan jednostek
# ─────────────────────────────────────────
class TestUnits:
    COUNTS = sd.SourceCounts(n_train=10_000, n_val=2_000, sites_train={"A": 7_000, "B": 3_000},
                             sites_val={"A": 1_400, "B": 600}, local_train=3_000, local_val=600,
                             foreign_train=7_000, local_esi1_val=6)

    def statuses(self, device="cpu", available=("tabpfn", "tabicl")):
        return {m: {"available": m in available, "device": device, "reason": None if m in available else "brak torch"}
                for m in ("tabpfn", "tabicl")}

    def test_cpu_guard_size_limit_and_order(self):
        assert config.TFM_CPU_MAX_CONTEXT == 5_000
        cfg = sd.validate_config(small_cfg(sizes=[1000, 3000, 5000], repeats=[1], foreign_rows=2_500))
        units, skipped = sd.build_units(cfg, self.COUNTS, self.statuses("cpu"), "run_lgbm")
        names = {u["unit"] for u in units}
        # pooled: kontekst = n ≤ 5000 → liczone na CPU
        assert {"pooled/n3000/r0/tabpfn", "pooled/n5000/r0/tabicl"} <= names
        assert any(s["unit"] == "site_local/n5000" for s in skipped)                    # > 3000 wierszy oddziału
        assert "site_local/n1000/r0/tabpfn+foreign" in names                            # 1000 + 2500 ≤ 5000
        assert "site_local/n3000/r0/tabicl" in names                                    # n = całe źródło — dozwolone
        assert "site_local/n3000/r0/tabicl+foreign" not in names                        # 3000 + 2500 > 5000 na CPU
        assert any(s["unit"] == "site_local/n3000/r0/tabicl+foreign" and "TFM_CPU_MAX_CONTEXT" in s["reason"]
                   for s in skipped)
        transported = [u for u in units if u["arm"] == sd.TRANSPORTED]
        assert len(transported) == 1 and transported[0]["source_run"] == "run_lgbm"
        assert all(u["source_run"] == "none" for u in units if u["arm"] != sd.TRANSPORTED)
        groups = [u["group"] for u in units]
        assert groups == sorted(groups, key=["lightgbm", "tabpfn", "tabicl"].index)      # grupy po kolei
        cfg_big = sd.validate_config(small_cfg(sizes=[6000], repeats=[1], scenarios=["pooled"]))
        units_big, skipped_big = sd.build_units(cfg_big, self.COUNTS, self.statuses("cpu"), None)
        assert {u["arm"] for u in units_big} == set(sd.LGBM_ARMS)
        assert any("TFM_CPU_MAX_CONTEXT" in s["reason"] for s in skipped_big)
        units_gpu, _ = sd.build_units(cfg_big, self.COUNTS, self.statuses("cuda"), None)
        assert {u["arm"] for u in units_gpu} == {*sd.LGBM_ARMS, "tabpfn", "tabicl"}

    def test_unavailable_model_skipped(self):
        cfg = sd.validate_config(small_cfg(sizes=[500], repeats=[2], foreign_rows=0, transport_rows=0))
        units, skipped = sd.build_units(cfg, self.COUNTS, self.statuses("cpu", available=("tabicl",)), None)
        arms = {u["arm"] for u in units}
        assert "tabpfn" not in arms and "tabicl" in arms and sd.TRANSPORTED not in arms
        assert not any(a.endswith(sd.FOREIGN_SUFFIX) for a in arms)
        assert {"unit": "*/tabpfn", "reason": "brak torch"} in skipped
        # Ten sam seed dla ramion jednej podpróby i modelu bazowego; różne powtórzenia → różne seedy
        seeds = {u["unit"]: u["seed"] for u in units}
        assert seeds["pooled/n500/r0/lightgbm_tuned"] == seeds["pooled/n500/r0/lightgbm_default"]
        assert seeds["pooled/n500/r0/lightgbm_tuned"] != seeds["pooled/n500/r1/lightgbm_tuned"]

    def test_estimates_from_benchmark(self, tmp_path):
        bench = tmp_path / "benchmark.json"
        bench.write_text(json.dumps({"kind": "tfm_setup_benchmark", "results": [      # schemat 13_tfm_setup
            {"model": "tabpfn", "n_ctx": 1000, "n_estimators": 8, "status": "ok", "rows_per_s": 1000.0, "fit_s": 1.0},
            {"model": "tabpfn", "n_ctx": 10000, "n_estimators": 4, "status": "ok", "rows_per_s": 200.0, "fit_s": 5.0},
            {"model": "tabpfn", "n_ctx": 50000, "n_estimators": 4, "status": "oom", "rows_per_s": None},
            {"model": "tabicl", "n_ctx": "x", "rows_per_s": 1}]}), encoding="utf-8")
        rates = sd._benchmark_rates(bench)
        assert list(rates) == ["tabpfn"]
        assert rates["tabpfn"] == [(1000.0, 1000.0, 1.0), (10000.0, 100.0, 10.0)]          # 4 → 8 estymatorów
        unit = {"arm": "tabpfn", "base_model": "tabpfn", "n": 1000, "context": 1000}
        assert sd.estimate_unit_s(unit, 2000, 40, rates, "cuda") == pytest.approx(1.0 + 2.0)
        assert sd.estimate_unit_s({**unit, "arm": "tabicl", "base_model": "tabicl"}, 2000, 40, rates, "cpu") is None
        gpu = sd.estimate_unit_s({**unit, "arm": "tabicl", "base_model": "tabicl"}, 20_000, 40, rates, "cuda")
        assert gpu == pytest.approx(6.0)                                                 # plan pamięci
        assert sd.estimate_unit_s({"arm": "lightgbm_tuned", "base_model": "lightgbm", "n": 500}, 2000, 40, {}, None) > 0
        assert sd._benchmark_rates(tmp_path / "brak.json") == {}


# ─────────────────────────────────────────
# Dostępność TFM
# ─────────────────────────────────────────
def test_tfm_status_fake_and_missing_deps(fake_tfm, monkeypatch):
    st = sd.tfm_status("tabpfn", "auto")
    assert st["available"] and st["fake"] and st["device"] == "cpu" and st["library_version"] == "fake"
    assert sd.tfm_status("tabicl", "gpu")["device"] == "cuda"
    monkeypatch.delenv("SOR_AI_TFM_FAKE")
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name in ("torch", "tabicl")
                        else real_find_spec(name, *a))
    missing = sd.tfm_status("tabicl", "auto")
    assert not missing["available"] and "torch" in missing["reason"] and "requirements-tfm.txt" in missing["reason"]


# ─────────────────────────────────────────
# Przebieg na syntetycznych danych (atrapa TFM)
# ─────────────────────────────────────────
def test_end_to_end_fake_tfm_and_resume(prepared_paths, fake_tfm, monkeypatch):
    results = sd.run_small_data(small_cfg())
    root = _root(prepared_paths)
    assert results["failures"] == [] and results["complete"] and results["n_done"] == results["n_units"]
    assert results["skipped_models"] == {}
    for name in ("results.json", "units.csv", "report.md", "summary.md", "plan.json",
                 "figures/curves_pooled.png", "figures/delta_site_local.png"):
        assert (root / name).exists(), name

    train, val = _split(prepared_paths, "train"), _split(prepared_paths, "val")
    # Podpróby: liczność, ≥ 2 wiersze każdej klasy źródła, site_local tylko z oddziału B
    for n in (60, 120):
        for r in (0, 1):
            ids = _ids(root, f"pooled_n{n}_r{r}")
            rows = train.set_index("row_id").loc[ids]
            assert len(ids) == n and (np.bincount(rows["target"], minlength=5) >= 2).all()
            local = train.set_index("row_id").loc[_ids(root, f"site_local_n{n}_r{r}")]
            assert len(local) == n and (local["site"] == "B").all()
            b_counts = np.bincount(train.loc[train["site"] == "B", "target"], minlength=5)
            assert (np.bincount(local["target"], minlength=5) >= np.minimum(2, b_counts)).all()
            foreign = train.set_index("row_id").loc[_ids(root, f"site_local_n{n}_r{r}_foreign")]
            assert len(foreign) == 50 and (foreign["site"] != "B").all()
    assert not np.array_equal(_ids(root, "pooled_n60_r0"), _ids(root, "pooled_n60_r1"))
    np.testing.assert_array_equal(np.sort(_ids(root, "site_local_eval")),
                                  np.sort(val.loc[val["site"] == "B", "row_id"].to_numpy()))
    assert len(_ids(root, "pooled_eval")) == len(val)

    # Układ sparowany: wszystkie modele jednej podpróby mają ten sam klucz prób; kontekst TFM = n (+ obce)
    units = {p.stem: _read(p) for p in (root / "units").glob("*.json")}
    key = units["pooled__n60__r0__lightgbm_tuned"]["samples"]
    assert units["pooled__n60__r0__tabpfn"]["samples"] == key == units["pooled__n60__r0__tabicl"]["samples"]
    assert units["site_local__n120__r1__tabpfn+foreign"]["n_context"] == 170
    assert units["site_local__n120__r1__tabicl"]["n_context"] == 120
    assert units["pooled__n60__r0__tabpfn"]["info"]["weights_override"] == "SOR_AI_TFM_FAKE=1"
    tuned = units["pooled__n120__r0__lightgbm_tuned"]["info"]
    assert tuned["n_trials"] == 2 and 2 <= tuned["k_folds"] <= 5 and tuned["n_estimators"] >= 1
    assert all(u["source_run"] == "none" for name, u in units.items() if "transported" not in name)
    trans = units["site_local__transported__lightgbm_transported"]
    assert set(trans["variants"]) == {"raw", "prior_known"} and "domyślne" in trans["info"]["params_note"]
    preds = np.load(root / units["pooled__n60__r1__tabicl"]["preds"])
    assert preds.shape == (len(val), 5) and np.allclose(preds.sum(axis=1), 1, atol=1e-4)

    # Status (orkiestrator): plan.json z source_run, wszystkie jednostki gotowe
    progress = analysis_progress(root)
    assert progress["total"] == results["n_units"] == progress["done"] and progress["complete"]

    # Statystyka i raport
    pooled = results["scenarios"]["pooled"]
    rows = [c for c in pooled["comparisons"] if c["arm"] == "tabpfn" and c["metric"] == "log_loss"]
    assert sorted(c["n"] for c in rows) == [60, 120] and all(c["R"] == 2 and c["p_holm"] is not None for c in rows)
    local = results["scenarios"]["site_local"]
    pairs = {(c["arm"], c["base"]) for c in local["comparisons"]}
    assert ("tabpfn+foreign", sd.BASELINE) in pairs and ("tabpfn+foreign", "tabpfn") in pairs
    report = (root / "report.md").read_text(encoding="utf-8")
    for fragment in ("## Decyzja", "## Akapit metod (do pracy)", "## Zastrzeżenia", "Clopper–Pearsona",
                     "ATRAPA estymatora", "Δ log loss vs LightGBM strojony", "pre-rejestracja",
                     "LightGBM przeniesiony (korekta priorów)", "Tylko"):
        assert fragment in report, fragment
    assert "Eksperyment A" in (root / "summary.md").read_text(encoding="utf-8")
    table = pd.read_csv(root / "units.csv")
    assert set(table["status"]) == {"done"} and "log_loss" in table.columns

    # Wznowienie: nic nie jest liczone ponownie; brak pliku predykcji → tylko ta jednostka od nowa
    def boom(*args, **kwargs):
        raise AssertionError("jednostka liczona ponownie")

    monkeypatch.setattr(sd, "tune_lightgbm_small", boom)
    monkeypatch.setattr(sd, "fit_predict_tfm", boom)
    monkeypatch.setattr(sd, "fit_predict_lgbm", boom)
    monkeypatch.setattr(sd, "fit_predict_transported", boom)
    sample_before = (root / "samples" / "pooled_n60_r0.npy").read_bytes()
    again = sd.run_small_data(small_cfg())
    assert again["failures"] == [] and again["n_done"] == results["n_units"]
    assert (root / "samples" / "pooled_n60_r0.npy").read_bytes() == sample_before
    pooled_again = again["scenarios"]["pooled"]["comparisons"]
    assert [c["delta"] for c in pooled_again] == [c["delta"] for c in pooled["comparisons"]]

    calls = []

    def counting_tfm(model_name, *args, **kwargs):
        calls.append(model_name)
        return np.full((len(args[2]), 5), 0.2), {"fake": True}

    monkeypatch.setattr(sd, "fit_predict_tfm", counting_tfm)
    (root / units["pooled__n120__r1__tabicl"]["preds"]).unlink()
    third = sd.run_small_data(small_cfg())
    assert calls == ["tabicl"] and third["complete"]


def test_cumulative_models_and_missing_tfm(prepared_paths, fake_tfm, monkeypatch, capsys):
    """Najpierw tylko TabPFN, potem tylko TabICL — wynik zawiera oba; brak zależności → pominięty TFM, LightGBM liczony."""
    cfg = dict(scenarios=["pooled"], sizes=[60], repeats=[2], foreign_rows=0, transport_rows=0, n_boot=20)
    first = sd.run_small_data(small_cfg(models=["tabpfn"], **cfg))
    assert "tabpfn" in first["scenarios"]["pooled"]["arms"]
    second = sd.run_small_data(small_cfg(models=["tabicl"], **cfg))
    assert {"tabpfn", "tabicl", *sd.LGBM_ARMS} == set(second["scenarios"]["pooled"]["arms"])
    assert second["complete"] and second["n_units"] == 2 * 4

    real_status = sd.tfm_status

    def no_torch(model, device="auto"):
        out = real_status(model, device)
        return {**out, "available": False, "reason": "brak pakietu torch (test)"}

    monkeypatch.setattr(sd, "tfm_status", no_torch)
    script = _load_script()
    code = script.run(["--tag", TAG, "--scenarios", "pooled", "--sizes", "60", "--repeats", "2", "--eval-rows", "0",
                       "--lgbm-trials", "2", "--bootstrap", "20", "--models", "tabpfn", "--foreign-rows", "0",
                       "--transport-rows", "0"])
    out = capsys.readouterr().out
    assert code == 0 and "skipped: tabpfn — brak pakietu torch (test)" in out
    results = _read(_root(prepared_paths) / "results.json")
    assert results["skipped_models"] == {"tabpfn": "brak pakietu torch (test)"}
    assert set(results["scenarios"]["pooled"]["arms"]) >= set(sd.LGBM_ARMS)
    report = (_root(prepared_paths) / "report.md").read_text(encoding="utf-8")
    assert "brak pakietu torch (test)" in report and "| TabPFN-3.5 | NIE |" in report


def test_plan_and_cli_errors(prepared_paths, fake_tfm, capsys):
    script = _load_script()
    assert script.run(["--tag", TAG, "--sizes", "60,5000", "--repeats", "3,1", "--plan"]) == 0
    out = capsys.readouterr().out
    assert "Plan eksperymentu A" in out and "RAZEM" in out and "lightgbm_tuned" in out
    assert not _root(prepared_paths).exists()                      # plan niczego nie zapisuje
    assert script.run(["--tag", TAG, "--models", "logreg"]) == 2
    assert script.run(["--tag", TAG, "--sizes", "60", "--repeats", "1,2"]) == 2
    assert script.run(["--tag", TAG, "--local-site", "Z", "--scenarios", "site_local", "--sizes", "60",
                       "--repeats", "1"]) == 2
    with pytest.raises(SystemExit):
        script.run(["--device", "tpu"])


def test_transported_uses_latest_tuned_lightgbm_run(prepared_paths, fake_tfm):
    from src.models.train import run_training

    result = run_training("lightgbm", profile="smoke", tag=TAG)
    run_dir = Path(result["run_dir"])
    for name in ("run_info.json", "DONE.json"):
        data = _read(run_dir / name)
        data["tuned"] = True
        (run_dir / name).write_text(json.dumps(data), encoding="utf-8")
    results = sd.run_small_data(small_cfg(models="none", scenarios=["site_local"], sizes=[60], repeats=[1],
                                          foreign_rows=0, n_boot=0))
    block = results["scenarios"]["site_local"]
    assert block["transported"]["source_run"] == run_dir.name and run_dir.name in block["transported"]["params_note"]
    plan = _read(_root(prepared_paths) / "plan.json")
    trans = [u for u in plan["units"] if u["arm"] == sd.TRANSPORTED]
    assert trans[0]["source_run"] == run_dir.name
    priors = block["transported"]["priors"]
    assert np.isclose(sum(priors["local"]), 1) and np.isclose(sum(priors["train"]), 1)


# ─────────────────────────────────────────
# Prawdziwe biblioteki (maleńkie losowe checkpointy) — pomijane bez torch / tabpfn / tabicl
# ─────────────────────────────────────────
@pytest.mark.skipif(not (HAS_TABPFN and HAS_TABICL), reason="brak torch / tabpfn / tabicl (requirements-tfm.txt)")
def test_real_libraries_tiny_run(prepared_paths, monkeypatch, tmp_path):
    import torch

    torch.set_num_threads(2)
    monkeypatch.delenv("SOR_AI_TFM_FAKE", raising=False)
    monkeypatch.setenv("SOR_AI_DEVICE", "cpu")
    monkeypatch.delenv("TABPFN_MODEL_CACHE_SIZE", raising=False)
    monkeypatch.setenv("SOR_AI_TABPFN_MODEL_PATH", str(write_tiny_tabpfn(tmp_path / "ckpt")))
    monkeypatch.setenv("SOR_AI_TABICL_MODEL_PATH", str(write_tiny_tabicl(tmp_path / "ckpt")))
    for name in ("TABPFN_DEFAULT_PARAMS", "TABICL_DEFAULT_PARAMS"):
        monkeypatch.setitem(getattr(config, name), "n_jobs", 2)
    results = sd.run_small_data(small_cfg(scenarios=["site_local"], sizes=[60], repeats=[1], foreign_rows=40,
                                          transport_rows=200, n_boot=20))
    assert results["failures"] == [] and results["complete"]
    status = results["tfm_status"]
    assert not status["tabpfn"]["fake"] and "SOR_AI_TABPFN_MODEL_PATH" in status["tabpfn"]["override"]
    root = _root(prepared_paths)
    units = {p.stem: _read(p) for p in (root / "units").glob("*.json")}
    pfn = units["site_local__n60__r0__tabpfn"]["info"]
    assert pfn["weights"]["file"] == "tabpfn-v3.5-random-tiny.safetensors" and pfn["runtime"]["fit_mode"] == "fit_preprocessors"
    assert units["site_local__n60__r0__tabicl+foreign"]["n_context"] == 100
    assert units["site_local__n60__r0__tabicl"]["info"]["runtime"]["kv_cache"] is False
    for name in ("site_local__n60__r0__tabpfn", "site_local__n60__r0__tabicl+foreign"):
        proba = np.load(root / units[name]["preds"])
        assert np.isfinite(proba).all() and np.allclose(proba.sum(axis=1), 1, atol=1e-4)
    import os

    assert "TABPFN_MODEL_CACHE_SIZE" not in os.environ                   # przywrócone po grupie TabPFN
    assert "PRZEKIEROWANIE WAG" in (root / "report.md").read_text(encoding="utf-8")


# ─────────────────────────────────────────
# Raport (na ręcznie zbudowanych wynikach)
# ─────────────────────────────────────────
def _cmp(arm, base, n, verdict, delta, metric="log_loss"):
    return {"arm": arm, "base": base, "n": n, "metric": metric, "R": 3, "delta": delta, "sd": 0.01,
            "t_ci_low": delta - 0.01, "t_ci_high": delta + 0.01, "p_t": 0.01, "boot_ci_low": delta - 0.005,
            "boot_ci_high": delta + 0.005, "p_boot": 0.002, "p_holm": 0.006, "verdict": verdict,
            "verdict_text": sd.verdict(delta, 0.006, metric)[1], "mid": 0.005}


def test_report_rendering_and_decision():
    metrics = {key: {"mean": 0.5, "sd": 0.01, "ci_low": 0.49, "ci_high": 0.51, "n": 3} for key in sd.TABLE_METRICS}
    esi1 = [{"n": 68, "k_argmax": 30, "k_cost": 50, "critical_cost": 2}] * 3
    points = {arm: {str(n): {"R": 3, "metrics": metrics, "esi1_counts": esi1, "repeats": [0, 1, 2]}
                    for n in (500, 5000, 50000)} for arm in (sd.BASELINE, "tabpfn", "tabpfn+foreign")}
    comparisons = [_cmp("tabpfn", sd.BASELINE, 500, "win", -0.03), _cmp("tabpfn", sd.BASELINE, 5000, "win", -0.01),
                   _cmp("tabpfn", sd.BASELINE, 50000, "loss", 0.02),
                   _cmp("tabpfn+foreign", "tabpfn", 500, "loss", 0.02), _cmp("tabpfn+foreign", "tabpfn", 5000, "tie", 0.0),
                   _cmp("tabpfn", sd.BASELINE, 500, "win", 0.03, "qwk_argmax")]
    fit = {"a": 0.7, "b": 2.0, "c": 0.5, "rmse": 0.001, "reliable": True, "reasons": []}
    block = {"arms": [sd.BASELINE, "tabpfn", "tabpfn+foreign"], "points": points, "comparisons": comparisons,
             "power_law": {sd.BASELINE: fit, "tabpfn": fit, "tabpfn+foreign": None},
             "crossover": {f"tabpfn|{sd.BASELINE}": {"status": "arm_better_below", "n_star": 12_345.0,
                                                     "ci_low": 8_000.0, "ci_high": 20_000.0, "in_range": True,
                                                     "agreement": 0.93}},
             "eval": {"n": 16_633, "n_esi1": 68}, "source_rows": 133_068, "figures": ["figures/curves_site_local.png"],
             "transported": {"variants": {"raw": {"log_loss": 0.9}, "prior_known": {"log_loss": 0.85}},
                             "esi1_counts": {"raw": {"n": 68, "k_argmax": 20, "k_cost": 40, "critical_cost": 1}},
                             "n_fit": 90_000, "n_es": 10_000, "params_note": "parametry runa X"}}
    cfg = {**sd.asdict(sd.SmallDataConfig()), "models_requested": ["tabpfn", "tabicl"], "models": ["tabpfn"]}
    results = {"feature_set": "triage", "tag": "thorough", "created": "2026-10-26", "data_fingerprint": "abc",
               "config": cfg, "n_units": 10, "n_done": 10, "failures": [], "scenarios": {"site_local": block},
               "tfm_status": {"tabpfn": {"available": True, "device": "cuda", "library_version": "9.2.0",
                                         "weights": {"file": "tabpfn-v3.5.ckpt", "sha256": "0123456789abcdef"}},
                              "tabicl": {"available": False, "reason": "brak wag — 13_tfm_setup"}},
               "skipped_models": {"tabicl": "brak wag — 13_tfm_setup"},
               "skipped_units": [{"unit": "x", "reason": "CPU: kontekst"}]}
    decision = sd.decision_lines(results)
    assert any("TFM wygrywa poniżej n ≈ 50 000" in line and "n* ≈ 12 345" in line and "95% CI 8 000–20 000" in line
               for line in decision)
    assert any("SZKODZĄ (transfer negatywny) przy n ∈ {500}" in line for line in decision)
    text = "\n".join(sd.report_lines(results))
    assert "| TabICLv2 | NIE |" in text and "brak wag — 13_tfm_setup" in text
    assert "> ⚠ Tylko 68 wizyt ESI 1" in text and "0.735 [" in text          # 50/68 z Clopper–Pearsonem
    assert "— (n < 5000)" in text and "zgodność bootstrapu 93%" in text
    assert "LightGBM przeniesiony (korekta priorów)" in text and "![site_local](figures/curves_site_local.png)" in text
    assert "## Akapit metod (do pracy)" in text and "Holma" in text and "Clopper–Pearsona" in text
    summary = "\n".join(sd.summary_lines(results))
    assert "pominięto TabICLv2: brak wag" in summary


def test_kept_units_require_same_weights(prepared_paths, fake_tfm, monkeypatch):
    """Jednostki TFM pominięte teraz (CPU) zostają z poprzedniego przebiegu (GPU) — ale nie przy innych wagach."""
    cfg = dict(scenarios=["pooled"], sizes=[60], repeats=[2], foreign_rows=0, transport_rows=0, n_boot=0,
               models=["tabpfn"])
    first = sd.run_small_data(small_cfg(device="gpu", **cfg))                  # atrapa „na GPU”: bez limitu CPU
    assert first["complete"] and "tabpfn" in first["scenarios"]["pooled"]["arms"]
    monkeypatch.setattr(config, "TFM_CPU_MAX_CONTEXT", 50)
    second = sd.run_small_data(small_cfg(device="cpu", **cfg))
    assert any("TFM_CPU_MAX_CONTEXT" in s["reason"] for s in second["skipped_units"])
    assert "tabpfn" in second["scenarios"]["pooled"]["arms"] and second["complete"]   # wyniki z GPU zachowane
    real_status = sd.tfm_status

    def other_weights(model, device="auto"):
        out = real_status(model, device)
        return {**out, "weights": {"file": "inne-wagi.ckpt", "sha256": "f" * 64, "size": 1}}

    monkeypatch.setattr(sd, "tfm_status", other_weights)
    third = sd.run_small_data(small_cfg(device="cpu", **cfg))
    assert "tabpfn" not in third["scenarios"]["pooled"]["arms"]                     # inne wagi → nie mieszamy
