"""
Testy eksperymentu B (`src/analysis/tfm_blend.py`, `scripts/15_tfm_blend.py`) na syntetycznych runach
(predykcje w formacie `train._predictions_frame`): TFM pomocny (bramka przechodzi), TFM bezużyteczny
(bramka nie przechodzi), reguła oszczędności, brak SD seedów (nierozstrzygnięte), pominięcia (brak
lightgbm / TFM → `skipped:` i kod 0), wznawianie powtórzeń, jawne --runs, odniesienie klasyczne, postęp dla
orkiestratora (plan.json + units) i polecenie 08 z recommendation.json wykonane naprawdę.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import tfm_blend as tb
from src.analysis.common import METRICS_VERSION, AnalysisArgumentError, analysis_progress
from src.utils.config import N_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAG = "standard"
N_VAL = 1600
LGBM = f"20260301_000000_lightgbm_triage_{TAG}"
PFN = f"20260301_000100_tabpfn_triage_{TAG}"
ICL = f"20260301_000200_tabicl_triage_{TAG}"
XGB = f"20260301_000300_xgboost_triage_{TAG}"
FAST = {"repeats": 2, "n_boot": 120, "plots": False}


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def make_world(n: int = N_VAL, seed: int = 0) -> dict[str, np.ndarray]:
    """
    Prawdziwe logity = offset + a. LightGBM i „pomocny” TFM widzą a z NIEZALEŻNYM szumem (łączenie uśrednia
    szum → wyraźny spadek log loss bez psucia kalibracji), „bezużyteczny” TFM — sam szum.
    """
    rng = np.random.default_rng(seed)
    offset = np.log(np.array([0.06, 0.28, 0.38, 0.21, 0.07]))
    a = rng.normal(0, 1.6, size=(n, N_CLASSES))
    y = np.array([rng.choice(N_CLASSES, p=p) for p in _softmax(offset + a)])

    def noisy(sd: float) -> np.ndarray:
        return _softmax(offset + a + rng.normal(0, sd, size=(n, N_CLASSES)))

    return {
        "y": y,
        "lightgbm": noisy(1.0),
        "helpful": noisy(1.0),
        "useless": _softmax(offset + rng.normal(0, 1.0, size=(n, N_CLASSES))),
        "xgboost": noisy(1.2),
    }


def write_run(root: Path, name: str, model: str, proba: np.ndarray, y: np.ndarray, *, tag: str = TAG,
              pretrained: bool = False, row_ids: np.ndarray | None = None, **info_extra) -> Path:
    """Run w formacie run_training: run_info.json, DONE.json, predictions_{val,test}.parquet (_predictions_frame)."""
    from src.models.train import _predictions_frame

    run_dir = root / name
    run_dir.mkdir(parents=True)
    n = len(y)
    ids = np.arange(n) if row_ids is None else row_ids
    meta = pd.DataFrame({"row_id": ids, "site": np.where(np.arange(n) % 3 == 0, "B", "A"), "esi": y + 1,
                         "target": y, "vitals_complete": (np.arange(n) % 4 != 0).astype(np.int8)})
    for split in ("val", "test"):
        _predictions_frame(meta, proba).to_parquet(run_dir / f"predictions_{split}.parquet", index=False)
    from src.utils.config import DATA_VERSION

    info = {"run_name": name, "model": model, "feature_set": "triage", "tag": tag, "profile": tag, "tuned": True,
            "custom": False, "weights_strategy": "none", "data_version": DATA_VERSION, "data_fingerprint": "fp-test",
            "train_sample": None, "eval_sample": None, **info_extra}
    if pretrained:
        info["pretrained"] = True
    (run_dir / "run_info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    (run_dir / "DONE.json").write_text(json.dumps({"run_name": name, "model": model}), encoding="utf-8")
    (run_dir / "model.joblib").write_bytes(b"fake")
    return run_dir


def write_seeds(root: Path, sd: float | None = 0.0008, lgbm_run: str = LGBM, tag: str = TAG) -> Path:
    """runs/analyses/seeds__triage__<tag>/results.json z SD log loss lightgbm na VAL."""
    path = root / "analyses" / f"seeds__triage__{tag}" / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    stat = {"mean": 0.9, "sd": sd, "n": 4 if sd is not None else 1, "values": [0.9]}
    payload = {"analysis": "seeds", "feature_set": "triage", "tag": tag, "metrics_version": METRICS_VERSION,
               "data_fingerprint": "fp-test", "source_runs": {"lightgbm": lgbm_run},
               "models": {"lightgbm": {"runs": [], "n_runs": stat["n"], "val": {"log_loss": stat},
                                       "test": {"log_loss": stat}}}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def world():
    return make_world()


@pytest.fixture
def runs(isolated_paths, world):
    """lightgbm + TabPFN pomocny + TabICL bezużyteczny + wyniki seedów."""
    root = isolated_paths.RUNS_DIR
    write_run(root, LGBM, "lightgbm", world["lightgbm"], world["y"])
    write_run(root, PFN, "tabpfn", world["helpful"], world["y"], pretrained=True)
    write_run(root, ICL, "tabicl", world["useless"], world["y"], pretrained=True)
    write_seeds(root)
    return root


def _tables_well_formed(markdown: str) -> None:
    """Każdy wiersz tabeli Markdown ma tyle kolumn co nagłówek (niezescapowane „|”)."""
    import re

    def cells(line: str) -> int:
        return len(re.findall(r"(?<!\\)\|", line))

    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|---"):
            width = cells(lines[i - 1])
            assert cells(line) == width, f"zły separator: {line!r}"
            j = i + 1
            while j < len(lines) and lines[j].startswith("|"):
                assert cells(lines[j]) == width, f"zła tabela: {lines[j]!r} (nagłówek: {lines[i - 1]!r})"
                j += 1


def _cfg(**kw) -> tb.BlendConfig:
    return tb.BlendConfig(**{"tag": TAG, **FAST, **kw})


# ─────────────────────────────────────────
# Funkcje pomocnicze
# ─────────────────────────────────────────
class TestHelpers:
    def test_parse_methods(self):
        assert tb.parse_methods("mean, geometric,mean") == ["mean", "geometric"]
        assert tb.parse_methods(["geometric"]) == ["geometric"]
        with pytest.raises(AnalysisArgumentError):
            tb.parse_methods("median")
        with pytest.raises(AnalysisArgumentError):
            tb.parse_methods("")

    def test_parse_tfm_models(self):
        assert tb.parse_tfm_models("tfm") == ["tabpfn", "tabicl"]
        assert tb.parse_tfm_models("icl,lightgbm") == ["tabicl"]
        assert tb.parse_tfm_models("lightgbm,tabpfn,tabicl") == ["tabpfn", "tabicl"]
        assert tb.parse_tfm_models("all") == []            # 'all' = 6 modeli klasycznych
        assert tb.parse_tfm_models("none") == []
        with pytest.raises(AnalysisArgumentError):
            tb.parse_tfm_models("gpt")

    def test_fold_strata_merges_rare_combinations(self):
        y = np.array([0, 0, 1, 1, 1, 2])
        site = np.array(["A", "B", "A", "A", "B", "A"])
        labels = tb.fold_strata(y, site)
        assert list(labels) == ["0", "0", "1_A", "1_A", "1", "2"]
        assert list(tb.fold_strata(y, None)) == ["0", "0", "1", "1", "1", "2"]

    def test_delta_summary_direction_and_ci(self):
        rng = np.random.default_rng(0)
        better = tb.delta_summary(-0.02, rng.normal(-0.02, 0.003, 500), 0.05, higher_is_better=False)
        assert better["ci_high"] < 0 and better["better"] == "zestaw"
        worse = tb.delta_summary(0.02, rng.normal(0.02, 0.003, 500), 0.05, higher_is_better=False)
        assert worse["better"] == "LightGBM"
        qwk = tb.delta_summary(0.02, rng.normal(0.02, 0.003, 500), 0.05, higher_is_better=True)
        assert qwk["better"] == "zestaw"
        null = tb.delta_summary(0.0, rng.normal(0, 0.01, 500), 0.05, higher_is_better=False)
        assert null["better"] == "brak istotnej różnicy" and null["ci_low"] < 0 < null["ci_high"]
        assert tb.delta_summary(None, np.array([]), 0.05, False)["better"] == "nieokreślone"

    def test_build_variants_max_two_tfms_and_reference(self):
        class R:
            def __init__(self, name):
                self.name = name

        sel = tb.Selection(lightgbm=R("l"), tfms={"tabpfn": R("p"), "tabicl": R("i")}, references={"xgboost": R("x")})
        variants = tb.build_variants(sel, ["mean", "geometric"])
        ids = [v.id for v in variants]
        assert ids[0] == tb.BASELINE
        assert {"lightgbm+tabpfn|mean", "lightgbm+tabpfn|geometric", "lightgbm+tabicl|mean",
                "lightgbm+tabpfn+tabicl|geometric", "tabpfn", "tabicl", "classical|mean"} <= set(ids)
        assert all(sum(m in ("tabpfn", "tabicl") for m in v.members) <= tb.MAX_TFMS for v in variants)
        classical = next(v for v in variants if v.set_id == tb.CLASSICAL_SET)
        assert classical.members == ("lightgbm", "xgboost") and classical.kind == "reference"
        sel.references = {}
        sel.tfms = {"tabicl": R("i")}
        ids = [v.id for v in tb.build_variants(sel, ["mean"])]
        assert ids == ["lightgbm", "lightgbm+tabicl|mean", "tabicl"]


def _deltas(ll: float, hi: float, under: float = 0.0, ece1: float = 0.0) -> dict:
    return {"log_loss": {"delta": ll, "ci_low": ll - 0.002, "ci_high": hi},
            "undertriage_target": {"delta": under}, "ece_esi1": {"delta": ece1}}


class TestGate:
    variant = tb.Variant("lightgbm+tabpfn|mean", "lightgbm+tabpfn", ("lightgbm", "tabpfn"), "mean", "candidate")

    def status(self, deltas: dict, sd: float | None = 0.001) -> tuple[str, dict]:
        gate = tb.evaluate_gate(self.variant, deltas, sd, 0.005)
        return gate["status"], {c["key"]: c["status"] for c in gate["criteria"]}

    def test_all_criteria_pass(self):
        status, crit = self.status(_deltas(-0.02, -0.015))
        assert status == tb.PASSED and set(crit.values()) == {tb.PASS}

    @pytest.mark.parametrize("deltas, failed", [
        (_deltas(-0.004, -0.002), "delta_log_loss"),           # poniżej MID
        (_deltas(-0.006, 0.001), "ci_upper"),                   # CI zawiera 0
        (_deltas(-0.02, -0.015, under=0.011), "undertriage_target"),
        (_deltas(-0.02, -0.015, ece1=0.0101), "ece_esi1"),
    ])
    def test_each_criterion_can_fail(self, deltas, failed):
        status, crit = self.status(deltas)
        assert status == tb.FAILED and crit[failed] == tb.FAIL

    def test_seed_noise(self):
        status, crit = self.status(_deltas(-0.006, -0.002), sd=0.004)      # |Δ| ≤ 2 × SD
        assert status == tb.FAILED and crit["seed_noise"] == tb.FAIL

    def test_missing_seed_sd_is_inconclusive(self):
        status, crit = self.status(_deltas(-0.02, -0.015), sd=None)
        assert status == tb.INCONCLUSIVE and crit["seed_noise"] == tb.NOT_CHECKED
        status, _ = self.status(_deltas(-0.001, 0.002), sd=None)          # inne kryterium niespełnione
        assert status == tb.FAILED

    def test_equal_to_mid_passes(self):
        status, crit = self.status(_deltas(-0.005, -0.001), sd=0.0001)
        assert crit["delta_log_loss"] == tb.PASS and status == tb.PASSED


def _gate(sid: str, status: str, n_tfm: int) -> dict:
    return {"status": status, "n_tfm": n_tfm, "variant": f"{sid}|mean"}


class TestParsimony:
    def setup_choice(self, d_pfn: float, d_icl: float, d_pair: float, statuses: dict[str, str]):
        sets = {"lightgbm+tabpfn": 1, "lightgbm+tabicl": 1, "lightgbm+tabpfn+tabicl": 2}
        best = {sid: tb.Variant(f"{sid}|mean", sid, tuple(sid.split("+")), "mean", "candidate") for sid in sets}
        dl = {"lightgbm+tabpfn": d_pfn, "lightgbm+tabicl": d_icl, "lightgbm+tabpfn+tabicl": d_pair}
        deltas = {f"{sid}|mean": {"log_loss": {"delta": d}} for sid, d in dl.items()}
        rng = np.random.default_rng(1)
        samples = {f"{sid}|mean": {"log_loss": d + rng.normal(0, 0.001, 200)} for sid, d in dl.items()}
        gates = {sid: _gate(sid, statuses.get(sid, tb.PASSED), n) for sid, n in sets.items()}
        return tb.choose(gates, deltas, samples, best, 0.005)

    def test_pair_needs_mid_over_best_single(self):
        choice = self.setup_choice(-0.02, -0.01, -0.023, {})
        assert choice["chosen_set"] == "lightgbm+tabpfn" and not choice["pair_chosen"]
        assert choice["pair_vs_single"]["delta"] == pytest.approx(-0.003)
        assert "oszczędności" in choice["reason"]

    def test_pair_chosen_when_much_better(self):
        choice = self.setup_choice(-0.02, -0.01, -0.03, {})
        assert choice["chosen_set"] == "lightgbm+tabpfn+tabicl" and choice["pair_chosen"]

    def test_best_passing_single(self):
        choice = self.setup_choice(-0.02, -0.03, -0.031, {"lightgbm+tabicl": tb.FAILED,
                                                          "lightgbm+tabpfn+tabicl": tb.FAILED})
        assert choice["chosen_set"] == "lightgbm+tabpfn" and choice["variant"] == "lightgbm+tabpfn|mean"

    def test_only_pair_passes(self):
        choice = self.setup_choice(-0.002, -0.002, -0.02, {"lightgbm+tabpfn": tb.FAILED,
                                                           "lightgbm+tabicl": tb.FAILED})
        assert choice["chosen_set"] == "lightgbm+tabpfn+tabicl"

    def test_nothing_passes(self):
        choice = self.setup_choice(-0.02, -0.01, -0.03, {s: tb.INCONCLUSIVE for s in
                                                         ("lightgbm+tabpfn", "lightgbm+tabicl", "lightgbm+tabpfn+tabicl")})
        assert choice["chosen_set"] == tb.BASELINE and choice["variant"] == tb.BASELINE
        assert "nierozstrzygnięte" in choice["reason"]


# ─────────────────────────────────────────
# Pełny przebieg
# ─────────────────────────────────────────
class TestEndToEnd:
    def test_helpful_passes_useless_fails(self, runs):
        results = tb.run_tfm_blend(_cfg(plots=True))
        assert results["complete"] and results["n_done"] == 2
        gate = results["gate"]
        assert gate["lightgbm+tabpfn"]["status"] == tb.PASSED
        assert gate["lightgbm+tabicl"]["status"] == tb.FAILED
        # Para: TabICL nic nie wnosi → reguła oszczędności wybiera jeden TFM
        assert results["parsimony"]["chosen_set"] == "lightgbm+tabpfn"
        assert not results["parsimony"]["pair_chosen"]
        d = results["deltas"][gate["lightgbm+tabpfn"]["variant"]]["log_loss"]
        assert d["delta"] < -0.05 and d["ci_high"] < 0
        assert results["deltas"]["tabicl"]["log_loss"]["delta"] > 0      # sam bezużyteczny TFM — gorszy

        root = tb.analysis_path("triage", TAG)
        rec = json.loads((root / "recommendation.json").read_text(encoding="utf-8"))
        assert rec["runs"] == [LGBM, PFN] and rec["includes_tfm"] and rec["noise_criterion_checked"]
        assert rec["command"].startswith(f"python scripts/08_final_report.py --feature-set triage --runs {LGBM} {PFN}")
        assert f"--method {rec['method']}" in rec["command"]
        assert "--include-pretrained" in rec["command_tag"] and "--models lightgbm,tabpfn" in rec["command_tag"]

        report = (root / "report.md").read_text(encoding="utf-8")
        assert report.index("## 1. Kryteria ustalone z góry") < report.index("## 3. Wyniki")
        assert "Tylko VAL" in report and rec["command"] in report
        _tables_well_formed(report)
        _tables_well_formed((root / "summary.md").read_text(encoding="utf-8"))
        assert {p.name for p in (root / "figures").glob("*.png")} == {"delta_forest.png", "diversity.png",
                                                                      "weights_folds.png"}
        stored = json.loads((root / "results.json").read_text(encoding="utf-8"))
        assert stored["seed_sd"]["sd"] == pytest.approx(0.0008)
        assert stored["folds"]["lightgbm+tabpfn|mean"]["weights"]["tabpfn"]["mean"] > 0.2
        assert (root / "summary.md").exists()
        # Postęp dla orkiestratora (run_all --status): plan.json + jednostki z source_run
        progress = analysis_progress(root)
        assert progress["total"] == 2 and progress["done"] == 2 and progress["complete"]

    def test_missing_seed_sd_is_inconclusive_and_not_recommended(self, runs):
        (runs / "analyses" / f"seeds__triage__{TAG}" / "results.json").unlink()
        results = tb.run_tfm_blend(_cfg(models="tabpfn"))
        assert results["gate"]["lightgbm+tabpfn"]["status"] == tb.INCONCLUSIVE
        rec = results["recommendation"]
        assert rec["decision"] == tb.BASELINE and rec["runs"] == [LGBM] and not rec["noise_criterion_checked"]
        assert "brak" in results["seed_sd"]["reason"]
        report = (tb.analysis_path("triage", TAG) / "report.md").read_text(encoding="utf-8")
        assert "09_seed_refits.py --feature-set triage --tag standard --models lightgbm" in report

    def test_seeds_from_other_lightgbm_run_are_ignored(self, runs):
        write_seeds(runs, sd=0.0008, lgbm_run="20250101_000000_lightgbm_triage_standard")
        results = tb.run_tfm_blend(_cfg(models="tabpfn"))
        assert results["seed_sd"]["sd"] is None
        assert results["gate"]["lightgbm+tabpfn"]["status"] == tb.INCONCLUSIVE

    def test_large_seed_sd_fails_noise_criterion(self, runs):
        write_seeds(runs, sd=0.5)
        results = tb.run_tfm_blend(_cfg(models="tabpfn"))
        crit = {c["key"]: c["status"] for c in results["gate"]["lightgbm+tabpfn"]["criteria"]}
        assert crit["seed_noise"] == tb.FAIL and results["recommendation"]["decision"] == tb.BASELINE

    def test_resume_reuses_repeats_and_force_recomputes(self, runs, monkeypatch):
        first = tb.run_tfm_blend(_cfg(models="tabpfn"))
        calls = []
        original = tb.crossfit_repeat

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(tb, "crossfit_repeat", spy)
        second = tb.run_tfm_blend(_cfg(models="tabpfn"))
        assert calls == []
        assert second["deltas"] == first["deltas"]
        # Usunięty plik predykcji → to powtórzenie liczone ponownie
        (tb.analysis_path("triage", TAG) / "preds" / "repeat1.npz").unlink()
        tb.run_tfm_blend(_cfg(models="tabpfn"))
        assert len(calls) == 1
        tb.run_tfm_blend(_cfg(models="tabpfn", force=True))
        assert len(calls) == 3

    def test_classical_reference(self, runs, world):
        write_run(runs, XGB, "xgboost", world["xgboost"], world["y"])
        results = tb.run_tfm_blend(_cfg(models="tabpfn", methods="mean"))
        ref = next(v for v in results["variants"] if v["set"] == tb.CLASSICAL_SET)
        assert ref["members"] == ["lightgbm", "xgboost"]
        assert results["references"] == {"xgboost": XGB}
        assert results["classical_superiority"]
        assert tb.CLASSICAL_SET not in results["gate"]                 # tylko odniesienie
        assert any(r["model_b"] == "xgboost" for r in results["diversity"])

    def test_reference_with_other_rows_is_dropped(self, runs, world):
        write_run(runs, XGB, "xgboost", world["xgboost"][:-10], world["y"][:-10])
        results = tb.run_tfm_blend(_cfg(models="tabpfn"))
        assert results["references"] == {}
        assert any("odniesienie klasyczne pominięte" in n for n in results["notes"])

    def test_plan_only_writes_nothing(self, runs):
        out = tb.run_tfm_blend(_cfg(), plan_only=True)
        assert any("warianty" in line for line in out["plan_lines"])
        assert not tb.analysis_path("triage", TAG).exists()


class TestRunSelection:
    def test_no_lightgbm_skips(self, isolated_paths, world):
        write_run(isolated_paths.RUNS_DIR, PFN, "tabpfn", world["helpful"], world["y"], pretrained=True)
        with pytest.raises(tb.BlendSkipped, match="lightgbm"):
            tb.run_tfm_blend(_cfg())

    def test_no_tfm_skips(self, isolated_paths, world):
        write_run(isolated_paths.RUNS_DIR, LGBM, "lightgbm", world["lightgbm"], world["y"])
        with pytest.raises(tb.BlendSkipped, match="TFM"):
            tb.run_tfm_blend(_cfg())

    def test_untuned_weighted_custom_and_other_tag_runs_ignored(self, isolated_paths, world):
        root = isolated_paths.RUNS_DIR
        write_run(root, LGBM, "lightgbm", world["lightgbm"], world["y"])
        write_run(root, "20260302_000000_tabpfn_triage_standard", "tabpfn", world["helpful"], world["y"], tuned=False)
        write_run(root, "20260302_000100_tabpfn_triage_standard", "tabpfn", world["helpful"], world["y"],
                  weights_strategy="balanced")
        write_run(root, "20260302_000200_tabpfn_triage_standard", "tabpfn", world["helpful"], world["y"], custom=True)
        write_run(root, "20260302_000300_tabpfn_triage_quick", "tabpfn", world["helpful"], world["y"], tag="quick")
        found, _ = tb.latest_source_runs("triage", TAG, ["lightgbm", "tabpfn"])
        assert list(found) == ["lightgbm"]

    def test_newest_run_wins(self, runs, world):
        newer = "20260305_000000_tabpfn_triage_standard"
        write_run(runs, newer, "tabpfn", world["helpful"], world["y"], pretrained=True)
        sel = tb.select_runs(_cfg())
        assert sel.tfms["tabpfn"].name == newer and sel.tfms["tabicl"].name == ICL

    def test_explicit_runs(self, runs):
        results = tb.run_tfm_blend(_cfg(tag=None, runs=[LGBM, ICL]))
        assert results["runs"] == {"lightgbm": LGBM, "tabicl": ICL} and results["tag"] == TAG
        assert results["recommendation"]["decision"] == tb.BASELINE

    @pytest.mark.parametrize("names, error", [
        ([LGBM, LGBM], AnalysisArgumentError),
        ([PFN], tb.BlendSkipped),
        (["brak_takiego_runu"], AnalysisArgumentError),
    ])
    def test_explicit_runs_refusals(self, runs, names, error):
        with pytest.raises(error):
            tb.select_runs(_cfg(runs=names))

    def test_classical_model_in_runs_refused(self, runs, world):
        write_run(runs, XGB, "xgboost", world["xgboost"], world["y"])
        with pytest.raises(AnalysisArgumentError, match="xgboost"):
            tb.select_runs(_cfg(runs=[LGBM, PFN, XGB]))

    def test_misaligned_rows_refused(self, isolated_paths, world):
        root = isolated_paths.RUNS_DIR
        write_run(root, LGBM, "lightgbm", world["lightgbm"], world["y"])
        write_run(root, PFN, "tabpfn", world["helpful"][:-5], world["y"][:-5], pretrained=True)
        with pytest.raises(AnalysisArgumentError, match="wyrównane"):
            tb.run_tfm_blend(_cfg())

    @pytest.mark.parametrize("kw", [{"repeats": 0}, {"n_boot": 0}, {"target": 1.5}, {"fit_population": "x"},
                                    {"methods": "median"}])
    def test_bad_config(self, runs, kw):
        with pytest.raises(AnalysisArgumentError):
            tb.run_tfm_blend(_cfg(**kw))


# ─────────────────────────────────────────
# Skrypt 15 i polecenie 08 z rekomendacji
# ─────────────────────────────────────────
class TestScript:
    def test_skipped_exit_zero(self, isolated_paths, capsys):
        module = _load_script("tfm_blend_script", "15_tfm_blend.py")
        assert module.run(["--feature-set", "triage", "--tag", TAG, "--models", "tabpfn,tabicl"]) == 0
        assert capsys.readouterr().out.startswith("skipped: brak ukończonego runu lightgbm")

    def test_bad_arguments_exit_two(self, runs):
        module = _load_script("tfm_blend_script", "15_tfm_blend.py")
        assert module.run(["--tag", TAG, "--methods", "median"]) == 2
        with pytest.raises(SystemExit):
            module.run(["--tag", TAG, "--target-undertriage", "2"])

    def test_plan(self, runs, capsys):
        module = _load_script("tfm_blend_script", "15_tfm_blend.py")
        assert module.run(["--tag", TAG, "--plan", "--repeats", "2"]) == 0
        out = capsys.readouterr().out
        assert "powtórzenia: 2" in out and LGBM in out
        assert "Bramka (ustalona z góry" in out and "2 × SD seedów LightGBM = 0.0016" in out

    def test_gate_logged_before_numbers(self, runs, caplog, monkeypatch):
        import logging

        monkeypatch.setattr(tb.log, "propagate", True)            # logger projektu nie propaguje do roota
        with caplog.at_level(logging.INFO, logger=tb.log.name):
            tb.run_tfm_blend(_cfg(models="tabpfn"))
        text = caplog.text
        assert text.index("Bramka (ustalona z góry") < text.index("log loss (cross-fitted)")

    def test_orchestrator_command_contract(self, runs):
        """Polecenie z `orchestrator.analysis_command` (lightgbm na liście modeli) → kod 0, wynik kompletny."""
        from src.pipeline import orchestrator

        cmd = orchestrator.analysis_command("tfm_blend", "triage", TAG, ["lightgbm", "tabpfn", "tabicl"])
        script = next(i for i, part in enumerate(cmd) if Path(part).name == "15_tfm_blend.py")
        module = _load_script("tfm_blend_script", "15_tfm_blend.py")
        assert module.run(cmd[script + 1:] + ["--repeats", "1", "--bootstrap", "30", "--no-plots"]) == 0
        root = orchestrator.analysis_path("tfm_blend", "triage", TAG)
        assert root == tb.analysis_path("triage", TAG)
        stored = json.loads((root / "results.json").read_text(encoding="utf-8"))
        assert stored["complete"] and set(stored["runs"]) == {"lightgbm", "tabpfn", "tabicl"}
        plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
        unit = json.loads((root / "units" / "repeat1.json").read_text(encoding="utf-8"))
        assert unit["source_run"] == plan["units"][0]["source_run"] == f"{LGBM}+{PFN}+{ICL}"
        assert (root / "summary.md").exists()
        assert analysis_progress(root)["complete"]

    def test_constant_tfm_gives_no_numpy_warnings(self, world):
        """TFM o stałych predykcjach (np. losowe wagi) → korelacje NaN (w raporcie „—”), bez ostrzeżeń numpy."""
        import warnings

        y = world["y"]
        constant = np.tile([0.0, 0.0, 1.0, 0.0, 0.0], (len(y), 1))      # E[ESI] stałe → korelacja 0/0
        data = tb.ValData(y=y, probas={"lightgbm": world["lightgbm"], "tabpfn": constant}, vc=None, site=None,
                          row_ids=np.arange(len(y)))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            rows = tb.diversity_rows(data, ["lightgbm", "tabpfn"])
        assert len(rows) == 1 and np.isnan(rows[0]["corr_expected"]) and np.isfinite(rows[0]["corr_errors"])

    def test_no_repeat_finished_exit_one(self, runs, monkeypatch):
        def boom(*args, **kwargs):
            raise ValueError("zepsute dane")

        monkeypatch.setattr(tb, "crossfit_repeat", boom)
        module = _load_script("tfm_blend_script", "15_tfm_blend.py")
        assert module.run(["--tag", TAG, "--models", "tabpfn", "--repeats", "1", "--bootstrap", "20",
                           "--no-plots"]) == 1
        progress = analysis_progress(tb.analysis_path("triage", TAG))
        assert progress["total"] == 1 and progress["done"] == 0

    def test_run_then_recommended_final_report(self, runs, capsys):
        module = _load_script("tfm_blend_script", "15_tfm_blend.py")
        code = module.run(["--feature-set", "triage", "--tag", TAG, "--models", "lightgbm,tabpfn,tabicl",
                           "--repeats", "2", "--bootstrap", "100", "--methods", "mean", "--no-plots",
                           "--target-undertriage", "0.2"])
        assert code == 0
        out = capsys.readouterr().out
        assert "Decyzja (VAL)" in out
        rec = json.loads((tb.analysis_path("triage", TAG) / "recommendation.json").read_text(encoding="utf-8"))
        assert "--target-undertriage 0.2" in rec["command"]
        args = rec["command_args"]
        assert args[:2] == ["python", "scripts/08_final_report.py"]
        final = _load_script("final_report_script_b", "08_final_report.py")
        out_dir = runs / "final_from_b"
        assert final.main(args[2:] + ["--bootstrap", "50", "--no-plots", "--out", str(out_dir)]) == 0
        recipe = json.loads((out_dir / "recipe.json").read_text(encoding="utf-8"))
        assert sorted(r["name"] for r in recipe["runs"]) == sorted(rec["runs"])
        no_tfm = json.loads((out_dir / "backend_ensemble_weights_no_tfm.json").read_text(encoding="utf-8"))
        assert set(no_tfm["weights"]) == {"lightgbm"}
