"""
Testy raportu końcowego `scripts/08_final_report.py` na fałszywych runach (format `run_training`,
helpery z tests/test_ensemble.py): pełny przebieg, plik wag zgodny z backendem (także ponowne
użycie z metodą / regułą / temperaturą z pliku), sekcja przyjęć, odmowy (niewyrównane wiersze,
inny zestaw cech / wersja danych, duplikat modelu, złe wagi, NaN w predykcjach, opcja sprzeczna
z plikiem wag), rejestr wcześniejszych raportów dla tych samych runów (dowolne --out).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ensemble import aligned_probas, apply_temperature, blend, evaluate, load_run
from tests.test_ensemble import N_ROWS, _make_proba, _true_labels, write_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ENSEMBLE = PROJECT_ROOT.parent / "backend" / "app" / "models" / "ensemble.py"
RUNS = {
    "20260102_000000_lightgbm_triage_standard": ("lightgbm", 2.5),
    "20260102_000100_xgboost_triage_standard": ("xgboost", 1.5),
    "20260102_000200_ebm_triage_standard": ("ebm", 2.0),
}


def _load_script():
    """Import skryptu (nazwa od cyfry); moduł w sys.modules — wymagają tego jego dataclassy."""
    spec = importlib.util.spec_from_file_location("final_report_script", PROJECT_ROOT / "scripts" / "08_final_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rewrite_predictions(run_dir: Path, **columns) -> None:
    """Dopisz / nadpisz kolumny meta w predictions_{val,test}.parquet (np. site, admitted)."""
    for split in ("val", "test"):
        path = run_dir / f"predictions_{split}.parquet"
        df = pd.read_parquet(path).sort_values("row_id").reset_index(drop=True)
        for name, values in columns.items():
            df[name] = values
        df.to_parquet(path, index=False)


@pytest.fixture
def report_runs(isolated_paths):
    """Trzy modele o różnej jakości, dwa oddziały (A/B), predykcje val i test."""
    y = _true_labels()
    root = isolated_paths.RUNS_DIR
    for i, (name, (model, quality)) in enumerate(RUNS.items()):
        run_dir = write_run(root, name, model, {s: _make_proba(y, quality, 10 * i + j)
                                                for j, s in enumerate(["val", "test"])}, y)
        _rewrite_predictions(run_dir, site=np.where(np.arange(N_ROWS) % 2 == 0, "A", "B"))
    return root


def _backend_loader():
    if not BACKEND_ENSEMBLE.exists():
        pytest.skip("brak backend/app/models/ensemble.py")
    spec = importlib.util.spec_from_file_location("backend_ensemble_for_tests", BACKEND_ENSEMBLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_ensemble_config


class TestFinalReport:
    def test_end_to_end(self, report_runs, capsys):
        module = _load_script()
        out = report_runs / "final_report__triage__a"
        weights = {"lightgbm": 0.6, "xgboost": 0.1, "ebm": 0.3}
        code = module.main(["--feature-set", "triage", "--tag", "standard", "--weights", json.dumps(weights),
                            "--bootstrap", "60", "--out", str(out)])
        assert code == 0
        for name in ("report.md", "metrics.json", "recipe.json", "backend_ensemble_weights.json"):
            assert (out / name).exists(), name
        figures = {p.name for p in (out / "figures").glob("*.png")}
        assert {"reliability_test_blend.png", "forest_test_argmax.png", "forest_test_cost.png",
                "confusion_test_cost_counts.png", "confusion_test_argmax_rowpct.png"} <= figures
        assert {p.name for p in (out / "tables").glob("*.csv")} >= {"ci_all_models.csv", "paired_blend_vs_best.csv",
                                                                     "sites_blend.csv"}

        report = (out / "report.md").read_text(encoding="utf-8")
        for fragment in ("## Receptura (wybrana na VAL)", "## Jak czytać", "## 3. Czy połączenie",
                         "## 4. Kalibracja", "## 5. Macierze pomyłek", "## 6. Połączenie na oddziałach",
                         "ECE idealnie skalibrowanego modelu", "Zbiór TESTOWY"):
            assert fragment in report, fragment
        assert "Pominięte:" in report                              # brak kolumny admitted → sekcja pominięta
        assert "Clopper–Pearsona" in report
        n_esi1 = int((_true_labels() == 0).sum())
        assert f"/{n_esi1})" in report                              # liczebność k/n przy metrykach ESI 1

        recipe = json.loads((out / "recipe.json").read_text(encoding="utf-8"))
        assert recipe["weights"] == pytest.approx(weights)
        assert recipe["best_single_model"] == "lightgbm"            # najniższy log loss na VAL
        assert recipe["temperature"] > 0 and recipe["rules"] == ["argmax", "cost"]
        assert sorted(r["name"] for r in recipe["runs"]) == sorted(RUNS)

        # Wynik połączenia = ta sama receptura policzona ręcznie (wagi + T z VAL → test)
        runs = [load_run(report_runs / name) for name in RUNS]
        y_val, p_val = aligned_probas(runs, "val")
        y_test, p_test = aligned_probas(runs, "test")
        from src.ensemble import fit_temperature

        t = fit_temperature(blend(p_val, weights), y_val)
        assert recipe["temperature"] == pytest.approx(t)
        final = apply_temperature(blend(p_test, weights), t)
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        qwk = metrics["test"]["ci"]["all"]["połączenie"]["cost"]["quadratic_weighted_kappa"]
        assert qwk["value"] == pytest.approx(evaluate(y_test, final, "cost")["quadratic_weighted_kappa"])
        assert qwk["ci_low"] <= qwk["value"] <= qwk["ci_high"] and qwk["n_valid"] == 60
        assert qwk["ci_method"] == "bootstrap-percentile"
        miss = metrics["test"]["ci"]["all"]["połączenie"]["cost"]["critical_miss_rate"]
        assert miss["ci_method"] == "clopper-pearson" and miss["n"] == n_esi1
        assert miss["ci_high"] > 0                                  # także przy 0 zdarzeń
        assert set(metrics["test"]["ci"]) == {"all", "vc"}
        assert set(metrics["test"]["sites"]) == {"A", "B"}
        assert metrics["test"]["admission"]["auc"] is None
        assert "lightgbm" in metrics["test"]["paired_all"]["all"]
        assert metrics["timings"]["razem_s"] > 0

        # Plik dla backendu: przyjmuje go loader backendu
        backend_path = out / "backend_ensemble_weights.json"
        backend = json.loads(backend_path.read_text(encoding="utf-8"))
        assert set(backend) == {"weights", "method", "temperature", "rule"}
        cfg = _backend_loader()(backend_path)
        assert cfg.rule == "cost" and cfg.method == "mean"
        assert cfg.temperature == pytest.approx(t, rel=1e-5)
        assert cfg.weights == pytest.approx(weights, abs=1e-6)

    def test_options_geometric_optimize_no_temperature(self, report_runs):
        module = _load_script()
        out = report_runs / "final_report__triage__b"
        assert module.main(["--runs", *RUNS, "--weights", "optimize", "--method", "geometric",
                            "--temperature", "none", "--rule", "argmax", "--bootstrap", "30", "--no-plots",
                            "--decimal-comma", "--out", str(out)]) == 0
        recipe = json.loads((out / "recipe.json").read_text(encoding="utf-8"))
        assert recipe["temperature"] is None and recipe["rules"] == ["argmax"]
        assert sum(recipe["weights"].values()) == pytest.approx(1.0)
        assert recipe["weights"]["lightgbm"] > recipe["weights"]["xgboost"]
        backend = json.loads((out / "backend_ensemble_weights.json").read_text(encoding="utf-8"))
        assert backend["temperature"] is None and backend["rule"] == "argmax" and backend["method"] == "geometric"
        cfg = _backend_loader()(out / "backend_ensemble_weights.json")
        assert cfg.temperature is None
        report = (out / "report.md").read_text(encoding="utf-8")
        assert "cała populacja, argmax" in report and "cała populacja, koszt" not in report   # tylko argmax
        assert "0," in report                                          # przecinek dziesiętny
        assert not (out / "figures").exists()

    def test_weights_from_backend_file_and_previous_report(self, report_runs):
        module = _load_script()
        first = report_runs / "final_report__triage__c"
        assert module.main(["--runs", *RUNS, "--bootstrap", "20", "--no-plots", "--out", str(first)]) == 0
        second = report_runs / "final_report__triage__d"
        assert module.main(["--runs", *RUNS, "--weights", str(first / "backend_ensemble_weights.json"),
                            "--bootstrap", "20", "--no-plots", "--out", str(second)]) == 0
        recipe = json.loads((second / "recipe.json").read_text(encoding="utf-8"))
        assert recipe["weights"] == pytest.approx({m: 1 / 3 for m, _ in RUNS.values()}, abs=1e-5)
        assert recipe["weights_mode"].startswith("z pliku")
        assert str(first) in recipe["previous_reports"]               # ostrzeżenie: test oceniany ponownie
        assert "powstało wcześniej" in (second / "report.md").read_text(encoding="utf-8")

    def test_weights_file_roundtrip_geometric_argmax(self, report_runs):
        """Plik backendu (geometric, argmax, T) jako --weights: metoda, reguła i temperatura przejęte."""
        module = _load_script()
        first = report_runs / "final_report__triage__g1"
        weights = {"lightgbm": 0.5, "xgboost": 0.2, "ebm": 0.3}
        assert module.main(["--runs", *RUNS, "--weights", json.dumps(weights), "--method", "geometric",
                            "--rule", "argmax", "--bootstrap", "20", "--no-plots", "--out", str(first)]) == 0
        file = first / "backend_ensemble_weights.json"
        before = json.loads(file.read_text(encoding="utf-8"))
        assert before["method"] == "geometric" and before["rule"] == "argmax" and before["temperature"] > 0

        second = report_runs / "final_report__triage__g2"
        assert module.main(["--runs", *RUNS, "--weights", str(file), "--bootstrap", "20", "--no-plots",
                            "--out", str(second)]) == 0
        recipe = json.loads((second / "recipe.json").read_text(encoding="utf-8"))
        assert recipe["method"] == "geometric" and recipe["backend_rule"] == "argmax"
        assert recipe["temperature_mode"] == "fit"
        assert set(recipe["options_from_weights_file"]) == {"method", "rule", "temperature"}
        assert recipe["weights"] == pytest.approx(weights, abs=1e-6)
        assert recipe["temperature"] == pytest.approx(before["temperature"], rel=1e-4)   # ponownie na VAL
        after = json.loads((second / "backend_ensemble_weights.json").read_text(encoding="utf-8"))
        assert after["method"] == "geometric" and after["rule"] == "argmax"
        assert after["weights"] == pytest.approx(before["weights"], abs=1e-6)
        assert after["temperature"] == pytest.approx(before["temperature"], rel=1e-4)
        assert "argmax" in recipe["rules"]

        # Plik bez temperatury → raport bez temperatury
        no_t = report_runs / "bez_t.json"
        no_t.write_text(json.dumps({**before, "temperature": None}), encoding="utf-8")
        third = report_runs / "final_report__triage__g3"
        assert module.main(["--runs", *RUNS, "--weights", str(no_t), "--bootstrap", "20", "--no-plots",
                            "--out", str(third)]) == 0
        recipe = json.loads((third / "recipe.json").read_text(encoding="utf-8"))
        assert recipe["temperature"] is None and recipe["method"] == "geometric"

    def test_registry_finds_reports_with_any_out_dir(self, report_runs, tmp_path):
        """Raporty w katalogach o dowolnych nazwach i w różnych miejscach — rejestr runs/final_reports.jsonl."""
        module = _load_script()
        first = tmp_path / "praca" / "raport_v1"
        second = tmp_path / "inne" / "raport_v2"
        assert module.main(["--runs", *RUNS, "--bootstrap", "10", "--no-plots", "--out", str(first)]) == 0
        assert module.main(["--runs", *RUNS, "--method", "geometric", "--bootstrap", "10", "--no-plots",
                            "--out", str(second)]) == 0
        recipe = json.loads((second / "recipe.json").read_text(encoding="utf-8"))
        assert recipe["previous_reports"] == [str(first.absolute())]
        assert recipe["previous_reports_detail"][0]["same_recipe"] is False
        registry = [json.loads(line) for line in
                    (report_runs / "final_reports.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [e["out_dir"] for e in registry] == [str(first.absolute()), str(second.absolute())]
        assert registry[0]["runs"] == sorted(RUNS) and registry[0]["recipe_hash"] != registry[1]["recipe_hash"]
        # Inny zestaw runów → brak wcześniejszych raportów
        third = tmp_path / "raport_v3"
        two = list(RUNS)[:2]
        assert module.main(["--runs", *two, "--bootstrap", "10", "--no-plots", "--out", str(third)]) == 0
        assert json.loads((third / "recipe.json").read_text(encoding="utf-8"))["previous_reports"] == []

    def test_admission_section(self, report_runs):
        y = _true_labels()
        rng = np.random.default_rng(0)
        admitted = (rng.random(N_ROWS) < np.where(y <= 1, 0.8, 0.15)).astype(int)
        for name in RUNS:
            _rewrite_predictions(report_runs / name, admitted=admitted)
        module = _load_script()
        out = report_runs / "final_report__triage__e"
        assert module.main(["--runs", *RUNS, "--bootstrap", "40", "--no-plots", "--out", str(out)]) == 0
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        adm = metrics["test"]["admission"]
        assert adm["source"] == "predykcje runów" and adm["n"] == N_ROWS
        scores = {row["score"]: row for row in adm["auc"]}
        assert scores["ESI z triażu (pielęgniarka)"]["value"] > 0.7
        assert any(k.startswith("Δ ") for k in scores)
        assert len(adm["by_level"]) == 5
        assert "## 7. Trafność względem przyjęcia" in (out / "report.md").read_text(encoding="utf-8")


class TestFinalReportRefusals:
    def _expect_refusal(self, argv: list[str], out: Path) -> None:
        module = _load_script()
        assert module.main([*argv, "--bootstrap", "10", "--no-plots", "--out", str(out)]) == 2
        assert not out.exists()

    def test_misaligned_rows(self, report_runs):
        y = _true_labels()
        p = {s: _make_proba(y, 2.0, 7) for s in ("val", "test")}
        write_run(report_runs, "20260103_000000_catboost_triage_standard", "catboost", p, y,
                  row_ids=np.arange(N_ROWS) + 1000)
        self._expect_refusal(["--runs", *RUNS, "20260103_000000_catboost_triage_standard"], report_runs / "x")

    def test_different_feature_set(self, report_runs):
        y = _true_labels()
        write_run(report_runs, "20260103_000000_catboost_full_standard", "catboost",
                  {s: _make_proba(y, 2.0, 8) for s in ("val", "test")}, y, feature_set="full")
        self._expect_refusal(["--runs", *RUNS, "20260103_000000_catboost_full_standard"], report_runs / "x")

    def test_different_data_version(self, report_runs):
        y = _true_labels()
        write_run(report_runs, "20260103_000000_catboost_triage_old", "catboost",
                  {s: _make_proba(y, 2.0, 9) for s in ("val", "test")}, y, data_version="esi-v1")
        self._expect_refusal(["--runs", *RUNS, "20260103_000000_catboost_triage_old"], report_runs / "x")

    def test_duplicate_model(self, report_runs):
        y = _true_labels()
        write_run(report_runs, "20260103_000000_lightgbm_triage_standard", "lightgbm",
                  {s: _make_proba(y, 2.0, 11) for s in ("val", "test")}, y)
        self._expect_refusal(["--runs", *RUNS, "20260103_000000_lightgbm_triage_standard"], report_runs / "x")

    def test_missing_test_predictions(self, report_runs):
        y = _true_labels()
        write_run(report_runs, "20260103_000000_catboost_triage_standard", "catboost",
                  {"val": _make_proba(y, 2.0, 12)}, y)
        self._expect_refusal(["--runs", *RUNS, "20260103_000000_catboost_triage_standard"], report_runs / "x")

    @pytest.mark.parametrize("weights", ['{"catboost": 1}', '{"lightgbm": -1}', '{"lightgbm": 0}', "{nie json"])
    def test_bad_weights(self, report_runs, weights):
        self._expect_refusal(["--runs", *RUNS, "--weights", weights], report_runs / "x")

    @pytest.mark.parametrize("option", [["--method", "mean"], ["--rule", "cost"], ["--backend-rule", "cost"],
                                        ["--temperature", "none"], ["--rule", "both", "--backend-rule", "cost"]])
    def test_option_conflicts_with_weights_file(self, report_runs, option):
        file = report_runs / "backend.json"
        file.write_text(json.dumps({"weights": {"lightgbm": 1, "ebm": 1}, "method": "geometric",
                                    "rule": "argmax", "temperature": 1.2}), encoding="utf-8")
        self._expect_refusal(["--runs", *RUNS, "--weights", str(file), *option], report_runs / "x")

    @pytest.mark.parametrize("payload", [{"weights": {"lightgbm": 1}, "method": "median"},
                                         {"weights": {"lightgbm": 1}, "rule": "max"},
                                         {"weights": {"lightgbm": 1}, "temperature": -1},
                                         {"weights": {"lightgbm": 1}, "extra": 1}])
    def test_bad_weights_file(self, report_runs, payload):
        file = report_runs / "backend.json"
        file.write_text(json.dumps(payload), encoding="utf-8")
        self._expect_refusal(["--runs", *RUNS, "--weights", str(file)], report_runs / "x")

    @pytest.mark.parametrize("split", ["val", "test"])
    def test_non_finite_probabilities(self, report_runs, split):
        path = report_runs / list(RUNS)[1] / f"predictions_{split}.parquet"
        df = pd.read_parquet(path)
        df.loc[df.index[4], [f"p_esi{k}" for k in range(1, 6)]] = np.nan
        df.to_parquet(path, index=False)
        self._expect_refusal(["--runs", *RUNS], report_runs / "x")

    def test_no_runs(self, isolated_paths):
        self._expect_refusal(["--feature-set", "triage"], isolated_paths.RUNS_DIR / "x")
