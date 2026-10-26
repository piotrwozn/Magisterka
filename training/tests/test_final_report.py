"""
Testy raportu końcowego `scripts/08_final_report.py` na fałszywych runach (format `run_training`,
helpery z tests/test_ensemble.py): pełny przebieg (wszystkie reguły, reguła główna `target`
z parametrami w pliku dla backendu — decyzje backendu = decyzje treningu), opcje reguł
(--main-rule, --target-undertriage, --guarantee, --no-safety-net, --fit-population, --rule both),
plik wag zgodny z backendem (także ponowne użycie z metodą / regułą / temperaturą z pliku),
sekcje przyjęć, podgrup, błędów, krzywych decyzyjnych i analiz runs/analyses (oraz ich pomijanie
przy braku danych), odmowy (niewyrównane wiersze, inny zestaw cech / wersja danych, duplikat
modelu, złe wagi, NaN w predykcjach, opcja sprzeczna z plikiem wag, zła reguła w pliku), rejestr
wcześniejszych raportów dla tych samych runów (dowolne --out).
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
from src.evaluation.decision import decide_rule
from src.evaluation.metrics import quadratic_weighted_kappa
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


def _backend_module():
    if not BACKEND_ENSEMBLE.exists():
        pytest.skip("brak backend/app/models/ensemble.py")
    spec = importlib.util.spec_from_file_location("backend_ensemble_for_tests", BACKEND_ENSEMBLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _backend_loader():
    return _backend_module().load_ensemble_config


def _final_test_proba(root: Path, weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Receptura policzona ręcznie: wagi + T dopasowana na VAL → prawdopodobieństwa TEST."""
    from src.ensemble import fit_temperature

    runs = [load_run(root / name) for name in RUNS]
    y_val, p_val = aligned_probas(runs, "val")
    y_test, p_test = aligned_probas(runs, "test")
    t = fit_temperature(blend(p_val, weights), y_val)
    return y_test, apply_temperature(blend(p_test, weights), t)


def _tables_well_formed(markdown: str) -> None:
    """Każdy wiersz tabeli Markdown ma tyle kolumn co nagłówek (brak „|” wewnątrz komórek)."""
    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|---"):
            width = lines[i - 1].count("|")
            j = i + 1
            while j < len(lines) and lines[j].startswith("|"):
                assert lines[j].count("|") == width, f"zła tabela: {lines[j]!r} (nagłówek: {lines[i - 1]!r})"
                j += 1


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
        assert {"reliability_test_blend.png", "forest_test_argmax.png", "forest_test_target.png",
                "confusion_test_target_counts.png", "confusion_test_argmax_rowpct.png", "tradeoff_val.png",
                "dca_esi12_test.png"} <= figures
        assert {p.name for p in (out / "tables").glob("*.csv")} >= {
            "ci_all_models.csv", "paired_blend_vs_best.csv", "sites_blend.csv", "tradeoff_val.csv", "subgroups.csv",
            "dca_label.csv", "calibration_classes.csv", "calibration_esi12.csv", "deferral.csv", "conformal.csv",
            "errors_pairs.csv"}

        report = (out / "report.md").read_text(encoding="utf-8")
        for fragment in ("## Receptura (wybrana na VAL)", "### Reguły decyzyjne", "## Punkt pracy reguły decyzyjnej",
                         "## Jak czytać", "## 1. Wynik końcowy", "## 3. Czy połączenie", "wyższość (C1)",
                         "## 4. Kalibracja", "## 5. Macierze pomyłek", "## 6. Połączenie na oddziałach",
                         "## 7. Trafność względem przyjęcia", "## 8. Podgrupy", "## 9. Analiza błędów",
                         "## 10. Krzywe decyzyjne", "## 11. Kalibracja według poziomu ESI",
                         "## 12. Odsyłanie niepewnych", "## 13. Analizy dodatkowe",
                         "ECE idealnie skalibrowanego modelu", "Zbiór TESTOWY", "progi T (główna)"):
            assert fragment in report, fragment
        assert "Pominięte:" in report                              # brak kolumny admitted → sekcja pominięta
        assert "run_all.py --stage analyses" in report              # brak wyników analiz → nota
        assert "Clopper–Pearsona" in report
        n_esi1 = int((_true_labels() == 0).sum())
        assert f"/{n_esi1})" in report                              # liczebność k/n przy metrykach ESI 1
        _tables_well_formed(report)

        recipe = json.loads((out / "recipe.json").read_text(encoding="utf-8"))
        assert recipe["weights"] == pytest.approx(weights)
        assert recipe["best_single_model"] == "lightgbm"            # najniższy log loss na VAL
        assert recipe["temperature"] > 0 and recipe["rules"] == ["argmax", "cost", "target", "bayes_target"]
        assert recipe["main_rule"] == recipe["backend_rule"] == "target"
        decision = recipe["decision"]
        assert decision["target_undertriage"] == pytest.approx(0.15) and decision["safety_net"] is True
        assert decision["fit_population"] == "vc" and not decision["guarantee"]
        assert set(decision["per_model"]) >= {"lightgbm", "xgboost", "ebm"}
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
        # Reguła główna: QWK na TEST = decide_rule z parametrami zapisanymi dla backendu
        spec = decision["main_spec"]
        assert spec["rule"] == "target" and len(spec["cutpoints"]) == 4
        target_qwk = metrics["test"]["ci"]["all"]["połączenie"]["target"]["quadratic_weighted_kappa"]["value"]
        assert target_qwk == pytest.approx(quadratic_weighted_kappa(y_test, decide_rule(final, spec)))
        from sklearn.metrics import confusion_matrix

        assert metrics["test"]["confusion"]["all"]["target"] == confusion_matrix(
            y_test, decide_rule(final, spec), labels=list(range(5))).tolist()
        assert qwk["ci_low"] <= qwk["value"] <= qwk["ci_high"] and qwk["n_valid"] == 60
        assert qwk["ci_method"] == "bootstrap-percentile"
        miss = metrics["test"]["ci"]["all"]["połączenie"]["cost"]["critical_miss_rate"]
        assert miss["ci_method"] == "clopper-pearson" and miss["n"] == n_esi1
        assert miss["ci_high"] > 0                                  # także przy 0 zdarzeń
        assert set(metrics["test"]["ci"]) == {"all", "vc"}
        assert set(metrics["test"]["ci"]["all"]["połączenie"]) == {"argmax", "cost", "target", "bayes_target"}
        assert set(metrics["test"]["sites"]) == {"A", "B"}
        assert metrics["test"]["admission"]["auc"] is None
        assert metrics["test"]["dca_admission"]["skipped"]           # bez przyjęć — pominięte z powodem
        assert "lightgbm" in metrics["test"]["paired_all"]["all"]
        assert all(v.get("skipped") for v in metrics["analyses"].values())
        assert metrics["timings"]["razem_s"] > 0
        # Podgrupy bez atrybutów pacjentów: tylko oddział i vc (z metryk predykcji)
        attrs = {r["attribute"] for r in metrics["test"]["subgroups"]["rows"]}
        assert attrs == {"oddział", "parametry życiowe"}

        # Plik dla backendu: przyjmuje go loader backendu; decyzje backendu = decyzje treningu
        backend_path = out / "backend_ensemble_weights.json"
        backend = json.loads(backend_path.read_text(encoding="utf-8"))
        assert set(backend) == {"weights", "method", "temperature", "rule", "cutpoints", "safety_q"}
        assert backend["cutpoints"] == pytest.approx(spec["cutpoints"])
        backend_module = _backend_module()
        cfg = backend_module.load_ensemble_config(backend_path)
        assert cfg.rule == "target" and cfg.method == "mean"
        assert cfg.temperature == pytest.approx(t, rel=1e-5)
        assert cfg.weights == pytest.approx(weights, abs=1e-6)
        backend_pred = np.array([backend_module.decide(row, cfg.rule) for row in final])
        np.testing.assert_array_equal(backend_pred, decide_rule(final, spec))

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
        assert set(backend) == {"weights", "method", "temperature", "rule"}
        cfg = _backend_loader()(out / "backend_ensemble_weights.json")
        assert cfg.temperature is None
        report = (out / "report.md").read_text(encoding="utf-8")
        assert "cała populacja, reguła argmax" in report                  # tylko argmax
        assert "populacja, reguła koszt" not in report
        assert "0," in report                                          # przecinek dziesiętny
        assert not (out / "figures").exists()
        _tables_well_formed(report)

    def test_weights_from_backend_file_and_previous_report(self, report_runs):
        module = _load_script()
        first = report_runs / "final_report__triage__c"
        assert module.main(["--runs", *RUNS, "--weights", "equal", "--bootstrap", "20", "--no-plots",
                            "--out", str(first)]) == 0
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

    def test_main_rule_bayes_and_decision_options(self, report_runs):
        """--main-rule bayes_target, inny cel T, gwarancja, bez siatki, dopasowanie na całym VAL."""
        module = _load_script()
        out = report_runs / "final_report__triage__bayes"
        assert module.main(["--runs", *RUNS, "--main-rule", "bayes_target", "--target-undertriage", "0.2",
                            "--guarantee", "--no-safety-net", "--fit-population", "all", "--bootstrap", "20",
                            "--no-plots", "--out", str(out)]) == 0
        recipe = json.loads((out / "recipe.json").read_text(encoding="utf-8"))
        decision = recipe["decision"]
        assert recipe["main_rule"] == "bayes_target" and decision["fit_population"] == "all"
        assert decision["target_undertriage"] == pytest.approx(0.2) and decision["guarantee"] is True
        target = decision["blend"]["fitted"]["target"]
        assert target["safety_q"] is None and target["guarantee"] is True        # --no-safety-net, --guarantee
        assert target["achieved"]["n"] == N_ROWS                                    # dopasowanie na całym VAL
        backend = json.loads((out / "backend_ensemble_weights.json").read_text(encoding="utf-8"))
        assert backend["rule"] == "bayes_target" and backend["lam"] > 0 and "cutpoints" not in backend
        cfg = _backend_loader()(out / "backend_ensemble_weights.json")
        assert cfg.rule == "bayes_target"
        report = (out / "report.md").read_text(encoding="utf-8")
        assert "Bayes T (główna)" in report and "T = 20%" in report

    def test_backend_file_with_target_rule_roundtrip(self, report_runs):
        """Plik backendu z regułą target jako --weights: reguła przejęta, parametry dopasowane ponownie (te same)."""
        module = _load_script()
        first = report_runs / "final_report__triage__t1"
        assert module.main(["--runs", *RUNS, "--bootstrap", "10", "--no-plots", "--out", str(first)]) == 0
        file = first / "backend_ensemble_weights.json"
        before = json.loads(file.read_text(encoding="utf-8"))
        assert before["rule"] == "target"
        second = report_runs / "final_report__triage__t2"
        assert module.main(["--runs", *RUNS, "--weights", str(file), "--bootstrap", "10", "--no-plots",
                            "--out", str(second)]) == 0
        recipe = json.loads((second / "recipe.json").read_text(encoding="utf-8"))
        assert "rule" in recipe["options_from_weights_file"] and recipe["main_rule"] == "target"
        assert recipe["weights_file_rule"]["cutpoints"] == pytest.approx(before["cutpoints"])
        after = json.loads((second / "backend_ensemble_weights.json").read_text(encoding="utf-8"))
        assert after["cutpoints"] == pytest.approx(before["cutpoints"]) and after["safety_q"] == before["safety_q"]
        # Sprzeczna reguła z plikiem → odmowa
        third = report_runs / "final_report__triage__t3"
        assert module.main(["--runs", *RUNS, "--weights", str(file), "--main-rule", "cost", "--bootstrap", "10",
                            "--no-plots", "--out", str(third)]) == 2
        assert not third.exists()

    def test_rule_both_keeps_cost_for_backend(self, report_runs):
        """Zgodność wstecz: --rule both (argmax + koszt) → reguła backendu cost, bez dopasowywanych reguł."""
        module = _load_script()
        out = report_runs / "final_report__triage__both"
        assert module.main(["--runs", *RUNS, "--rule", "both", "--bootstrap", "10", "--no-plots",
                            "--out", str(out)]) == 0
        recipe = json.loads((out / "recipe.json").read_text(encoding="utf-8"))
        assert recipe["rules"] == ["argmax", "cost"] and recipe["main_rule"] == "cost"
        backend = json.loads((out / "backend_ensemble_weights.json").read_text(encoding="utf-8"))
        assert backend["rule"] == "cost" and set(backend) == {"weights", "method", "temperature", "rule"}

    def test_sections_with_attributes_and_admissions(self, report_runs):
        """Atrybuty pacjentów i przyjęcia w predykcjach → podgrupy, równość, błędy wg dolegliwości, DCA przyjęć."""
        y = _true_labels()
        rng = np.random.default_rng(3)
        admitted = (rng.random(N_ROWS) < np.where(y <= 1, 0.8, 0.15)).astype(int)
        arrival = rng.choice(3, size=N_ROWS, p=[0.35, 0.6, 0.05])          # „inny” ≈ 15 wierszy → za mała grupa
        columns = {
            "admitted": admitted, "age": rng.integers(18, 95, N_ROWS).astype(float),
            "is_male": rng.integers(0, 2, N_ROWS).astype(float),
            "arrival_ambulance": (arrival == 0).astype(float), "arrival_self": (arrival == 1).astype(float),
            "arrival_other": (arrival == 2).astype(float),
            "cc_group_cardiac": (rng.random(N_ROWS) < 0.2).astype(float),
            "cc_group_abdominal": (rng.random(N_ROWS) < 0.2).astype(float),
        }
        for name in RUNS:
            _rewrite_predictions(report_runs / name, **columns)
        module = _load_script()
        out = report_runs / "final_report__triage__attrs"
        assert module.main(["--runs", *RUNS, "--bootstrap", "20", "--out", str(out)]) == 0
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        rows = metrics["test"]["subgroups"]["rows"]
        assert {r["attribute"] for r in rows} == {"płeć", "wiek", "sposób przybycia", "oddział", "parametry życiowe"}
        other = next(r for r in rows if r["group"] == "inny")
        assert other["skipped"]                                             # n < 30 → pominięta z notą
        men = next(r for r in rows if r["group"] == "mężczyźni")
        assert men["under_delta_low"] <= men["under_delta"] <= men["under_delta_high"]
        assert men["n_admitted"] == int(((columns["is_male"] == 1) & (admitted == 1)).sum())
        complaints = {r["group"] for r in metrics["test"]["errors"]["complaints"]}
        assert {"kardiologiczne", "brzuszne", "brak grupy (pozostałe dolegliwości)"} <= complaints
        assert metrics["test"]["errors"]["label_noise"]["missed_n"] >= 0
        dca = metrics["test"]["dca_admission"]
        assert "skipped" not in dca and len(dca["delta_vs_nurse_rule"]) == 5
        assert metrics["test"]["admission_subgroups"]["rows"]
        assert len(metrics["test"]["admission_discordance"]) == 5
        report = (out / "report.md").read_text(encoding="utf-8")
        for fragment in ("### 7.1. Model a pielęgniarka w podgrupach", "### 7.2. Niezgodność",
                         "zakotwiczone w przyjęciu", "Według grupy dolegliwości", "### 10.2. Przyjęcie do szpitala",
                         "za mała podgrupa", "model: ESI 3–5 (niedotriażowani)", "przedział Newcombe'a)"):
            assert fragment in report, fragment
        _tables_well_formed(report)
        assert (out / "figures" / "dca_admission_test.png").exists()
        assert (out / "tables" / "admission_subgroups.csv").exists()

    def test_analyses_summary_and_superiority(self, report_runs, isolated_paths):
        """Wyniki runs/analyses (seedy, leave-site-out, krzywa uczenia) → sekcja 13 i reguła wyższości C1."""
        from src.analysis.common import METRICS_VERSION

        root = isolated_paths.RUNS_DIR / "analyses"
        stat = {"mean": 0.7, "sd": 0.0001, "min": 0.7, "max": 0.7, "n": 4, "values": [0.7] * 4}
        current = {"metrics_version": METRICS_VERSION, "data_fingerprint": "abc123",
                   "source_runs": {"lightgbm": "20260102_000000_lightgbm_triage_standard"}}
        seeds = {**current, "analysis": "seeds", "includes_source": True, "target_rule_note": "B1 + B2 (T = 0.15)",
                 "models": {"lightgbm": {"val": {"qwk_argmax": stat, "qwk_target": stat, "log_loss": stat},
                                         "test": {"qwk_argmax": stat, "qwk_target": stat, "log_loss": stat,
                                                  "undertriage_target": stat, "vc_qwk_target": stat}}}}
        site = {**current, "analysis": "site_out", "sites": {"A": {"lightgbm": {
            "raw": {"n": 100, "qwk_argmax": 0.5, "undertriage_argmax": 0.6, "log_loss": 1.1},
            "prior_em": {"n": 100, "qwk_argmax": 0.4, "em_suspect": True},
            "in_distribution": {"n": 50, "qwk_argmax": 0.6}, "n": 100, "nurse_admission_auc": 0.75}}}}
        curve = {**current, "analysis": "learning_curve", "models": {"lightgbm": {
            "points": [{"fraction": f, "n_fit": int(1000 * f), "val": {"log_loss": 1.0 - 0.1 * f, "qwk_argmax": 0.5}}
                       for f in (0.25, 0.5, 1.0)],
            "power_law": {"log_loss": {"a": 0.8, "c": 0.5, "reliable": False, "predicted": {"4000": 0.85}}}}}}
        for kind, payload in (("seeds", seeds), ("site_out", site), ("learning_curve", curve)):
            path = root / f"{kind}__triage__standard" / "results.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        module = _load_script()
        out = report_runs / "final_report__triage__analyses"
        assert module.main(["--feature-set", "triage", "--tag", "standard", "--bootstrap", "30", "--no-plots",
                            "--out", str(out)]) == 0
        report = (out / "report.md").read_text(encoding="utf-8")
        for fragment in ("### 13.1. Zmienność między seedami", "0.700 ± 0.000", "### 13.2. Walidacja z wyłączeniem",
                         "EM ⚠", "### 13.3. Krzywa uczenia", "**niewiarygodna**", "refity modelu lightgbm"):
            assert fragment in report, fragment
        _tables_well_formed(report)
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        sup = metrics["test"]["superiority"]
        assert sup["seed_sd"]["cała populacja|target|quadratic_weighted_kappa"] == pytest.approx(0.0001)
        assert sup["seed_sd"]["vc|target|quadratic_weighted_kappa"] == pytest.approx(0.0001)
        assert sup["seed_sd"]["cała populacja|argmax|undertriage_rate"] is None     # brak w wynikach seedów
        assert all(v != "—" for v in sup["verdicts"].values())
        assert {k: v.get("path") is not None for k, v in metrics["analyses"].items()} == {
            "seeds": True, "site_out": True, "learning_curve": True}
        best = metrics["recipe"]["best_single_model"]
        for verdict in sup["verdicts"].values():                         # zwycięzca zawsze nazwany
            assert not verdict.startswith("TAK") or verdict in ("TAK — połączenie", f"TAK — {best}")
        assert f"zwycięzca wynika z kierunku metryki (może nim być {best}" in report

        # Wyniki analiz z innego runa niż run raportu → pominięte z notą „nieaktualne”
        for kind in ("seeds", "site_out", "learning_curve"):
            path = root / f"{kind}__triage__standard" / "results.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["source_runs"] = {"lightgbm": "20250101_000000_lightgbm_triage_standard"}
            path.write_text(json.dumps(data), encoding="utf-8")
        out2 = report_runs / "final_report__triage__stale"
        assert module.main(["--feature-set", "triage", "--tag", "standard", "--bootstrap", "30", "--no-plots",
                            "--out", str(out2)]) == 0
        report = (out2 / "report.md").read_text(encoding="utf-8")
        assert report.count("nieaktualne") >= 3 and "0.700 ± 0.000" not in report
        sup = json.loads((out2 / "metrics.json").read_text(encoding="utf-8"))["test"]["superiority"]
        assert sup["seed_sd"]["cała populacja|target|quadratic_weighted_kappa"] is None

    def test_refit_with_model_does_not_replace_main_run(self, report_runs):
        """Refit z --save-models (nowszy, z model.joblib) — bez --tag raport bierze run główny."""
        import shutil

        main_name = "20260102_000000_lightgbm_triage_standard"
        refit = report_runs / "20260105_000000_lightgbm_triage_standard-seed1"
        shutil.copytree(report_runs / main_name, refit)
        info = json.loads((refit / "run_info.json").read_text(encoding="utf-8"))
        info.update({"tag": "standard-seed1", "custom": True, "refit_of": main_name})
        (refit / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
        module = _load_script()
        out = report_runs / "final_report__refit"
        assert module.main(["--bootstrap", "20", "--no-plots", "--out", str(out)]) == 0
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        assert sorted(r["name"] for r in metrics["recipe"]["runs"]) == sorted(RUNS)

    def test_without_vitals_column_falls_back_to_all(self, isolated_paths):
        """Predykcje bez vitals_complete: reguły dopasowane na całym VAL (nota), raport bez podgrupy vc."""
        y = _true_labels()
        root = isolated_paths.RUNS_DIR
        for i, (name, (model, quality)) in enumerate(RUNS.items()):
            write_run(root, name, model, {s: _make_proba(y, quality, 10 * i + j) for j, s in enumerate(["val", "test"])},
                      y, vitals=False)
        module = _load_script()
        out = root / "final_report__triage__novc"
        assert module.main(["--runs", *RUNS, "--bootstrap", "10", "--no-plots", "--out", str(out)]) == 0
        recipe = json.loads((out / "recipe.json").read_text(encoding="utf-8"))
        assert recipe["decision"]["fit_population"] == "all"
        assert "vitals_complete" in recipe["decision"]["fit_population_note"]
        report = (out / "report.md").read_text(encoding="utf-8")
        assert "### 1.2. Podgrupa vc" not in report and "brak kolumny vitals_complete" in report


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
                                         {"weights": {"lightgbm": 1}, "extra": 1},
                                         {"weights": {"lightgbm": 1}, "rule": "target"},
                                         {"weights": {"lightgbm": 1}, "rule": "cost", "lam": 2.0},
                                         {"weights": {"lightgbm": 1}, "rule": "bayes_target", "lam": -1},
                                         {"weights": {"lightgbm": 1}, "cutpoints": [1, 2, 3, 4]}])
    def test_bad_weights_file(self, report_runs, payload):
        file = report_runs / "backend.json"
        file.write_text(json.dumps(payload), encoding="utf-8")
        self._expect_refusal(["--runs", *RUNS, "--weights", str(file)], report_runs / "x")

    @pytest.mark.parametrize("option", [["--main-rule", "argmax", "--backend-rule", "cost"],
                                        ["--rule", "both", "--main-rule", "target"],
                                        ["--rule", "argmax", "--main-rule", "cost"]])
    def test_inconsistent_rule_options(self, report_runs, option):
        self._expect_refusal(["--runs", *RUNS, *option], report_runs / "x")

    @pytest.mark.parametrize("value", ["0", "1.5"])
    def test_bad_target_undertriage(self, report_runs, value):
        module = _load_script()
        with pytest.raises(SystemExit):
            module.main(["--runs", *RUNS, "--target-undertriage", value])

    @pytest.mark.parametrize("split", ["val", "test"])
    def test_non_finite_probabilities(self, report_runs, split):
        path = report_runs / list(RUNS)[1] / f"predictions_{split}.parquet"
        df = pd.read_parquet(path)
        df.loc[df.index[4], [f"p_esi{k}" for k in range(1, 6)]] = np.nan
        df.to_parquet(path, index=False)
        self._expect_refusal(["--runs", *RUNS], report_runs / "x")

    def test_no_runs(self, isolated_paths):
        self._expect_refusal(["--feature-set", "triage"], isolated_paths.RUNS_DIR / "x")


# ─────────────────────────────────────────
# Receptura z modelem fundamentalnym (TFM): --include-pretrained i backend_ensemble_weights_no_tfm.json
# ─────────────────────────────────────────
TFM_RUN = "20260102_000300_tabpfn_triage_standard"
NO_TFM_FILE = "backend_ensemble_weights_no_tfm.json"


@pytest.fixture
def tfm_report_runs(report_runs):
    """report_runs + run TabPFN (run_info `pretrained: true`) z tym samym tagiem i wierszami."""
    y = _true_labels()
    run_dir = write_run(report_runs, TFM_RUN, "tabpfn", {s: _make_proba(y, 2.8, 40 + j)
                                                         for j, s in enumerate(["val", "test"])}, y)
    _rewrite_predictions(run_dir, site=np.where(np.arange(N_ROWS) % 2 == 0, "A", "B"))
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    info["pretrained"] = True
    (run_dir / "run_info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return report_runs


class TestNoTfmVariant:
    def _run(self, argv: list[str], out: Path) -> int:
        return _load_script().main([*argv, "--bootstrap", "40", "--no-plots", "--out", str(out)])

    def test_tag_selection_skips_tfm_without_flag(self, tfm_report_runs):
        out = tfm_report_runs / "r_plain"
        assert self._run(["--feature-set", "triage", "--tag", "standard"], out) == 0
        recipe = json.loads((out / "recipe.json").read_text(encoding="utf-8"))
        assert "tabpfn" not in {r["model"] for r in recipe["runs"]}
        assert "no_tfm" not in recipe                                   # klasyczna receptura bez zmian
        assert not (out / NO_TFM_FILE).exists()
        assert NO_TFM_FILE not in (out / "report.md").read_text(encoding="utf-8")

    def test_include_pretrained_writes_no_tfm_variant(self, tfm_report_runs):
        out = tfm_report_runs / "r_tfm"
        assert self._run(["--feature-set", "triage", "--tag", "standard", "--include-pretrained"], out) == 0
        recipe = json.loads((out / "recipe.json").read_text(encoding="utf-8"))
        assert "tabpfn" in {r["model"] for r in recipe["runs"]}
        backend = json.loads((out / "backend_ensemble_weights.json").read_text(encoding="utf-8"))
        no_tfm = json.loads((out / NO_TFM_FILE).read_text(encoding="utf-8"))
        assert "tabpfn" in backend["weights"] and "tabpfn" not in no_tfm["weights"]
        assert set(no_tfm["weights"]) <= {"lightgbm", "xgboost", "ebm"}
        assert no_tfm["rule"] == backend["rule"] == "target" and len(no_tfm["cutpoints"]) == 4
        assert no_tfm["method"] == backend["method"] and no_tfm["temperature"] > 0
        assert recipe["no_tfm"]["dropped"] == ["tabpfn"]
        assert recipe["no_tfm"]["weights_mode"].startswith("optimize")
        report = (out / "report.md").read_text(encoding="utf-8")
        assert NO_TFM_FILE in report and "Receptura zawiera model fundamentalny (tabpfn)" in report
        # Plik bez TFM = receptura dopasowana na VAL bez TFM (ta sama obróbka co 08: wagi → T → reguła)
        from src.ensemble import fit_temperature, optimize_weights
        from src.evaluation import report_sections as rs

        runs = [load_run(tfm_report_runs / name) for name in RUNS]
        y_val, p_val, vc = aligned_probas(runs, "val", return_vc=True)
        weights = optimize_weights(p_val, y_val)
        p = blend(p_val, weights)
        t = fit_temperature(p, y_val)
        spec = rs.fit_rules(y_val, apply_temperature(p, t), vc, rs.DecisionOptions(rules=["target"])).specs["target"]
        assert no_tfm["temperature"] == pytest.approx(round(t, 6))
        assert no_tfm["cutpoints"] == pytest.approx(spec["cutpoints"])
        # Format zgodny z backendem (także TFM w głównym pliku — znany model, nie odrzucany)
        cfg = _backend_loader()(out / NO_TFM_FILE)
        assert set(cfg.weights) == set(no_tfm["weights"])
        assert "tabpfn" in _backend_loader()(out / "backend_ensemble_weights.json").weights

    def test_explicit_weights_dropped_and_renormalised(self, tfm_report_runs):
        out = tfm_report_runs / "r_explicit"
        weights = {"lightgbm": 0.3, "xgboost": 0.1, "tabpfn": 0.6}
        assert self._run(["--runs", RUNS_LIST[0], RUNS_LIST[1], TFM_RUN, "--weights", json.dumps(weights),
                          "--temperature", "none", "--main-rule", "cost", "--rule", "both"], out) == 0
        no_tfm = json.loads((out / NO_TFM_FILE).read_text(encoding="utf-8"))
        assert no_tfm["weights"] == {"lightgbm": pytest.approx(0.75), "xgboost": pytest.approx(0.25)}
        assert no_tfm["temperature"] is None and no_tfm["rule"] == "cost"

    def test_tfm_with_zero_weight_writes_no_variant(self, tfm_report_runs):
        out = tfm_report_runs / "r_zero"
        assert self._run(["--runs", RUNS_LIST[0], TFM_RUN, "--weights", json.dumps({"lightgbm": 1, "tabpfn": 0})],
                         out) == 0
        assert not (out / NO_TFM_FILE).exists()

    def test_only_tfm_members_noted(self, tfm_report_runs):
        out = tfm_report_runs / "r_only"
        assert self._run(["--runs", TFM_RUN], out) == 0
        assert not (out / NO_TFM_FILE).exists()
        assert "wyłącznie z modeli fundamentalnych" in (out / "report.md").read_text(encoding="utf-8")


RUNS_LIST = list(RUNS)
