"""
Testy obliczeń sekcji raportu końcowego (`src/evaluation/report_sections.py`) i trybu
`05_explain.py --global`: bootstrap dla dowolnych reguł (zgodny z `statistics.bootstrap_samples`),
dopasowanie reguł na VAL (z awaryjnym przejściem na cały VAL), przedział Newcombe'a, net benefit,
kalibracja per grupa, odsyłanie i conformal, podgrupy (mała podgrupa pominięta), analiza błędów,
reguła wyższości C1, wczytywanie wyników runs/analyses, atrybuty pacjentów z predykcji i ze splitu,
krzywe kształtu EBM i wykresy globalne SHAP.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.common import METRICS_VERSION
from src.evaluation import report_sections as rs
from src.evaluation.decision import decide_rule, normalize_spec
from src.evaluation.metrics import quadratic_weighted_kappa
from src.evaluation.statistics import bootstrap_samples

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _data(n: int = 1500, seed: int = 0, quality: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.choice(5, size=n, p=[0.05, 0.3, 0.4, 0.2, 0.05])
    logits = rng.normal(0, 1, (n, 5))
    logits[np.arange(n), y] += quality
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    vc = np.arange(n) % 3 != 0
    return y, p, vc


# ─────────────────────────────────────────
# Bootstrap dla podanych decyzji
# ─────────────────────────────────────────
class TestBootstrapDecisions:
    def test_matches_statistics_for_argmax_and_cost(self):
        y, p, _ = _data()
        q = np.roll(p, 1, axis=0)
        decisions = {name: {r: decide_rule(pp, {"rule": r}) for r in ("argmax", "cost")}
                     for name, pp in (("a", p), ("b", q))}
        ours = rs.bootstrap_decisions(y, {"a": p, "b": q}, decisions, None, n_boot=40, seed=3)
        ref = bootstrap_samples(y, {"a": p, "b": q}, ["argmax", "cost"], None, n_boot=40, seed=3)
        for rule in ("argmax", "cost"):
            pd.testing.assert_frame_equal(ours.ci("a", rule), ref.ci("a", rule))
            pd.testing.assert_frame_equal(ours.paired("a", "b", None, [rule]), ref.paired("a", "b", None, [rule]))

    def test_custom_decisions_point_values_and_counts(self):
        y, p, _ = _data(seed=1)
        pred = np.clip(y + (np.arange(len(y)) % 7 == 0), 0, 4)        # co 7. pacjent o poziom mniej pilny
        res = rs.bootstrap_decisions(y, {"m": p}, {"m": {"moja": pred}}, None, n_boot=20)
        ci = res.ci("m", "moja").set_index("metric")
        assert ci.loc["quadratic_weighted_kappa", "value"] == pytest.approx(quadratic_weighted_kappa(y, pred))
        high = np.isin(y, (0, 1))
        assert ci.loc["undertriage_rate", "k"] == int((high & (pred >= 2)).sum())
        assert ci.loc["undertriage_rate", "n"] == int(high.sum())
        point = rs.decision_point_metrics(y, p, pred, ["quadratic_weighted_kappa", "log_loss"])
        assert point["quadratic_weighted_kappa"] == pytest.approx(quadratic_weighted_kappa(y, pred))

    def test_rejects_inconsistent_inputs(self):
        y, p, _ = _data(n=100)
        with pytest.raises(ValueError):
            rs.bootstrap_decisions(y, {"a": p}, {"a": {"r": np.full(len(y), 7)}}, n_boot=5)
        with pytest.raises(ValueError):
            rs.bootstrap_decisions(y, {"a": p, "b": p}, {"a": {"r": y}, "b": {"s": y}}, n_boot=5)


# ─────────────────────────────────────────
# Reguły decyzyjne na VAL
# ─────────────────────────────────────────
class TestFitRules:
    def test_specs_roundtrip_and_predictions(self):
        y, p, vc = _data()
        opts = rs.DecisionOptions()
        fitted = rs.fit_rules(y, p, vc, opts)
        assert fitted.population == "vc" and fitted.mask is not None
        assert set(fitted.specs) == set(rs.RULE_ORDER)
        for rule, spec in fitted.specs.items():
            assert normalize_spec(spec) == spec                     # format backendu
        preds = rs.predict_rules(p, fitted)
        np.testing.assert_array_equal(preds["target"], decide_rule(p, fitted.specs["target"]))
        assert json.dumps(fitted.to_dict())                          # JSON-owe
        summary = rs.val_rule_summary(y, p, fitted, opts)
        assert summary["target"]["kind"].startswith("cross-fit")
        assert summary["argmax"]["kind"] == "bez dopasowania"
        assert summary["target"]["fit_population"]["n"] == int(vc.sum())

    def test_population_fallbacks(self):
        y, p, vc = _data(n=400)
        mask, pop, note = rs.fit_mask(y, None, "vc")
        assert mask is None and pop == "all" and "vitals_complete" in note
        tiny = np.zeros(len(y), dtype=bool)
        tiny[np.flatnonzero(y >= 2)[:50]] = True                     # vc bez pacjentów ESI 1–2
        mask, pop, note = rs.fit_mask(y, tiny, "vc")
        assert pop == "all" and "ESI 1–2" in note
        assert rs.fit_mask(y, vc, "all")[1] == "all"

    def test_only_requested_rules(self):
        y, p, vc = _data(n=400)
        fitted = rs.fit_rules(y, p, vc, rs.DecisionOptions(main_rule="cost", rules=["argmax", "cost"]))
        assert set(fitted.specs) == {"argmax", "cost"} and not fitted.objects

    def test_tradeoff_section_marks_chosen_point(self, tmp_path):
        y, p, vc = _data(n=800)
        opts = rs.DecisionOptions(target=0.125)
        sec = rs.tradeoff_section(y, p, vc, "vc", opts, tmp_path / "t.png")
        assert (tmp_path / "t.png").exists() and sec["figure"] == "t.png"
        table = sec["table"]
        assert np.isclose(table["target"].dropna().astype(float), 0.125).any()   # cel T dodany do siatki
        lines = rs.render_tradeoff(sec, rs.Fmt(), opts)
        assert any("**progi T + siatka" in line for line in lines)
        assert rs.chosen_point(rs.DecisionOptions(main_rule="cost")) == {"method": "cost", "param": 3.0}


# ─────────────────────────────────────────
# Statystyka pomocnicza
# ─────────────────────────────────────────
class TestStatisticsHelpers:
    def test_newcombe_matches_published_example(self):
        # Newcombe (1998), metoda 10: 56/70 vs 48/80 → Δ 0,2000 [0,0524; 0,3339]
        d, lo, hi = rs.newcombe_diff(56, 70, 48, 80)
        assert d == pytest.approx(0.2)
        assert lo == pytest.approx(0.0524, abs=5e-4) and hi == pytest.approx(0.3339, abs=5e-4)
        assert all(math.isnan(v) for v in rs.newcombe_diff(1, 0, 1, 10))

    def test_net_benefit_by_hand(self):
        outcome = np.array([1, 1, 0, 0, 0])
        score = np.array([0.9, 0.4, 0.6, 0.1, 0.2])
        nb = rs.net_benefit_score(outcome, score, [0.5])
        assert nb[0] == pytest.approx(1 / 5 - 1 / 5 * 1.0)          # TP 1, FP 1, waga 0,5/0,5
        assert rs.net_benefit_all(outcome, [0.25])[0] == pytest.approx(0.4 - 0.6 / 3)
        fixed = rs.net_benefit_fixed(outcome, score >= 0.3, [0.2, 0.5])
        np.testing.assert_allclose(fixed, rs.net_benefit_score(outcome, (score >= 0.3).astype(float), [0.2, 0.5]))

    def test_logistic_fit_recovers_coefficients(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=20000)
        yb = (rng.random(20000) < 1 / (1 + np.exp(-(0.5 + 1.5 * x)))).astype(float)
        a, b = rs.logistic_fit(x, yb)
        assert a == pytest.approx(0.5, abs=0.06) and b == pytest.approx(1.5, abs=0.08)
        # perfekcyjnie skalibrowane p → wyraz wolny „in the large” ≈ 0
        p = rng.uniform(0.05, 0.95, 20000)
        yy = (rng.random(20000) < p).astype(float)
        citl = rs.logistic_fit(np.log(p / (1 - p)), yy, slope=False)[0]
        assert abs(citl) < 0.06

    def test_calibration_groups_oe_one_for_matching_frequencies(self):
        y = np.repeat(np.arange(5), 100)
        p = np.full((500, 5), 0.2)
        sec = rs.calibration_groups(y, p, [("wszyscy", np.ones(500, dtype=bool)), ("pusta", np.zeros(500, bool))])
        assert len(sec["classes"]) == 1                                 # pusta grupa pominięta
        row = sec["classes"][0]
        for k in range(5):
            assert row[f"oe_{k}"] == pytest.approx(1.0) and row[f"oe_{k}_low"] < 1 < row[f"oe_{k}_high"]
        assert sec["esi12"][0]["oe"] == pytest.approx(1.0)
        md = "\n".join(rs.render_calibration_groups(sec, rs.Fmt(decimal_comma=True)))
        assert "1,00" in md


# ─────────────────────────────────────────
# Odsyłanie i conformal
# ─────────────────────────────────────────
class TestDeferralConformal:
    def test_uncertainty_scores(self):
        _, p, _ = _data(n=50)
        el_var = rs.uncertainty_score(p, {"rule": "target", "cutpoints": [0.5, 1.5, 2.5, 3.5]})
        assert (el_var >= 0).all()
        sharp = np.eye(5)[[0, 2, 4]]
        assert np.allclose(rs.uncertainty_score(sharp, {"rule": "target"}), 0)
        assert np.allclose(rs.uncertainty_score(sharp, {"rule": "cost"}), 0)
        assert np.allclose(rs.uncertainty_score(sharp, {"rule": "bayes_target", "lam": 4.0}), 0)
        assert np.allclose(rs.uncertainty_score(sharp, {"rule": "argmax"}), 0)

    def test_deferral_monotone(self):
        y, p, vc = _data(n=3000, seed=4)
        spec = {"rule": "argmax"}
        pred = decide_rule(p, spec)
        sec = rs.deferral_section(p, vc, y, p, vc, pred, spec)
        rows = [r for r in sec["rows"] if r["population"] == "all"]
        assert rows[0]["fraction"] == 0 and rows[0]["decided_share"] == 1.0
        shares = [r["decided_share"] for r in rows]
        assert shares == sorted(shares, reverse=True) and shares[-1] < 0.8
        md = "\n".join(rs.render_deferral(sec, rs.conformal_section(y, p, y, p, vc), rs.Fmt(), "argmax"))
        assert "12.1. Odsyłanie" in md and "12.2. Zbiory conformal" in md

    def test_conformal_coverage(self):
        y_val, p_val, _ = _data(n=6000, seed=5)
        y_test, p_test, vc = _data(n=6000, seed=6)
        sec = rs.conformal_section(y_val, p_val, y_test, p_test, vc, alpha=0.1)
        rows = {(r["method"], r["population"]): r for r in sec["rows"]}
        assert rows[("LAC", "all")]["coverage"] == pytest.approx(0.9, abs=0.03)
        mondrian = rows[("LAC warunkowy po ESI", "all")]
        for k in range(5):
            assert mondrian[f"cov_{k}"] == pytest.approx(0.9, abs=0.06)


# ─────────────────────────────────────────
# Podgrupy, przyjęcia, błędy
# ─────────────────────────────────────────
def _meta_and_attrs(n: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame({"site": np.where(np.arange(n) % 2 == 0, "A", "B"),
                         "vitals_complete": (np.arange(n) % 3 != 0).astype(int)}, index=np.arange(n))
    arrival = rng.choice(3, size=n, p=[0.4, 0.58, 0.02])
    attrs = pd.DataFrame({"age": rng.integers(18, 95, n).astype(float), "is_male": rng.integers(0, 2, n),
                          "arrival_ambulance": (arrival == 0).astype(int), "arrival_self": (arrival == 1).astype(int),
                          "arrival_other": (arrival == 2).astype(int),
                          "cc_group_cardiac": (rng.random(n) < 0.3).astype(int)}, index=meta.index)
    return meta, attrs


class TestSubgroupsAdmissionsErrors:
    def test_subgroups_with_reference_and_small_group(self):
        y, p, _ = _data(n=1000, seed=7)
        meta, attrs = _meta_and_attrs(1000)
        groups = rs.subgroup_masks(meta, attrs)
        assert [g["attribute"] for g in groups if g["reference"]] == ["płeć", "wiek", "sposób przybycia", "oddział",
                                                                     "parametry życiowe"]
        pred = decide_rule(p, {"rule": "cost"})
        esi = y + 1.0
        admitted = (np.random.default_rng(1).random(1000) < np.where(y <= 1, 0.8, 0.2)).astype(float)
        sec = rs.subgroup_section(y, p, pred, groups, esi=esi, admitted=admitted, pred_argmax=p.argmax(1),
                                  n_boot=20)
        rows = {r["group"]: r for r in sec["rows"]}
        assert rows["inny"]["skipped"]                                    # ~20 pacjentów < 30
        assert rows["mężczyźni"]["reference_group"] == "kobiety"
        assert rows["mężczyźni"]["under_delta_low"] <= rows["mężczyźni"]["under_delta"]
        women = rows["kobiety"]
        sel = (attrs["is_male"].to_numpy() == 0) & (admitted == 1)
        assert women["n_admitted"] == int(sel.sum())
        assert women["adm_model_k"] == int((pred[sel] >= 2).sum()) and women["adm_nurse_k"] == int((esi[sel] >= 3).sum())
        assert women["qwk_low"] <= women["qwk"] <= women["qwk_high"]
        md = "\n".join(rs.render_subgroups(sec, rs.Fmt(), "cost", 95, "test"))
        assert "odniesienie" in md and "za mała podgrupa" in md and "zakotwiczone w przyjęciu" in md
        assert rs.subgroup_section(y, p, pred, [])["skipped"]

    def test_admission_tables(self):
        y, p, _ = _data(n=800, seed=8)
        meta, attrs = _meta_and_attrs(800)
        admitted = (np.random.default_rng(2).random(800) < np.where(y <= 1, 0.8, 0.2)).astype(float)
        esi = y + 1.0
        pred = decide_rule(p, {"rule": "argmax"})
        sub = rs.admission_subgroups(admitted, esi, p, rs.subgroup_masks(meta, attrs), 20, 0.05, 0)
        ok = [r for r in sub["rows"] if not r.get("skipped")]
        assert ok and all(r["auc_model_low"] <= r["auc_model"] <= r["auc_model_high"] for r in ok)
        disc = rs.discordance_table(admitted, esi, pred)
        assert sum(r["n_up"] + r["n_same"] + r["n_down"] for r in disc) == 800
        assert disc[0]["n_up"] == 0                                        # ESI 1 nie da się „podnieść”
        sec = rs.admission_section(admitted, esi, p, {"argmax": pred}, 20, 0.05, 0)
        assert len(sec["by_level"]) == 5 and (sec["auc"]["score"] == rs.NURSE).any()
        md = "\n".join(rs.render_admission_extra(sub, disc, rs.Fmt(), "argmax", 95))
        assert "7.1." in md and "7.2." in md

    def test_error_analysis(self):
        y, p, _ = _data(n=1000, seed=9)
        _, attrs = _meta_and_attrs(1000)
        pred = decide_rule(p, {"rule": "argmax"})
        admitted = np.ones(1000)
        admitted[::2] = 0
        sec = rs.error_analysis(y, pred, attrs, admitted)
        assert sum(pair["n"] for pair in sec["pairs"]) == sec["n_undertriaged"]
        groups = {r["group"]: r for r in sec["complaints"]}
        assert set(groups) == {"kardiologiczne", "brak grupy (pozostałe dolegliwości)"}
        assert sum(r["n_under"] for r in groups.values()) == sec["n_undertriaged"]   # grupy rozłączne tutaj
        assert sec["label_noise"]["missed_n"] + sec["label_noise"]["kept_n"] == int((y == 1).sum())
        no_attrs = rs.error_analysis(y, pred, None, None)
        assert "complaints_skipped" in no_attrs and "label_noise" not in no_attrs
        assert "Według pary" in "\n".join(rs.render_errors(no_attrs, rs.Fmt(), "argmax"))

    @staticmethod
    def _noise_text(missed: tuple[int, int], kept: tuple[int, int]) -> str:
        sec = {"n_undertriaged": 0, "pairs": [], "esi1_as_esi2": 0, "n_esi1": 0,
               "label_noise": {"missed_k": missed[0], "missed_n": missed[1], "kept_k": kept[0], "kept_n": kept[1]}}
        return "\n".join(rs.render_errors(sec, rs.Fmt(), "target"))

    def test_label_noise_sentence_follows_the_data(self):
        """Zdanie o szumie etykiet tylko, gdy pominięci ESI 2 są przyjmowani ISTOTNIE rzadziej."""
        rarer = self._noise_text((30, 100), (60, 100))
        assert "Ślad szumu etykiet" in rarer and "istotnie rzadziej" in rarer
        same = self._noise_text((58, 100), (57, 100))                     # pełne dane: 58% vs 57%
        assert "Ślad szumu" not in same and "nie różni się istotnie" in same
        more = self._noise_text((80, 100), (40, 100))
        assert "Ślad szumu" not in more and "CZĘŚCIEJ" in more
        assert "przyjęci do szpitala" not in self._noise_text((0, 0), (5, 10))

    def test_subgroup_text_names_fit_population(self):
        y, p, _ = _data(n=600, seed=11)
        meta, attrs = _meta_and_attrs(600)
        sec = rs.subgroup_section(y, p, p.argmax(1), rs.subgroup_masks(meta, attrs), n_boot=10)
        vc = "\n".join(rs.render_subgroups(sec, rs.Fmt(), "target", 95, None, population="vc"))
        assert "dopasowaną na VAL w podgrupie vc" in vc and "całej populacji" not in vc
        assert "dopasowaną na całym VAL" in "\n".join(rs.render_subgroups(sec, rs.Fmt(), "target", 95, None, "all"))
        assert "bez parametrów" in "\n".join(rs.render_subgroups(sec, rs.Fmt(), "argmax", 95, None, "vc"))

    def test_dca_sections(self, tmp_path):
        y, p, vc = _data(n=1000, seed=10)
        pred = {"argmax": decide_rule(p, {"rule": "argmax"})}
        label = rs.dca_label_section(y, p, vc, pred, tmp_path / "dca.png")
        assert label["figure"] == "dca.png" and set(label) >= {"all", "vc"}
        rng = np.random.default_rng(3)
        adm = (rng.random(1000) < np.where(y <= 1, 0.7, 0.15)).astype(float)
        el = (p * np.arange(5)).sum(1)
        sec = rs.dca_admission_section(adm, el, y + 1.0, adm, el, y + 1.0, pred, 20, 0.05, 0)
        assert sec["recalibration"]["b"] > 0                                # wyższy −E[ESI] → częściej przyjęcie
        assert len(sec["delta_vs_nurse_rule"]) == len(rs.DCA_DELTA_THRESHOLDS)
        skipped = rs.dca_admission_section(None, el, y + 1.0, adm, el, y + 1.0, pred, 20, 0.05, 0)
        assert skipped["skipped"]
        md = "\n".join(rs.render_dca(label, skipped, rs.Fmt(), "argmax", 95))
        assert "10.1." in md and "Pominięte" in md


# ─────────────────────────────────────────
# Reguła wyższości C1 i wyniki runs/analyses
# ─────────────────────────────────────────
class TestAnalysesAndSuperiority:
    def test_superiority_branches(self):
        q = "quadratic_weighted_kappa"
        assert rs.superiority(q, 0.02, -0.001, 0.04, 0.001).startswith("nie (CI zawiera 0)")
        assert "MID" in rs.superiority(q, 0.005, 0.001, 0.009, 0.0001)
        assert "2 × SD" in rs.superiority(q, 0.02, 0.01, 0.03, 0.02)
        assert rs.superiority(q, 0.02, 0.01, 0.03, 0.001) == "TAK — A"
        assert rs.superiority(q, 0.02, 0.01, 0.03, 0.001, "połączenie", "ebm") == "TAK — połączenie"
        assert "brak SD" in rs.superiority(q, 0.02, 0.01, 0.03, None)
        assert rs.superiority(q, None, None, None, None) == "—"
        assert "|" not in "".join(rs.superiority(q, d, d - 0.001, d + 0.001, 0.01) for d in (0.005, 0.015, 0.05))

    def test_superiority_names_the_better_side(self):
        """Δ = połączenie − model: wynik gorszy dla połączenia nigdy nie daje „TAK — połączenie”."""
        sup = lambda m, d, lo, hi, sd: rs.superiority(m, d, lo, hi, sd, "połączenie", "ebm")  # noqa: E731
        assert sup("quadratic_weighted_kappa", -0.03, -0.05, -0.01, 0.004) == "TAK — ebm"
        assert sup("auc_macro", 0.02, 0.01, 0.03, 0.001) == "TAK — połączenie"
        assert sup("undertriage_rate", 0.03, 0.015, 0.045, 0.004) == "TAK — ebm"
        assert sup("undertriage_rate", -0.03, -0.045, -0.015, 0.004) == "TAK — połączenie"
        assert sup("log_loss", 0.02, 0.01, 0.03, 0.002) == "TAK — ebm"
        assert sup("log_loss", -0.02, -0.03, -0.01, 0.002) == "TAK — połączenie"

    def test_seed_sd_and_load_analyses(self, tmp_path):
        assert all(v.get("skipped") for v in rs.load_analyses("triage", "standard", tmp_path).values())
        assert "tagu" in rs.load_analyses("triage", None, tmp_path)["seeds"]["skipped"]
        stat = lambda sd: {"mean": 0.5, "sd": sd, "n": 3}  # noqa: E731
        payload = {"metrics_version": METRICS_VERSION,
                   "models": {"a": {"test": {"qwk_target": stat(0.01), "log_loss": stat(0.002),
                                             "vc_qwk_target": stat(0.03)}},
                              "b": {"test": {"qwk_target": stat(0.02)}}}}
        path = tmp_path / "analyses" / "seeds__triage__standard" / "results.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        (tmp_path / "analyses" / "site_out__triage__standard").mkdir()
        (tmp_path / "analyses" / "site_out__triage__standard" / "results.json").write_text("{zły", encoding="utf-8")
        loaded = rs.load_analyses("triage", "standard", tmp_path)
        assert loaded["seeds"]["results"] == payload and "nieczytelny" in loaded["site_out"]["skipped"]
        seeds = loaded["seeds"]
        assert rs.seed_sd(seeds, ["a"], "quadratic_weighted_kappa", "target") == pytest.approx(0.01)
        assert rs.seed_sd(seeds, ["a", "b"], "quadratic_weighted_kappa", "target") == pytest.approx(0.02)
        assert rs.seed_sd(seeds, ["a"], "quadratic_weighted_kappa", "target", population="vc") == pytest.approx(0.03)
        assert rs.seed_sd(seeds, ["a"], "log_loss", "bayes_target") == pytest.approx(0.002)
        assert rs.seed_sd(seeds, ["a"], "quadratic_weighted_kappa", "bayes_target") is None
        assert rs.seed_sd(None, ["a"], "log_loss", "argmax") is None
        md = "\n".join(rs.render_analyses(loaded, rs.Fmt(), rs.DecisionOptions()))
        assert "13.1." in md and "nieczytelny" in md and "13.3." in md

    @staticmethod
    def _write(root: Path, kind: str, payload: dict) -> None:
        path = root / "analyses" / f"{kind}__triage__standard" / "results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_load_analyses_skips_stale_results(self, tmp_path):
        """Inna wersja metryk / inne dane / inny run modelu → nie trafia do raportu bez noty."""
        base = {"metrics_version": METRICS_VERSION, "data_fingerprint": "fp1",
                "source_runs": {"a": "run_a", "b": "run_b_old"}}
        stat = {"mean": 0.5, "sd": 0.01, "n": 3}
        self._write(tmp_path, "seeds", {**base, "models": {"a": {"test": {"qwk_target": stat}},
                                                           "b": {"test": {"qwk_target": stat}}}})
        self._write(tmp_path, "site_out", {**base, "sites": {"A": {"b": {"raw": {"n": 10}}}}})
        self._write(tmp_path, "learning_curve", {**base, "metrics_version": METRICS_VERSION - 1, "models": {}})
        loaded = rs.load_analyses("triage", "standard", tmp_path, fingerprint="fp1",
                                  run_names={"a": "run_a", "b": "run_b_new"})
        assert set(loaded["seeds"]["results"]["models"]) == {"a"}
        assert loaded["seeds"]["stale_models"] == {"b": {"analysis": "run_b_old", "report": "run_b_new"}}
        assert "nieaktualne" in loaded["site_out"]["skipped"]                 # jedyny model nieaktualny
        assert "wersji metryk" in loaded["learning_curve"]["skipped"]
        md = "\n".join(rs.render_analyses(loaded, rs.Fmt(), rs.DecisionOptions()))
        assert "Pominięto nieaktualne wyniki modeli" in md and "run_b_old" in md
        assert rs.seed_sd(loaded["seeds"], ["b"], "quadratic_weighted_kappa", "target") is None
        other_data = rs.load_analyses("triage", "standard", tmp_path, fingerprint="fp2")
        assert "innych danych" in other_data["seeds"]["skipped"]
        assert set(rs.load_analyses("triage", "standard", tmp_path)["seeds"]["results"]["models"]) == {"a", "b"}

    def test_site_out_oe_column(self):
        site = {"results": {"sites": {"A": {"m": {"raw": {"n": 10, "oe_esi1": 4.95},
                                                  "prior_em": {"n": 10, "oe_esi1": 7e113, "em_suspect": True}}}}}}
        md = "\n".join(rs.render_analyses({"seeds": {"skipped": "x"}, "site_out": site,
                                            "learning_curve": {"skipped": "x"}}, rs.Fmt(), rs.DecisionOptions()))
        assert "| 4.95 |" in md and "| > 999 |" in md and "> 1 = model ZANIŻA" in md

    def test_learning_curve_reasons_rendered(self):
        fit = {"a": 0.35, "c": 1.2, "reliable": False, "predicted": {"401779": 0.35},
               "reasons": ["asymptota / ekstrapolacja gorsza od ostatniego punktu"]}
        loaded = {"seeds": {"skipped": "x"}, "site_out": {"skipped": "x"},
                  "learning_curve": {"results": {"models": {"lightgbm": {"points": [], "power_law": {"qwk_cost": fit}}}}}}
        md = "\n".join(rs.render_analyses(loaded, rs.Fmt(), rs.DecisionOptions()))
        assert "**niewiarygodna** — nie raportuj (asymptota / ekstrapolacja gorsza od ostatniego punktu)" in md


# ─────────────────────────────────────────
# Atrybuty pacjentów (predykcje / split po row_id)
# ─────────────────────────────────────────
class TestAttributes:
    def test_from_prediction_frame(self, isolated_paths):
        meta, attrs = _meta_and_attrs(20)
        frame = attrs.iloc[::-1]                                          # kolejność inna niż w meta
        loaded, source = rs.load_attributes(meta, "test", "abc", frame)
        assert source == "predykcje runów"
        pd.testing.assert_frame_equal(loaded, attrs[loaded.columns])
        none, reason = rs.load_attributes(meta, "test", "abc", None)       # brak splitów → powód
        assert none is None and "splitów" in reason

    def test_from_prepared_split_and_fingerprint_guard(self, prepared_paths):
        from src.data import dataset

        test = dataset.load_split("test", columns=["row_id", "age", "is_male"]).set_index("row_id")
        meta = pd.DataFrame(index=test.index[:30])
        loaded, source = rs.load_attributes(meta, "test", dataset.data_fingerprint())
        assert "row_id" in source and loaded["age"].tolist() == test["age"].iloc[:30].tolist()
        refused, reason = rs.load_attributes(meta, "test", "inny-odcisk")
        assert refused is None and "odcisk" in reason


# ─────────────────────────────────────────
# 05_explain --global (C11)
# ─────────────────────────────────────────
def _explain_script():
    spec = importlib.util.spec_from_file_location("script_05_explain_global", SCRIPTS_DIR / "05_explain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_frame(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    hr = rng.normal(85, 20, n)
    age = rng.uniform(18, 95, n)
    X = pd.DataFrame({"triage_vital_hr": hr, "age": age, "noise": rng.normal(size=n)}).astype(np.float32)
    X.loc[X.index[::17], "triage_vital_hr"] = np.nan
    risk = 0.03 * np.nan_to_num(np.abs(hr - 80), nan=10) + 0.02 * (age - 50)
    y = np.digitize(-risk + rng.normal(0, 0.6, n), np.quantile(-risk, [0.05, 0.35, 0.75, 0.95]))
    return X, y


class TestExplainGlobal:
    def test_run_global_tree_model(self, tmp_path):
        from src.models import get_model

        script = _explain_script()
        X, y = _synthetic_frame()
        model = get_model("lightgbm", params={"n_estimators": 30, "n_jobs": 1, "verbose": -1})
        model.fit_estimator(X, y)
        model.feature_set = "triage"
        summary = script.run_global(model, X, tmp_path / "global", size=200, top_k=2, seed=0, run_name="run_x")
        files = set(summary["files"])
        assert {"shap_mean_abs.csv", "shap_summary_all_levels.png", "shap_beeswarm_esi1.png",
                "shap_dependence_triage_vital_hr.png", "shap_dependence_age.png"} <= files
        assert all((tmp_path / "global" / f).exists() for f in files)
        assert len(summary["shap"]["top_features"]) == 2 and "ebm_shapes" not in summary
        stored = json.loads((tmp_path / "global" / "global_summary.json").read_text(encoding="utf-8"))
        assert stored["run"] == "run_x"

    def test_logreg_shap_folds_missing_indicators(self):
        """Regresja logistyczna z add_indicator: SHAP w przestrzeni cech modelu, addytywny w logitach."""
        from src.explain.shap_explainer import SHAPTriageExplainer
        from src.models import get_model

        X, y = _synthetic_frame()
        assert X.isna().any().any()                      # są braki → są wskaźniki braków
        model = get_model("logreg")
        model.fit_estimator(X, y)
        assert len(model.transformed_feature_names()) > X.shape[1]
        explainer = SHAPTriageExplainer(model, background_data=X.head(50))
        values, base = explainer.explain_values(X.head(40))
        assert values.shape == (40, 5, X.shape[1])
        clf = model.model.steps[-1][1]
        logits = clf.decision_function(explainer._transform(X.head(40)))
        np.testing.assert_allclose(base[None, :] + values.sum(axis=2), logits, atol=1e-8)

    def test_run_global_survives_shap_error(self, tmp_path, monkeypatch):
        """Błąd SHAP (np. regresja logistyczna: kolumny wskaźników braków) → podsumowanie i wyjście bez wyjątku."""
        from src.models import get_model

        script = _explain_script()
        X, y = _synthetic_frame(n=300)
        model = get_model("lightgbm", params={"n_estimators": 10, "n_jobs": 1, "verbose": -1})
        model.fit_estimator(X, y)
        model.feature_set = "triage"

        def broken(self, X):
            raise ValueError("Nieoczekiwany kształt wartości SHAP: (300, 314, 5)")

        monkeypatch.setattr(script.SHAPTriageExplainer, "explain_values", broken)
        summary = script.run_global(model, X, tmp_path / "g", size=100, top_k=2, seed=0)
        assert "Nieoczekiwany kształt" in summary["shap"]["error"] and summary["files"] == []
        assert (tmp_path / "g" / "global_summary.json").exists()

    def test_latest_run_with_model_skips_refits(self, isolated_paths):
        from tests.test_ensemble import write_run

        script = _explain_script()
        y = np.arange(20) % 5
        proba = {"val": np.full((20, 5), 0.2)}
        main = write_run(isolated_paths.RUNS_DIR, "20260101_000000_ebm_triage_standard", "ebm", proba, y)
        refit = write_run(isolated_paths.RUNS_DIR, "20260102_000000_ebm_triage_standard-seed1", "ebm", proba, y,
                          tag="standard-seed1")
        info = json.loads((refit / "run_info.json").read_text(encoding="utf-8"))
        info["refit_of"] = main.name
        (refit / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
        assert (refit / "model.joblib").exists()
        assert script.latest_run_with_model("triage", "ebm").name == main.name

    def test_ebm_shape_curves(self, tmp_path):
        from src.models import get_model

        script = _explain_script()
        X, y = _synthetic_frame(n=500, seed=1)
        model = get_model("ebm", params={"max_rounds": 60, "outer_bags": 3, "interactions": 0, "n_jobs": 1})
        model.fit_estimator(X, y)
        shapes = script.ebm_shape_curves(model.model, X, {"triage_vital_hr": 80.0, "age": 45.0, "brak": 1.0})
        assert set(shapes) == {"triage_vital_hr", "age"}
        hr = shapes["triage_vital_hr"]
        assert len(hr["grid"]) == script.EBM_SHAPE_GRID and hr["sd"] is not None and (hr["sd"] >= 0).all()
        assert np.isfinite(hr["missing"]) and hr["sign_changes_p1_p99"] >= 0
        model.feature_set = "triage"
        summary = script.run_global(model, X, tmp_path / "ebm", size=100, top_k=1, seed=0)
        assert "ebm_shape_triage_vital_hr.png" in summary["files"]
        assert set(summary["ebm_shapes"]) == {"triage_vital_hr", "age"}
