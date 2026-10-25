"""
Testy statystyki na zapisanych predykcjach (`src.evaluation.statistics`) i nowych wykresów
(`plot_reliability`, `plot_metric_ci`).

Wszystko na syntetycznych prawdopodobieństwach: zgodność z `full_evaluation`, wagi bootstrapu
= jawne losowanie, pokrycie przedziałów w symulacji (także rzadkich zdarzeń — przedział
Clopper–Pearsona), warstwy ESI, porównania parami (spójność p z przedziałem przy remisach w 0),
odrzucanie NaN, ECE i temperatura.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.ensemble import apply_temperature, evaluate
from src.evaluation.statistics import (
    CI_COLUMNS,
    DEFAULT_CI_METRICS,
    PROPORTION_METRICS,
    _AucPrep,
    _paired_summary,
    _Prepared,
    bootstrap_auc,
    bootstrap_ci,
    bootstrap_ci_rules,
    bootstrap_samples,
    calibrate_on_val,
    calibration_summary,
    check_proba,
    ci_to_dict,
    clopper_pearson,
    crossfit_temperature,
    ece_classwise,
    ece_consistency,
    ece_top_label,
    format_ci,
    paired_bootstrap_diff,
    paired_bootstrap_diffs,
    point_metrics,
    reliability_table,
    resample_weights,
    stratified_folds,
)
from src.utils.config import CLASS_NAMES, N_CLASSES

PRIORS = [0.03, 0.3, 0.4, 0.2, 0.07]


# ─────────────────────────────────────────
# Dane syntetyczne
# ─────────────────────────────────────────
def noisy_model(n: int, quality: float = 1.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """y z rozkładu ESI + prawdopodobieństwa skupione wokół prawdziwej klasy (jak w test_ensemble)."""
    rng = np.random.default_rng(seed)
    y = rng.choice(N_CLASSES, size=n, p=PRIORS)
    logits = rng.normal(0, 1, size=(n, N_CLASSES))
    logits[np.arange(n), y] += quality
    p = np.exp(logits)
    return y, p / p.sum(axis=1, keepdims=True)


def calibrated_model(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Idealnie skalibrowany model: p ~ Dirichlet, y ~ Categorical(p)."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(N_CLASSES, 0.7), size=n)
    cum = p.cumsum(axis=1)
    y = (rng.random(n)[:, None] > cum).sum(axis=1)
    return np.minimum(y, N_CLASSES - 1), p


# ─────────────────────────────────────────
# Metryki punktowe i ważone
# ─────────────────────────────────────────
class TestPointMetrics:
    @pytest.mark.parametrize("rule", ["argmax", "cost"])
    def test_match_full_evaluation(self, rule):
        y, p = noisy_model(3000, seed=1)
        mine = point_metrics(y, p, rule=rule)
        ref = evaluate(y, p, rule)
        for metric, value in mine.items():
            if metric == "recall_esi12":
                expected = 1.0 - ref["undertriage_rate"]
            elif metric == "ece":
                expected = ece_top_label(y, p)
            else:
                expected = ref[metric]
            assert value == pytest.approx(expected, abs=1e-10), metric

    def test_weights_equal_explicit_resample(self):
        """Krotności z bootstrapu ≡ metryki na jawnie powielonych wierszach."""
        y, p = noisy_model(800, seed=2)
        W = next(resample_weights(y, 1, seed=5, stratify=True))
        idx = np.repeat(np.arange(len(y)), W[0].astype(int))
        for rule in ("argmax", "cost"):
            prep = _Prepared(y, p, [rule], list(DEFAULT_CI_METRICS))
            weighted = prep.compute(W)
            explicit = point_metrics(y[idx], p[idx], rule=rule)
            for metric, value in explicit.items():
                key = ("*", metric) if metric in ("auc_macro", "log_loss", "brier", "ece") else (rule, metric)
                assert weighted[key][0] == pytest.approx(value, abs=1e-10), metric

    def test_per_class_metrics(self):
        y, p = noisy_model(2000, seed=3)
        values = point_metrics(y, p, ["recall_esi3", "auc_esi2"])
        ref = evaluate(y, p, "argmax")
        assert values["recall_esi3"] == pytest.approx(ref["recall_esi3"])
        assert values["auc_esi2"] == pytest.approx(ref["auc_esi2"])

    def test_unknown_metric(self):
        y, p = noisy_model(100)
        with pytest.raises(ValueError, match="Nieznane metryki"):
            point_metrics(y, p, ["f1_macro"])

    @pytest.mark.parametrize("frac", [0.02, 0.5, 0.95])
    def test_auc_with_ties_matches_sklearn(self, frac):
        rng = np.random.default_rng(4)
        positive = rng.random(3000) < frac
        score = np.round(rng.random(3000) + 0.4 * positive, 1)          # dużo remisów
        assert _AucPrep(score, positive)(np.ones((1, 3000)))[0] == pytest.approx(roc_auc_score(positive, score))
        idx = rng.integers(0, 3000, 3000)
        w = np.bincount(idx, minlength=3000)[None].astype(float)
        assert _AucPrep(score, positive)(w)[0] == pytest.approx(roc_auc_score(positive[idx], score[idx]))


# ─────────────────────────────────────────
# Losowania
# ─────────────────────────────────────────
class TestResampling:
    def test_stratified_keeps_class_counts(self):
        y = np.array([0] * 3 + [1] * 50 + [2] * 80 + [3] * 40 + [4] * 7)
        counts = np.bincount(y)
        for W in resample_weights(y, 200, seed=0, stratify=True):
            for c in range(N_CLASSES):
                np.testing.assert_array_equal(W[:, y == c].sum(axis=1), counts[c])
            assert ((W[:, y == 0]).sum(axis=1) == 3).all()            # rzadkie ESI 1 zawsze obecne

    def test_unstratified_keeps_total_only(self):
        y = np.array([0] * 3 + [2] * 197)
        W = np.vstack(list(resample_weights(y, 300, seed=0, stratify=False)))
        np.testing.assert_array_equal(W.sum(axis=1), len(y))
        assert (W[:, :3].sum(axis=1) == 0).any()                      # czasem bez ESI 1

    def test_independent_of_chunk_size(self):
        y = np.random.default_rng(0).choice(5, size=300)
        a = np.vstack(list(resample_weights(y, 25, seed=3, chunk=7)))
        b = np.vstack(list(resample_weights(y, 25, seed=3, chunk=25)))
        np.testing.assert_array_equal(a, b)

    def test_rare_class_n_valid(self):
        """Bez warstw recall_esi1 bywa nieokreślony (pominięty), z warstwami — zawsze określony."""
        rng = np.random.default_rng(1)
        y = np.concatenate([[0, 0], rng.choice([1, 2, 3], size=198)])
        p = rng.dirichlet(np.ones(N_CLASSES), size=len(y))
        loose = bootstrap_ci(y, p, ["recall_esi1", "accuracy"], n_boot=300, stratify=False).set_index("metric")
        assert loose.loc["recall_esi1", "n_valid"] < 300
        assert loose.loc["accuracy", "n_valid"] == 300
        strict = bootstrap_ci(y, p, ["recall_esi1"], n_boot=300, stratify=True).set_index("metric")
        assert strict.loc["recall_esi1", "n_valid"] == 300


# ─────────────────────────────────────────
# Przedziały ufności
# ─────────────────────────────────────────
class TestBootstrapCI:
    def test_columns_and_point_values(self):
        y, p = noisy_model(1500, seed=6)
        df = bootstrap_ci(y, p, rule="cost", n_boot=200)
        assert list(df.columns) == CI_COLUMNS
        assert list(df["metric"]) == list(DEFAULT_CI_METRICS)
        point = point_metrics(y, p, rule="cost")
        for row in df.itertuples():
            assert row.value == pytest.approx(point[row.metric])
            assert row.ci_low <= row.ci_high
            assert row.n_valid == 200
        qwk = df.set_index("metric").loc["quadratic_weighted_kappa"]
        assert qwk.ci_low < qwk.value < qwk.ci_high

    def test_deterministic_seed(self):
        y, p = noisy_model(500, seed=7)
        a = bootstrap_ci(y, p, ["quadratic_weighted_kappa"], n_boot=100, seed=1)
        b = bootstrap_ci(y, p, ["quadratic_weighted_kappa"], n_boot=100, seed=1)
        c = bootstrap_ci(y, p, ["quadratic_weighted_kappa"], n_boot=100, seed=2)
        pd.testing.assert_frame_equal(a, b)
        assert a.loc[0, "ci_low"] != c.loc[0, "ci_low"]

    def test_rules_share_probability_metrics(self):
        y, p = noisy_model(600, seed=8)
        out = bootstrap_ci_rules(y, p, n_boot=80)
        a, c = out["argmax"].set_index("metric"), out["cost"].set_index("metric")
        for metric in ("auc_macro", "log_loss", "brier", "ece"):
            assert a.loc[metric].equals(c.loc[metric])
        assert a.loc["undertriage_rate", "value"] != c.loc["undertriage_rate", "value"]

    @pytest.mark.parametrize("stratify", [False, True])
    def test_coverage_simulation(self, stratify):
        """95% przedział zawiera prawdziwą wartość w ~95% powtórzonych prób (luźne granice)."""
        metrics = ["accuracy", "quadratic_weighted_kappa", "log_loss", "recall_esi12"]
        rng = np.random.default_rng(123)
        y_pop, p_pop = calibrated_model(400_000, seed=99)
        truth = point_metrics(y_pop, p_pop, metrics)
        hits = dict.fromkeys(metrics, 0)
        reps = 120
        for r in range(reps):
            idx = rng.integers(0, len(y_pop), size=400)
            df = bootstrap_ci(y_pop[idx], p_pop[idx], metrics, n_boot=200, seed=r, stratify=stratify)
            for row in df.itertuples():
                hits[row.metric] += int(row.ci_low <= truth[row.metric] <= row.ci_high)
        for metric, h in hits.items():
            assert 0.85 <= h / reps <= 0.995, (metric, h / reps)

    def test_ci_to_dict_nan_to_none(self):
        y = np.array([1, 2, 3, 4] * 30)
        p = np.full((len(y), N_CLASSES), 0.2)
        out = ci_to_dict(bootstrap_ci(y, p, ["critical_miss_rate", "accuracy"], n_boot=20))
        assert out["critical_miss_rate"]["value"] is None          # brak ESI 1 → nieokreślone
        assert out["critical_miss_rate"]["n_valid"] == 0
        assert isinstance(out["accuracy"]["value"], float)

    def test_invalid_inputs(self):
        y, p = noisy_model(50)
        with pytest.raises(ValueError):
            bootstrap_ci(y, p, n_boot=0)
        with pytest.raises(ValueError):
            bootstrap_ci(y, p, alpha=1.5)
        with pytest.raises(ValueError):
            bootstrap_ci(y[:10], p, n_boot=5)


# ─────────────────────────────────────────
# Porównania parami
# ─────────────────────────────────────────
class TestPaired:
    def test_identical_predictions(self):
        y, p = noisy_model(1000, seed=9)
        for metric in ("quadratic_weighted_kappa", "undertriage_rate", "log_loss"):
            out = paired_bootstrap_diff(y, p, p.copy(), metric, rule="cost", n_boot=200)
            assert out["delta"] == 0.0
            assert out["ci_low"] <= 0.0 <= out["ci_high"]
            assert out["p_two_sided"] == 1.0
            assert out["better"] == "brak istotnej różnicy"

    def test_clearly_better_model(self):
        y, good = noisy_model(2000, quality=2.5, seed=10)
        rng = np.random.default_rng(11)
        logits = rng.normal(0, 1, size=good.shape)
        logits[np.arange(len(y)), y] += 0.8
        weak = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        qwk = paired_bootstrap_diff(y, good, weak, "quadratic_weighted_kappa", n_boot=300)
        assert qwk["delta"] > 0 and qwk["ci_low"] > 0
        assert qwk["p_two_sided"] < 0.01 and qwk["better"] == "A"
        ll = paired_bootstrap_diff(y, good, weak, "log_loss", n_boot=300)
        assert ll["delta"] < 0 and ll["ci_high"] < 0 and ll["better"] == "A"      # mniej = lepiej
        reverse = paired_bootstrap_diff(y, weak, good, "log_loss", n_boot=300)
        assert reverse["better"] == "B"

    def test_shared_samples_equal_pairwise_call(self):
        y, a = noisy_model(700, seed=12)
        _, b = noisy_model(700, quality=1.0, seed=13)
        b = b[np.argsort(np.random.default_rng(0).permutation(700))]
        res = bootstrap_samples(y, {"a": a, "b": b, "c": (a + b) / 2}, ["argmax"],
                                ["quadratic_weighted_kappa", "log_loss"], n_boot=100, seed=4)
        many = res.paired("a", "b").set_index("metric")
        direct = paired_bootstrap_diffs(y, a, b, ["quadratic_weighted_kappa", "log_loss"], ["argmax"],
                                        n_boot=100, seed=4).set_index("metric")
        pd.testing.assert_frame_equal(many, direct)
        assert set(res.ci("c", "argmax")["metric"]) == {"quadratic_weighted_kappa", "log_loss"}

    def test_p_value_consistent_with_ci(self):
        """Odsetki (dyskretne Δ z remisami w 0): przedział bez zera ⇔ p < alpha ⇔ wniosek A/B — dokładnie."""
        y, good = noisy_model(3000, quality=2.0, seed=14)
        _, other = noisy_model(3000, quality=1.9, seed=15)
        table = paired_bootstrap_diffs(y, good, other, ["undertriage_rate", "critical_miss_rate", "accuracy"],
                                       ["argmax", "cost"], n_boot=400)
        for row in table.itertuples():
            excludes = not (row.ci_low <= 0 <= row.ci_high)
            assert excludes == (row.p_two_sided < 0.05), row
            assert excludes == (row.better != "brak istotnej różnicy"), row

    def test_ties_at_zero_p_not_below_alpha(self):
        """10 losowań Δ < 0, 20 z Δ = 0, 970 z Δ > 0: przedział zawiera 0 → p ≥ alpha (nie 0,04)."""
        d = np.random.default_rng(0).permutation(np.r_[np.full(10, -0.01), np.zeros(20), np.full(970, 0.01)])
        out = _paired_summary(0.01, d, 0.05, "undertriage_rate")
        assert out["ci_low"] <= 0.0 <= out["ci_high"]
        assert out["p_two_sided"] >= 0.05
        assert out["better"] == "brak istotnej różnicy"
        clear = np.r_[np.full(5, -0.01), np.zeros(10), np.full(985, 0.01)]     # 15 ≤ 0 < c = 25
        out = _paired_summary(0.01, clear, 0.05, "quadratic_weighted_kappa")
        assert out["ci_low"] > 0 and out["p_two_sided"] < 0.05 and out["better"] == "A"

    @pytest.mark.parametrize("n_boot", [200, 401, 1000, 1999])
    def test_p_and_verdict_from_same_tails(self, n_boot):
        """Losowe dyskretne rozkłady Δ z remisami: przedział bez zera ⇔ p < alpha ⇔ wniosek A/B."""
        rng = np.random.default_rng(n_boot)
        for _ in range(300):
            shift = rng.integers(-4, 5)
            d = (rng.integers(-3, 4, size=n_boot) + shift * rng.random()).round()
            d[rng.random(n_boot) < rng.random() * 0.3] = np.nan          # część losowań nieokreślona
            for alpha in (0.05, 0.1):
                out = _paired_summary(float(np.nanmean(d)), d, alpha, "log_loss")
                excludes = not (out["ci_low"] <= 0 <= out["ci_high"])
                assert excludes == (out["p_two_sided"] < alpha), (out, alpha)
                assert excludes == (out["better"] != "brak istotnej różnicy"), out


# ─────────────────────────────────────────
# Odsetki rzadkich zdarzeń: przedział Clopper–Pearsona z k/n
# ─────────────────────────────────────────
def _rare_event_data(n_esi1: int, seed: int, p_miss: float = 0.02, p_hit: float = 0.10,
                     n_other: int = 62) -> tuple[np.ndarray, np.ndarray]:
    """ESI 1: z prawd. p_miss decyzja ESI 4 (krytyczne pominięcie), p_hit — ESI 1, reszta — ESI 2."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n_esi1, dtype=int), rng.choice([1, 2, 3, 4], size=n_other)]
    u = rng.random(n_esi1)
    pred = y.copy()
    pred[:n_esi1] = np.where(u < p_miss, 3, np.where(u < p_miss + p_hit, 0, 1))
    proba = np.full((len(y), N_CLASSES), 0.01)
    proba[np.arange(len(y)), pred] = 0.96
    return y, proba


class TestRareEventIntervals:
    def test_clopper_pearson_matches_scipy(self):
        from scipy.stats import binomtest

        for k, n in ((0, 38), (1, 38), (5, 10), (37, 38), (38, 38), (120, 5000)):
            ref = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
            low, high = clopper_pearson(k, n)
            assert low == pytest.approx(ref.low, abs=1e-9) and high == pytest.approx(ref.high, abs=1e-9)
        assert clopper_pearson(0, 38)[1] == pytest.approx(0.0925, abs=5e-4)
        assert all(np.isnan(clopper_pearson(0, 0)))
        with pytest.raises(ValueError):
            clopper_pearson(5, 3)

    def test_zero_events_upper_bound_positive(self):
        """0 krytycznych pominięć na 38 ESI 1: górna granica ~9,3%, nie [0; 0] jak z bootstrapu."""
        y, proba = _rare_event_data(38, seed=0, p_miss=0.0, p_hit=1.0)
        metrics = ["critical_miss_rate", "recall_esi1", "undertriage_rate", "quadratic_weighted_kappa"]
        df = bootstrap_ci(y, proba, metrics, n_boot=100).set_index("metric")
        miss = df.loc["critical_miss_rate"]
        assert miss["value"] == 0.0 and miss["ci_low"] == 0.0
        assert miss["ci_high"] == pytest.approx(clopper_pearson(0, 38)[1]) and miss["ci_high"] > 0.09
        assert (miss["k"], miss["n"], miss["ci_method"]) == (0, 38, "clopper-pearson")
        recall = df.loc["recall_esi1"]                                    # 38/38 → dolna granica < 1
        assert recall["value"] == 1.0 and recall["ci_high"] == 1.0 and recall["ci_low"] < 0.91
        qwk = df.loc["quadratic_weighted_kappa"]
        assert qwk["ci_method"] == "bootstrap-percentile" and np.isnan(qwk["k"])
        old = bootstrap_ci(y, proba, ["critical_miss_rate"], n_boot=100, proportion_ci="bootstrap")
        assert old.loc[0, "ci_high"] == 0.0 and old.loc[0, "ci_method"] == "bootstrap-percentile"
        out = ci_to_dict(df.reset_index())
        assert out["critical_miss_rate"]["k"] == 0 and out["critical_miss_rate"]["n"] == 38
        assert out["quadratic_weighted_kappa"]["k"] is None

    def test_rare_event_coverage(self):
        """Prawdziwe 2% krytycznych pominięć i 10% czułości ESI 1 przy 38 ESI 1: pokrycie ≥ ~95%."""
        reps = 300
        hits = {"critical_miss_rate": 0, "recall_esi1": 0}
        boot_hits = 0
        truth = {"critical_miss_rate": 0.02, "recall_esi1": 0.10}
        for r in range(reps):
            y, proba = _rare_event_data(38, seed=1000 + r)
            df = bootstrap_ci(y, proba, list(truth), n_boot=10, seed=r).set_index("metric")
            for m, t in truth.items():
                hits[m] += int(df.loc[m, "ci_low"] <= t <= df.loc[m, "ci_high"])
            old = bootstrap_ci(y, proba, ["critical_miss_rate"], n_boot=200, seed=r, proportion_ci="bootstrap")
            boot_hits += int(old.loc[0, "ci_low"] <= 0.02 <= old.loc[0, "ci_high"])
        for m, h in hits.items():
            assert h / reps >= 0.92, (m, h / reps)
        assert boot_hits / reps < 0.8                       # percentyle bootstrapu: wyraźnie za wąskie

    def test_counts_for_all_proportions(self):
        y, p = noisy_model(2000, seed=30)
        df = bootstrap_ci(y, p, rule="cost", n_boot=20).set_index("metric")
        for m in DEFAULT_CI_METRICS:
            row = df.loc[m]
            if m in PROPORTION_METRICS:
                assert row["ci_method"] == "clopper-pearson"
                assert row["value"] == pytest.approx(row["k"] / row["n"])
                assert row["ci_low"] <= row["value"] <= row["ci_high"]
            else:
                assert row["ci_method"] == "bootstrap-percentile"
        assert df.loc["accuracy", "n"] == len(y)
        assert df.loc["critical_miss_rate", "n"] == int((y == 0).sum())
        with pytest.raises(ValueError, match="proportion_ci"):
            bootstrap_ci(y, p, n_boot=5, proportion_ci="wilson")


# ─────────────────────────────────────────
# Niepoprawne prawdopodobieństwa
# ─────────────────────────────────────────
class TestInvalidProba:
    @pytest.mark.parametrize("bad_row", [[np.nan] * 5, [0.0] * 5, [0.5, np.inf, 0.1, 0.1, 0.1],
                                         [0.6, -0.1, 0.2, 0.2, 0.1]])
    def test_rejected_everywhere(self, bad_row):
        y, p = noisy_model(100, seed=31)
        p = p.copy()
        p[7] = bad_row
        for call in (lambda: bootstrap_ci(y, p, n_boot=5), lambda: point_metrics(y, p),
                     lambda: ece_top_label(y, p), lambda: ece_classwise(y, p),
                     lambda: reliability_table(y, p), lambda: calibration_summary(y, p),
                     lambda: paired_bootstrap_diff(y, p, noisy_model(100, seed=32)[1], n_boot=5)):
            with pytest.raises(ValueError, match="1 z 100 wierszy"):
                call()

    def test_valid_rows_normalized(self):
        p = np.array([[2.0, 1.0, 1.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2]])
        np.testing.assert_allclose(check_proba(p).sum(axis=1), 1.0)
        assert check_proba(p)[0, 0] == pytest.approx(0.5)
        with pytest.raises(ValueError, match="macierzy"):
            check_proba(np.ones(5))


class TestBootstrapAuc:
    def test_values_and_reference(self):
        rng = np.random.default_rng(16)
        adm = (rng.random(2000) < 0.3).astype(int)
        strong = adm + rng.normal(0, 0.8, 2000)
        weak = adm + rng.normal(0, 3.0, 2000)
        table = bootstrap_auc(adm, {"ref": weak, "model": strong, "same": weak.copy()}, reference="ref", n_boot=200)
        rows = table.set_index("score")
        assert rows.loc["model", "value"] == pytest.approx(roc_auc_score(adm, strong))
        assert rows.loc["Δ model − ref", "ci_low"] > 0
        assert rows.loc["Δ same − ref", "value"] == 0.0
        assert rows.loc["Δ same − ref", "p_two_sided"] == 1.0

    def test_single_class_raises(self):
        with pytest.raises(ValueError, match="jedną klasę"):
            bootstrap_auc(np.zeros(10), {"a": np.arange(10.0)}, n_boot=5)


# ─────────────────────────────────────────
# Kalibracja i temperatura
# ─────────────────────────────────────────
class TestCalibration:
    def test_ece_near_zero_for_calibrated(self):
        y, p = calibrated_model(60_000, seed=17)
        assert ece_top_label(y, p) < 0.015
        assert ece_top_label(y, p, strategy="quantile") < 0.015
        classwise = ece_classwise(y, p)
        assert set(classwise) == {*CLASS_NAMES, "mean"}
        assert classwise["mean"] < 0.01

    def test_ece_large_for_overconfident(self):
        y, p = calibrated_model(20_000, seed=18)
        sharp = apply_temperature(p, 0.4)
        assert ece_top_label(y, sharp) > 0.08
        assert ece_classwise(y, sharp)["mean"] > ece_classwise(y, p)["mean"] * 2
        summary = calibration_summary(y, sharp)
        assert summary["overconfidence"] > 0.05                       # pewność > trafność
        assert summary["mce_top_label"] >= summary["ece_top_label"]

    def test_ece_reference_detects_miscalibration(self):
        """Szum ECE idealnego modelu maleje z licznością; zbyt pewny model → małe p, skalibrowany → nie."""
        y, p = calibrated_model(20_000, seed=24)
        ok = ece_consistency(y, p, n_sim=200)
        assert ok["p_value"] > 0.01 and ok["ece"] <= ok["null_q95"] * 1.5
        bad = ece_consistency(y, apply_temperature(p, 0.6), n_sim=200)
        assert bad["p_value"] < 0.01 and bad["ece"] > bad["null_q95"]
        small = ece_consistency(y[:500], p[:500], n_sim=200)
        assert small["null_mean"] > ok["null_mean"] * 3                # mała grupa = duże zawyżenie ECE
        summary = calibration_summary(y, p, n_null=50)
        assert summary["ece_reference"]["n_sim"] == 50
        assert calibration_summary(y, p)["ece_reference"] is None

    def test_reliability_table(self):
        y, p = calibrated_model(5000, seed=19)
        for class_idx in (None, 0, 3):
            table = reliability_table(y, p, class_idx=class_idx, n_bins=10)
            assert list(table.columns) == ["bin", "bin_low", "bin_high", "mean_pred", "observed_freq", "count"]
            assert table["count"].sum() == len(y)
            assert (table["mean_pred"] >= table["bin_low"] - 1e-12).all()
            assert (table["mean_pred"] <= table["bin_high"] + 1e-12).all()
            assert table["observed_freq"].between(0, 1).all()
        quantile = reliability_table(y, p, n_bins=10, strategy="quantile")
        assert quantile["count"].sum() == len(y)
        with pytest.raises(ValueError, match="strategia"):
            reliability_table(y, p, strategy="log")

    def test_temperature_reduces_log_loss_keeps_argmax(self):
        y_val, p_val = calibrated_model(8000, seed=20)
        y_test, p_test = calibrated_model(8000, seed=21)
        sharp_val, sharp_test = apply_temperature(p_val, 0.5), apply_temperature(p_test, 0.5)
        t, val_cal, test_cal = calibrate_on_val(sharp_val, y_val, sharp_test)
        assert t > 1.0
        before = point_metrics(y_test, sharp_test, ["log_loss"])["log_loss"]
        after = point_metrics(y_test, test_cal, ["log_loss"])["log_loss"]
        assert after < before
        np.testing.assert_array_equal(test_cal.argmax(axis=1), sharp_test.argmax(axis=1))
        np.testing.assert_array_equal(val_cal.argmax(axis=1), sharp_val.argmax(axis=1))
        assert calibrate_on_val(sharp_val, y_val)[2] is None

    def test_crossfit_temperature(self):
        y, p = calibrated_model(6000, seed=22)
        sharp = apply_temperature(p, 0.5)
        cf, temps = crossfit_temperature(sharp, y, n_folds=2)
        assert len(temps) == 2 and all(t > 1.0 for t in temps)
        assert abs(temps[0] - temps[1]) < 0.5
        np.testing.assert_allclose(cf.sum(axis=1), 1.0)
        np.testing.assert_array_equal(cf.argmax(axis=1), sharp.argmax(axis=1))
        assert point_metrics(y, cf, ["log_loss"])["log_loss"] < point_metrics(y, sharp, ["log_loss"])["log_loss"]

    def test_stratified_folds_balanced(self):
        y = np.array([0] * 5 + [1] * 40 + [2] * 55)
        folds = stratified_folds(y, n_folds=2, seed=0)
        for c in (0, 1, 2):
            counts = np.bincount(folds[y == c], minlength=2)
            assert abs(counts[0] - counts[1]) <= 1


class TestFormat:
    def test_format_ci(self):
        assert format_ci(0.71234, 0.70111, 0.72399) == "0.712 [0.701; 0.724]"
        assert format_ci(0.1234, 0.11, 0.135, percent=True) == "12.3% [11.0; 13.5]"
        assert format_ci(0.5, 0.4, 0.6, decimal_comma=True) == "0,500 [0,400; 0,600]"
        assert format_ci(None, None, None) == "—"
        assert format_ci(0.5, None, None) == "0.500"


# ─────────────────────────────────────────
# Wykresy
# ─────────────────────────────────────────
class TestPlots:
    def test_plot_reliability(self, tmp_path):
        from src.evaluation.visualizations import plot_reliability

        y, p = calibrated_model(2000, seed=23)
        assert plot_reliability(y, p, path=tmp_path / "one.png").exists()
        out = plot_reliability(y, {"przed T": apply_temperature(p, 0.5), "po T": p}, path=tmp_path / "two.png",
                               n_bins=10, strategy="quantile")
        assert out.exists() and out.stat().st_size > 10_000
        with pytest.raises(ValueError, match="Najwyżej"):
            plot_reliability(y, {str(i): p for i in range(4)}, path=tmp_path / "x.png")

    def test_plot_metric_ci(self, tmp_path):
        from src.evaluation.visualizations import plot_metric_ci

        df = pd.DataFrame([
            {"label": label, "metric": metric, "group": group, "value": v, "ci_low": v - 0.02, "ci_high": v + 0.02}
            for label, v in (("połączenie", 0.7), ("xgboost", 0.68))
            for metric in ("QWK", "undertriage") for group in ("całość", "vc")
        ])
        assert plot_metric_ci(df, tmp_path / "forest.png", highlight=["połączenie"]).exists()
        biased = df.assign(value=df["ci_low"] - 0.01)                 # wartość poza przedziałem (np. ECE)
        assert plot_metric_ci(biased, tmp_path / "biased.png").exists()
        simple = df[(df["metric"] == "QWK") & (df["group"] == "całość")].drop(columns=["metric", "group"])
        assert plot_metric_ci(simple, tmp_path / "simple.png").exists()
        with pytest.raises(ValueError, match="Brak kolumn"):
            plot_metric_ci(df.drop(columns=["ci_low"]), tmp_path / "bad.png")
