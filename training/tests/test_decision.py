"""
Testy reguł decyzyjnych (`src.evaluation.decision`): progi EL z celem undertriage (B1/B6),
siatka bezpieczeństwa (B2), decyzja Bayesa z λ (B4), spec w formacie backendu, cross-fitting
i krzywa kompromisu (C6).

Dane syntetyczne: skalibrowane prawdopodobieństwa (etykiety losowane z nich), ~70% wierszy
„vitals complete”.
"""

from __future__ import annotations

import json
from functools import partial

import numpy as np
import pandas as pd
import pytest

from src.evaluation import decide
from src.evaluation.decision import (
    RULES,
    BayesRule,
    TargetRule,
    apply_cutpoints,
    apply_safety_net,
    bayes_cost_matrix,
    crossfit_rule,
    decide_rule,
    expected_level,
    fit_bayes_rule,
    fit_safety_net,
    fit_target_rule,
    normalize_spec,
    population_mask,
    rule_from_dict,
    tradeoff_curve,
)
from src.evaluation.metrics import critical_miss_rate, quadratic_weighted_kappa, undertriage_rate
from src.evaluation.visualizations import plot_tradeoff
from src.utils.config import (
    DECISION_FIT_POPULATION,
    DECISION_TARGET_UNDERTRIAGE,
    SAFETY_NET_MAX_CRITICAL_MISS,
    TRADEOFF_TARGETS,
)

ESI1, ESI2, ESI3, ESI4, ESI5 = range(5)


def _data(seed: int = 0, n: int = 6000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    proba = rng.dirichlet([0.3, 1.0, 1.4, 1.0, 0.5], n)
    u = rng.random(n)[:, None]
    y = (u > np.cumsum(proba, axis=1)).sum(axis=1)          # etykieta ~ proba (skalibrowane)
    vc = rng.random(n) < 0.7
    return proba, y, vc


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _data(0)


# ─────────────────────────────────────────
# Stosowanie reguł
# ─────────────────────────────────────────
class TestApply:
    def test_expected_level(self):
        proba, _, _ = _data(1, 500)
        el = expected_level(proba)
        np.testing.assert_allclose(el, proba @ np.arange(5.0), rtol=0, atol=1e-12)
        # Wiersz po wierszu bit w bit to samo (backend liczy jeden wektor)
        assert all(expected_level(p) == el[i] for i, p in enumerate(proba))

    def test_cutpoints_strictly_exceeded(self):
        cuts = [0.5, 1.5, 2.5, 3.5]
        el = np.array([0.0, 0.5, 0.51, 1.5, 2.0, 2.5, 3.49, 3.5, 3.51, 4.0])
        assert apply_cutpoints(el, cuts).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    @pytest.mark.parametrize("cuts", [[1, 2, 3], [1, 3, 2, 4], [1, 2, 3, np.nan], "1234", [1, 2, True, 4]])
    def test_bad_cutpoints(self, cuts):
        with pytest.raises(ValueError):
            apply_cutpoints(np.array([1.0]), cuts)

    def test_safety_net(self):
        proba = np.array([[0.05, 0.10, 0.0, 0.25, 0.60], [0.0, 0.05, 0.05, 0.3, 0.6], [0.5, 0.5, 0, 0, 0]])
        pred = np.array([3, 4, 0])
        assert apply_safety_net(pred, proba, 0.15).tolist() == [2, 4, 0]     # nigdy nie obniża pilności
        assert apply_safety_net(pred, proba, None).tolist() == [3, 4, 0]

    def test_bayes_cost_matrix(self):
        np.testing.assert_array_equal(bayes_cost_matrix(0.0), (np.arange(5)[None, :] - np.arange(5)[:, None]) ** 2.0)
        extra = bayes_cost_matrix(2.5) - bayes_cost_matrix(0.0)
        assert (extra[:2, 2:] == 2.5).all() and extra.sum() == 2.5 * 6


class TestDecideRule:
    def test_argmax_and_cost_match_decide(self, data):
        proba, _, _ = data
        for rule in ("argmax", "cost"):
            np.testing.assert_array_equal(decide_rule(proba, {"rule": rule}), decide(proba, rule))

    def test_target_and_bayes(self, data):
        proba, _, _ = data
        spec = {"rule": "target", "cutpoints": [1.0, 1.9, 2.4, 3.0], "safety_q": 0.2}
        expected = apply_safety_net(apply_cutpoints(proba @ np.arange(5.0), spec["cutpoints"]), proba, 0.2)
        np.testing.assert_array_equal(decide_rule(proba, spec), expected)
        np.testing.assert_array_equal(decide_rule(proba, {"rule": "bayes_target", "lam": 3.0}),
                                      decide(proba, "cost", costs=bayes_cost_matrix(3.0)))

    @pytest.mark.parametrize("spec", [
        {"rule": "median"},
        {"rule": "target"},
        {"rule": "target", "cutpoints": [1, 2, 3, 4], "safety_q": 0.0},
        {"rule": "target", "cutpoints": [1, 2, 3, 4], "safety_q": 1.2},
        {"rule": "target", "cutpoints": [1, 2, 3, 4], "lam": 1.0},
        {"rule": "bayes_target"},
        {"rule": "bayes_target", "lam": 0},
        {"rule": "bayes_target", "lam": True},
        {"rule": "cost", "cutpoints": [1, 2, 3, 4]},
        {"rule": "argmax", "note": "x"},
        "argmax",
    ])
    def test_invalid_spec(self, spec):
        with pytest.raises(ValueError):
            decide_rule(np.full((2, 5), 0.2), spec)

    def test_normalize_spec(self):
        assert normalize_spec({"rule": "target", "cutpoints": (1, 2, 3, 4)}) == \
            {"rule": "target", "cutpoints": [1.0, 2.0, 3.0, 4.0], "safety_q": None}
        assert normalize_spec({}) == {"rule": "argmax"}
        assert RULES == ("argmax", "cost", "target", "bayes_target")

    def test_population_mask(self):
        vc = np.array([1, 0, 1], dtype=np.int8)
        assert population_mask(vc, "vc").tolist() == [True, False, True]
        assert population_mask(vc, "all") is None
        assert DECISION_FIT_POPULATION == "vc"
        with pytest.raises(ValueError):
            population_mask(vc, "site")


# ─────────────────────────────────────────
# B1 / B6 — reguła z celem undertriage
# ─────────────────────────────────────────
class TestTargetRule:
    @pytest.mark.parametrize("target", [0.05, 0.15, 0.30])
    def test_hits_target_on_fit_rows(self, data, target):
        proba, y, vc = data
        rule = fit_target_rule(y, proba, target, mask=vc)
        pred = rule.predict(proba)
        under = undertriage_rate(y[vc], pred[vc])
        n_high = int((y[vc] <= ESI2).sum())
        assert target - 2.0 / n_high <= under <= target
        assert rule.achieved["undertriage"] == pytest.approx(under)
        assert rule.fit_n == int(vc.sum())
        assert list(rule.cutpoints) == sorted(rule.cutpoints)
        # Klasa ≥ ESI3 dokładnie wtedy, gdy EL > próg ESI2/ESI3
        el = expected_level(proba)
        np.testing.assert_array_equal(pred >= ESI3, el > rule.cutpoints[1])

    def test_other_cutpoints_maximise_qwk(self, data):
        proba, y, vc = data
        rule = fit_target_rule(y, proba, 0.15, mask=vc)
        el = expected_level(proba[vc])
        best = quadratic_weighted_kappa(y[vc], rule.predict(proba[vc]))
        c = rule.cutpoints
        naive = quadratic_weighted_kappa(y[vc], apply_cutpoints(el, [min(0.5, c[1] - 1e-6), c[1], 2.5, 3.5]))
        assert best >= naive
        for j in (0, 2, 3):                           # małe przesunięcie pojedynczego progu nie pomaga
            for step in (-0.02, 0.02):
                trial = list(c)
                trial[j] += step
                if trial == sorted(trial):
                    assert quadratic_weighted_kappa(y[vc], apply_cutpoints(el, trial)) <= best + 1e-12
        assert rule.achieved["qwk"] == pytest.approx(best)

    def test_mask_selects_fit_rows(self, data):
        proba, y, vc = data
        rule = fit_target_rule(y, proba, 0.15, mask=vc)
        y_noise = y.copy()
        y_noise[~vc] = np.random.default_rng(5).integers(0, 5, (~vc).sum())
        assert fit_target_rule(y_noise, proba, 0.15, mask=vc) == rule
        assert fit_target_rule(y[vc], proba[vc], 0.15) .cutpoints == rule.cutpoints

    def test_guarantee_uses_upper_bound(self, data):
        proba, y, vc = data
        mean = fit_target_rule(y, proba, 0.15, mask=vc)
        guar = fit_target_rule(y, proba, 0.15, guarantee=True, mask=vc)
        assert guar.achieved["undertriage_upper"] <= 0.15 < mean.achieved["undertriage_upper"]
        assert guar.achieved["undertriage"] < mean.achieved["undertriage"]
        assert guar.cutpoints[1] > mean.cutpoints[1]
        assert guar.guarantee and not mean.guarantee

    def test_needs_urgent_patients_and_valid_target(self, data):
        proba, y, _ = data
        with pytest.raises(ValueError):
            fit_target_rule(np.full(len(y), ESI4), proba, 0.15)
        for bad in (0.0, 1.0, -0.1):
            with pytest.raises(ValueError):
                fit_target_rule(y, proba, bad)
        with pytest.raises(ValueError):
            fit_target_rule(y, proba, 0.15, mask=np.zeros(len(y), bool))

    def test_default_target(self):
        assert DECISION_TARGET_UNDERTRIAGE == 0.15

    def test_round_trip(self, data):
        proba, y, vc = data
        rule = fit_target_rule(y, proba, 0.15, mask=vc, safety_net=True)
        payload = json.loads(json.dumps(rule.to_dict()))          # JSON bez NaN i numpy
        restored = TargetRule.from_dict(payload)
        assert restored == rule
        assert rule_from_dict(payload) == rule
        np.testing.assert_array_equal(restored.predict(proba), rule.predict(proba))
        np.testing.assert_array_equal(decide_rule(proba, rule.spec()), rule.predict(proba))
        np.testing.assert_array_equal(decide_rule(proba, rule), rule.predict(proba))
        assert set(rule.spec()) == {"rule", "cutpoints", "safety_q"}


# ─────────────────────────────────────────
# B2 — siatka bezpieczeństwa
# ─────────────────────────────────────────
class TestSafetyNet:
    @pytest.mark.parametrize("max_miss", [0.02, 0.05])
    def test_largest_q_meeting_target(self, data, max_miss):
        proba, y, _ = data
        base = fit_target_rule(y, proba, 0.15).predict(proba)
        assert critical_miss_rate(y, base) > max_miss
        q = fit_safety_net(y, proba, base, max_miss)
        assert q is not None and q >= 0.05
        assert critical_miss_rate(y, apply_safety_net(base, proba, q)) <= max_miss
        bigger = np.nextafter(q, 1.0)
        assert critical_miss_rate(y, apply_safety_net(base, proba, bigger)) > max_miss
        assert SAFETY_NET_MAX_CRITICAL_MISS == 0.005

    def test_impossible_target_gives_minimal_miss(self):
        y = np.array([ESI1, ESI1, ESI1, ESI3])
        proba = np.array([[0.01, 0.02, 0.07, 0.4, 0.5],      # P(ESI1–2) = 0.03 < min_q
                          [0.10, 0.10, 0.10, 0.3, 0.4],      # 0.20
                          [0.05, 0.25, 0.10, 0.3, 0.3],      # 0.30
                          [0.0, 0.1, 0.8, 0.1, 0.0]])
        base = np.array([4, 3, 3, 2])
        q = fit_safety_net(y, proba, base, 0.0, min_q=0.05)
        assert q == pytest.approx(0.20)                       # łapie 2 z 3, mniej się nie da
        assert fit_safety_net(y, proba, base, 0.0, min_q=0.25) == pytest.approx(0.30)
        assert fit_safety_net(y, proba, base, 0.0, min_q=0.5) is None   # nic do złapania

    def test_none_when_not_needed(self):
        y = np.array([ESI1, ESI2, ESI3])
        proba = np.full((3, 5), 0.2)
        assert fit_safety_net(y, proba, np.array([0, 1, 2])) is None
        assert fit_safety_net(np.array([ESI2, ESI3, ESI4]), proba, np.array([3, 3, 3])) is None

    def test_mask(self):
        y = np.array([ESI1, ESI1])
        proba = np.array([[0.1, 0.1, 0.2, 0.3, 0.3], [0.2, 0.2, 0.2, 0.2, 0.2]])
        base = np.array([3, 3])
        assert fit_safety_net(y, proba, base, 0.0, mask=np.array([False, True])) == pytest.approx(0.4)
        assert fit_safety_net(y, proba, base, 0.0) == pytest.approx(0.2)

    @pytest.mark.parametrize("population", ["all", "mask"])
    def test_in_target_rule(self, data, population):
        proba, y, vc = data
        plain = fit_target_rule(y, proba, 0.15, mask=vc)
        net = fit_target_rule(y, proba, 0.15, mask=vc, safety_net=True, max_critical_miss=0.02,
                              safety_population=population)
        assert net.cutpoints == plain.cutpoints
        assert net.safety_q is not None
        rows = vc if population == "mask" else np.ones_like(vc)
        assert critical_miss_rate(y[rows], net.predict(proba)[rows]) <= 0.02
        assert net.achieved["max_critical_miss"] == 0.02
        assert net.achieved["safety_fit_critical_miss"] <= 0.02
        assert net.achieved["safety_fit_n_esi1"] == int((y[rows] == ESI1).sum())
        assert net.achieved["critical_miss"] <= net.achieved["critical_miss_base"]
        assert net.achieved["undertriage"] == plain.achieved["undertriage"]   # net nie zmienia ESI1–2 vs 3–5
        assert 0 < net.achieved["safety_changed"] < 0.5
        with pytest.raises(ValueError):
            fit_target_rule(y, proba, 0.15, safety_net=True, safety_population="vc")


# ─────────────────────────────────────────
# B4 — decyzja Bayesa z λ
# ─────────────────────────────────────────
class TestBayesRule:
    @pytest.mark.parametrize("target", [0.10, 0.20])
    def test_smallest_lambda_meeting_target(self, data, target):
        proba, y, vc = data
        rule = fit_bayes_rule(y, proba, target, mask=vc)
        assert rule.lam > 0
        under = undertriage_rate(y[vc], rule.predict(proba)[vc])
        assert under <= target
        smaller = decide(proba[vc], "cost", costs=bayes_cost_matrix(rule.lam * 0.99))
        assert undertriage_rate(y[vc], smaller) > target
        np.testing.assert_array_equal(np.asarray(rule.costs), bayes_cost_matrix(rule.lam))
        assert rule.achieved["undertriage"] == pytest.approx(under)

    def test_round_trip(self, data):
        proba, y, vc = data
        rule = fit_bayes_rule(y, proba, 0.15, mask=vc)
        payload = json.loads(json.dumps(rule.to_dict()))
        restored = BayesRule.from_dict(payload)
        assert restored == rule and rule_from_dict(payload) == rule
        np.testing.assert_array_equal(restored.predict(proba), rule.predict(proba))
        np.testing.assert_array_equal(decide_rule(proba, rule.spec()), rule.predict(proba))
        assert rule.spec() == {"rule": "bayes_target", "lam": rule.lam}

    def test_target_rule_has_higher_qwk(self, data):
        proba, y, vc = data
        t = fit_target_rule(y, proba, 0.15, mask=vc).predict(proba)
        b = fit_bayes_rule(y, proba, 0.15, mask=vc).predict(proba)
        assert quadratic_weighted_kappa(y[vc], t[vc]) >= quadratic_weighted_kappa(y[vc], b[vc])


# ─────────────────────────────────────────
# Cross-fitting i krzywa kompromisu
# ─────────────────────────────────────────
class TestCrossfitAndTradeoff:
    def test_crossfit_uses_other_folds(self, data):
        proba, y, vc = data
        calls = []

        def fitter(yy, pp, mask=None):
            calls.append((len(yy), None if mask is None else int(mask.sum())))
            return {"rule": "argmax"}

        pred = crossfit_rule(y, proba, fitter, n_folds=3, mask=vc)
        np.testing.assert_array_equal(pred, decide(proba, "argmax"))
        assert len(calls) == 3
        assert sum(n for n, _ in calls) == 2 * len(y)            # każdy wiersz w 2 z 3 dopasowań
        assert sum(m for _, m in calls) == 2 * int(vc.sum())
        with pytest.raises(ValueError):
            crossfit_rule(y, proba, fitter, n_folds=1)

    def test_crossfit_target_close_to_target(self, data):
        proba, y, vc = data
        pred = crossfit_rule(y, proba, partial(fit_target_rule, target_undertriage=0.15), mask=vc)
        assert abs(undertriage_rate(y[vc], pred[vc]) - 0.15) < 0.03

    def test_tradeoff_curve_and_plot(self, data, tmp_path):
        proba, y, vc = data
        df = tradeoff_curve(y, proba, targets=(0.10, 0.20), mask=vc, cost_multipliers=(1.0, 3.0))
        assert isinstance(df, pd.DataFrame)
        assert df["method"].value_counts().to_dict() == {
            "cost": 2, "target": 2, "target_safety": 2, "bayes_target": 2, "argmax": 1}
        assert (df["n"] == int(vc.sum())).all()
        for col in ("qwk", "undertriage", "critical_miss", "overtriage", "accuracy", "expected_cost"):
            assert df[col].notna().all(), col
        in_sample = tradeoff_curve(y, proba, targets=(0.10,), mask=vc, crossfit=False, cost_multipliers=())
        fitted = in_sample[in_sample["method"].isin(["target", "target_safety", "bayes_target"])]
        assert (fitted["undertriage"] <= 0.10).all()
        assert fitted["param"].notna().all()
        assert tuple(TRADEOFF_TARGETS)[0] == 0.05

        path = plot_tradeoff(df, tmp_path / "tradeoff.png", chosen={"method": "target_safety", "target": 0.10})
        assert path.exists() and path.stat().st_size > 0
        assert plot_tradeoff(df, tmp_path / "no_choice.png", chosen={"method": "target", "target": 0.33}).exists()
        with pytest.raises(ValueError):
            plot_tradeoff(df.drop(columns=["qwk"]), tmp_path / "x.png")
