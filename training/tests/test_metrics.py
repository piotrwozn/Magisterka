"""
Testy metryk ewaluacji (`src.evaluation.metrics`).

Semantyka ESI: indeks 0–4 = ESI 1–5; NIŻSZY indeks = WYŻSZA pilność.
Niedotriażowanie = przewidziano wyższy indeks (niższą pilność) niż prawdziwy.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    brier_score,
    cost_matrix,
    critical_miss_rate,
    decide,
    expected_cost,
    full_evaluation,
    linear_weighted_kappa,
    mistriage_rates,
    overtriage_rate,
    quadratic_weighted_kappa,
    undertriage_rate,
)
from src.utils.config import (
    CLASS_NAMES,
    N_CLASSES,
    OVERTRIAGE_COST_MULTIPLIER,
    UNDERTRIAGE_COST_MULTIPLIER,
)

ESI1, ESI2, ESI3, ESI4, ESI5 = range(5)


# ─────────────────────────────────────────
# Kappa
# ─────────────────────────────────────────
class TestKappa:
    def test_perfect_agreement(self):
        y = np.array([0, 1, 2, 3, 4, 2, 2])
        assert quadratic_weighted_kappa(y, y) == pytest.approx(1.0)
        assert linear_weighted_kappa(y, y) == pytest.approx(1.0)

    def test_extreme_disagreement_negative(self):
        y_true = np.array([0, 0, 4, 4])
        y_pred = np.array([4, 4, 0, 0])
        assert quadratic_weighted_kappa(y_true, y_pred) < 0

    def test_off_by_one_better_than_off_by_three(self):
        y_true = np.array([0, 1, 2, 3, 4] * 4)
        near = np.clip(y_true + 1, 0, 4)
        far = np.clip(y_true + 3, 0, 4)
        assert quadratic_weighted_kappa(y_true, near) > quadratic_weighted_kappa(y_true, far)


# ─────────────────────────────────────────
# Metryki bezpieczeństwa
# ─────────────────────────────────────────
class TestUndertriage:
    def test_perfect(self):
        y = np.array([0, 0, 1, 1, 2, 3, 4])
        assert undertriage_rate(y, y) == 0.0

    def test_high_acuity_to_esi3_plus_is_undertriage(self):
        """ESI 1–2 przewidziane jako ESI 3–5."""
        y_true = np.array([ESI1, ESI1, ESI2, ESI2])
        y_pred = np.array([ESI3, ESI4, ESI3, ESI5])
        assert undertriage_rate(y_true, y_pred) == 1.0

    def test_esi1_as_esi2_is_not_undertriage_rate(self):
        """ESI 1 → ESI 2 zostaje w grupie wysokiej pilności (liczy się w mistriage_under)."""
        y_true = np.array([ESI1, ESI2])
        y_pred = np.array([ESI2, ESI2])
        assert undertriage_rate(y_true, y_pred) == 0.0
        assert mistriage_rates(y_true, y_pred)["mistriage_under"] == 0.5

    def test_partial(self):
        y_true = np.array([ESI1, ESI1, ESI2, ESI2])
        y_pred = np.array([ESI1, ESI3, ESI2, ESI4])
        assert undertriage_rate(y_true, y_pred) == 0.5

    def test_overtriage_of_high_acuity_ignored(self):
        y_true = np.array([ESI2, ESI2])
        y_pred = np.array([ESI1, ESI1])
        assert undertriage_rate(y_true, y_pred) == 0.0

    def test_low_acuity_errors_ignored(self):
        y_true = np.array([ESI1, ESI3, ESI4])
        y_pred = np.array([ESI1, ESI5, ESI5])
        assert undertriage_rate(y_true, y_pred) == 0.0

    def test_no_high_acuity_is_nan(self):
        y = np.array([ESI3, ESI4, ESI5])
        assert np.isnan(undertriage_rate(y, y))


class TestCriticalMiss:
    def test_esi1_to_esi4_or_5(self):
        y_true = np.array([ESI1] * 4)
        y_pred = np.array([ESI4, ESI5, ESI1, ESI2])
        assert critical_miss_rate(y_true, y_pred) == 0.5

    def test_esi1_to_esi3_is_not_critical_miss(self):
        y_true = np.array([ESI1, ESI1])
        y_pred = np.array([ESI3, ESI2])
        assert critical_miss_rate(y_true, y_pred) == 0.0
        assert undertriage_rate(y_true, y_pred) == 0.5

    def test_esi2_never_critical_miss(self):
        y_true = np.array([ESI1, ESI2])
        y_pred = np.array([ESI1, ESI5])
        assert critical_miss_rate(y_true, y_pred) == 0.0

    def test_no_esi1_is_nan(self):
        assert np.isnan(critical_miss_rate(np.array([1, 2]), np.array([4, 4])))


class TestOvertriage:
    def test_low_acuity_to_esi1_or_2(self):
        y_true = np.array([ESI4, ESI5, ESI4, ESI5])
        y_pred = np.array([ESI1, ESI2, ESI3, ESI5])
        assert overtriage_rate(y_true, y_pred) == 0.5

    def test_low_acuity_to_esi3_not_overtriage(self):
        y_true = np.array([ESI4, ESI5])
        y_pred = np.array([ESI3, ESI3])
        assert overtriage_rate(y_true, y_pred) == 0.0

    def test_no_low_acuity_is_nan(self):
        assert np.isnan(overtriage_rate(np.array([0, 1, 2]), np.array([0, 1, 2])))


class TestMistriage:
    def test_definitions(self):
        y_true = np.array([ESI1, ESI2, ESI3, ESI4, ESI5])
        y_pred = np.array([ESI2, ESI2, ESI1, ESI5, ESI2])
        rates = mistriage_rates(y_true, y_pred)
        assert rates["mistriage_under"] == pytest.approx(2 / 5)   # ESI1→2, ESI4→5
        assert rates["mistriage_over"] == pytest.approx(2 / 5)    # ESI3→1, ESI5→2
        assert rates["within_one_accuracy"] == pytest.approx(3 / 5)

    def test_perfect(self):
        y = np.arange(5)
        assert mistriage_rates(y, y) == {
            "mistriage_under": 0.0, "mistriage_over": 0.0, "within_one_accuracy": 1.0,
        }


# ─────────────────────────────────────────
# Macierz kosztów i reguła decyzyjna
# ─────────────────────────────────────────
class TestCostMatrix:
    def test_shape_and_diagonal(self):
        c = cost_matrix()
        assert c.shape == (N_CLASSES, N_CLASSES)
        assert (np.diag(c) == 0).all()
        assert (c >= 0).all()

    def test_values(self):
        c = cost_matrix()
        assert c[ESI1, ESI2] == 3.0       # niedotriażowanie o 1 poziom: 1² × 3
        assert c[ESI2, ESI1] == 1.0       # przetriażowanie o 1 poziom: 1² × 1
        assert c[ESI1, ESI5] == 48.0      # 4² × 3
        assert c[ESI5, ESI1] == 16.0      # 4² × 1
        assert c[ESI2, ESI4] == 12.0      # 2² × 3

    def test_asymmetry(self):
        c = cost_matrix()
        ratio = UNDERTRIAGE_COST_MULTIPLIER / OVERTRIAGE_COST_MULTIPLIER
        for t in range(N_CLASSES):
            for p in range(t + 1, N_CLASSES):
                assert c[t, p] > c[p, t]
                assert c[t, p] == pytest.approx(ratio * c[p, t])

    def test_custom_multipliers(self):
        c = cost_matrix(n_classes=3, under=5.0, over=2.0)
        np.testing.assert_array_equal(c, [[0, 5, 20], [2, 0, 5], [8, 2, 0]])


class TestDecide:
    def test_argmax(self):
        proba = np.array([[0.1, 0.6, 0.2, 0.05, 0.05], [0.0, 0.0, 0.1, 0.2, 0.7]])
        np.testing.assert_array_equal(decide(proba, "argmax"), [ESI2, ESI5])

    def test_cost_shifts_to_higher_acuity(self):
        """Argmax = ESI 3, ale duże P(ESI 2) → reguła kosztowa wybiera ESI 2."""
        proba = np.array([[0.05, 0.40, 0.45, 0.10, 0.0]])
        assert decide(proba, "argmax")[0] == ESI3
        assert decide(proba, "cost")[0] == ESI2

    def test_cost_keeps_confident_prediction(self):
        proba = np.array([[0.0, 0.02, 0.96, 0.02, 0.0]])
        assert decide(proba, "cost")[0] == ESI3

    def test_cost_matches_bayes_rule(self):
        rng = np.random.default_rng(0)
        proba = rng.dirichlet(np.ones(N_CLASSES), 50)
        expected = (proba @ cost_matrix()).argmin(axis=1)
        np.testing.assert_array_equal(decide(proba, "cost"), expected)

    def test_cost_not_more_undertriage_than_argmax(self):
        """Na przykładzie z jednomodalnymi prawdopodobieństwami reguła kosztowa nie zwiększa niedotriażowania."""
        rng = np.random.default_rng(42)
        y = rng.choice(5, size=2000, p=[0.05, 0.3, 0.4, 0.2, 0.05])
        levels = np.arange(N_CLASSES)
        center = y + rng.normal(0.3, 0.8, size=len(y))     # model lekko zaniża pilność
        logits = -1.5 * (levels[None, :] - center[:, None]) ** 2
        proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

        pred_argmax = decide(proba, "argmax")
        pred_cost = decide(proba, "cost")
        assert undertriage_rate(y, pred_cost) <= undertriage_rate(y, pred_argmax)
        assert mistriage_rates(y, pred_cost)["mistriage_under"] <= mistriage_rates(y, pred_argmax)["mistriage_under"]
        assert expected_cost(y, pred_cost) <= expected_cost(y, pred_argmax)

    def test_custom_costs(self):
        proba = np.array([[0.3, 0.7]])
        costs = np.array([[0.0, 10.0], [1.0, 0.0]])
        assert decide(proba, "cost", costs=costs)[0] == 0

    def test_unknown_rule(self):
        with pytest.raises(ValueError, match="Nieznana reguła"):
            decide(np.eye(5), "median")


class TestExpectedCostAndBrier:
    def test_expected_cost(self):
        assert expected_cost(np.arange(5), np.arange(5)) == 0.0
        assert expected_cost(np.array([ESI1]), np.array([ESI5])) == 48.0
        assert expected_cost(np.array([ESI1, ESI2]), np.array([ESI2, ESI1])) == pytest.approx((3 + 1) / 2)

    def test_brier_perfect_and_uniform(self):
        y = np.arange(5)
        assert brier_score(y, np.eye(5)) == 0.0
        assert brier_score(y, np.full((5, 5), 0.2)) == pytest.approx(0.8)

    def test_brier_worst(self):
        y = np.array([0])
        assert brier_score(y, np.array([[0, 0, 0, 0, 1.0]])) == pytest.approx(2.0)


# ─────────────────────────────────────────
# Pełna ewaluacja
# ─────────────────────────────────────────
class TestFullEvaluation:
    @pytest.fixture
    def sample(self):
        rng = np.random.default_rng(42)
        n = 200
        y_true = np.resize(np.arange(5), n)
        y_proba = rng.dirichlet(np.ones(5), n)
        y_pred = y_proba.argmax(axis=1)
        return y_true, y_pred, y_proba

    def test_returns_all_keys(self, sample):
        result = full_evaluation(*sample, print_report=False)
        keys = [
            "quadratic_weighted_kappa", "linear_weighted_kappa", "cohen_kappa", "accuracy",
            "f1_macro", "f1_weighted", "auc_macro", "log_loss", "brier",
            "undertriage_rate", "overtriage_rate", "critical_miss_rate",
            "mistriage_under", "mistriage_over", "within_one_accuracy", "expected_cost",
            "confusion_matrix",
        ]
        for name in CLASS_NAMES:
            n = name.lower()
            keys += [f"auc_{n}", f"precision_{n}", f"recall_{n}", f"f1_{n}", f"support_{n}"]
        for key in keys:
            assert key in result, f"Brak klucza: {key}"
        assert "auc_esi1" in result and "auc_esi5" in result
        assert not [k for k in result if "orange" in k or "yellow" in k]

    def test_confusion_matrix_5x5(self, sample):
        y_true, y_pred, _ = sample
        cm = np.array(full_evaluation(*sample, print_report=False)["confusion_matrix"])
        assert cm.shape == (5, 5)
        assert cm.sum() == len(y_true)
        assert (cm.sum(axis=1) == np.bincount(y_true, minlength=5)).all()

    def test_values_consistent_with_functions(self, sample):
        y_true, y_pred, y_proba = sample
        result = full_evaluation(*sample, print_report=False)
        assert result["quadratic_weighted_kappa"] == pytest.approx(quadratic_weighted_kappa(y_true, y_pred))
        assert result["brier"] == pytest.approx(brier_score(y_true, y_proba))
        assert result["expected_cost"] == pytest.approx(expected_cost(y_true, y_pred))
        assert result["support_esi1"] == int((y_true == 0).sum())

    def test_with_prefix(self):
        y = np.arange(5)
        result = full_evaluation(y, y, np.eye(5), prefix="val_cost", print_report=False)
        assert result["val_cost_quadratic_weighted_kappa"] == pytest.approx(1.0)
        assert result["val_cost_undertriage_rate"] == 0.0
        assert result["val_cost_auc_esi1"] == pytest.approx(1.0)
        assert all(k.startswith("val_cost_") for k in result)

    def test_without_proba(self, sample):
        y_true, y_pred, _ = sample
        result = full_evaluation(y_true, y_pred, print_report=False)
        assert "auc_macro" not in result
        assert "quadratic_weighted_kappa" in result

    def test_print_report_runs(self, sample):
        full_evaluation(*sample, print_report=True, prefix="test")

    def test_missing_class_in_sample(self):
        y_true = np.array([ESI2, ESI3, ESI4, ESI5, ESI2, ESI3])
        y_pred = np.array([ESI2, ESI3, ESI4, ESI4, ESI3, ESI3])
        result = full_evaluation(y_true, y_pred, np.full((6, 5), 0.2), print_report=False)
        assert np.isnan(result["critical_miss_rate"])
        assert "brier" in result
