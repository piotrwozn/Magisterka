"""Testy metryk ewaluacji."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    critical_miss_rate,
    full_evaluation,
    overtriage_rate,
    quadratic_weighted_kappa,
    undertriage_rate,
)


class TestUndertriageRate:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1, 2, 3, 4])
        y_pred = np.array([0, 0, 1, 1, 2, 3, 4])
        assert undertriage_rate(y_true, y_pred) == 0.0

    def test_full_undertriage(self):
        """Wszystkie Red/Orange błędnie zaklasyfikowane jako Yellow+."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([2, 3, 2, 3])
        assert undertriage_rate(y_true, y_pred) == 1.0

    def test_partial_undertriage(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 2, 1, 3])  # 1 z 4 to undertriage  → 0.5
        assert undertriage_rate(y_true, y_pred) == 0.5

    def test_no_high_acuity(self):
        """Gdy nie ma Red/Orange w y_true, rate = NaN."""
        y_true = np.array([2, 3, 4])
        y_pred = np.array([2, 3, 4])
        assert np.isnan(undertriage_rate(y_true, y_pred))


class TestCriticalMissRate:
    def test_all_red_caught(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 1, 2])
        assert critical_miss_rate(y_true, y_pred) == 0.0

    def test_critical_miss(self):
        """Red sklasyfikowany jako Green/Blue (3 lub 4)."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([3, 4, 0, 1])  # 2 z 4 to critical miss
        assert critical_miss_rate(y_true, y_pred) == 0.5


class TestOvertriageRate:
    def test_no_overtriage(self):
        y_true = np.array([3, 4, 3, 4])
        y_pred = np.array([3, 4, 2, 3])
        assert overtriage_rate(y_true, y_pred) == 0.0

    def test_full_overtriage(self):
        y_true = np.array([3, 4])
        y_pred = np.array([0, 1])
        assert overtriage_rate(y_true, y_pred) == 1.0


class TestQuadraticWeightedKappa:
    def test_perfect_agreement(self):
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        qwk = quadratic_weighted_kappa(y_true, y_pred)
        assert pytest.approx(qwk, abs=1e-6) == 1.0

    def test_complete_disagreement_extremes(self):
        """QWK powinien być silnie ujemny dla ekstremalnie błędnych predykcji."""
        y_true = np.array([0, 0, 4, 4])
        y_pred = np.array([4, 4, 0, 0])
        qwk = quadratic_weighted_kappa(y_true, y_pred)
        assert qwk < 0


class TestFullEvaluation:
    def test_returns_all_keys(self):
        rng = np.random.default_rng(42)
        n = 100
        y_true = rng.integers(0, 5, n)
        y_pred = rng.integers(0, 5, n)
        y_proba = rng.dirichlet(np.ones(5), n)

        result = full_evaluation(y_true, y_pred, y_proba, print_report=False)

        # Wszystkie kluczowe metryki muszą być
        for key in [
            "quadratic_weighted_kappa",
            "f1_macro",
            "auc_macro",
            "undertriage_rate",
            "overtriage_rate",
            "critical_miss_rate",
            "confusion_matrix",
        ]:
            assert key in result, f"Brak klucza: {key}"

    def test_with_prefix(self):
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        y_proba = np.eye(5)
        result = full_evaluation(y_true, y_pred, y_proba, prefix="test", print_report=False)

        assert "test_quadratic_weighted_kappa" in result
        assert "test_undertriage_rate" in result
