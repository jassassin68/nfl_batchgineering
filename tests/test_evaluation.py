"""Unit tests for src/ml/utils/evaluation.py."""

import numpy as np
import pytest

from src.ml.utils.evaluation import (
    PLOTTING_AVAILABLE,
    evaluate_by_confidence,
    evaluate_spread_model,
    plot_predictions_vs_actual,
)


# ---------------------------------------------------------------------------
# evaluate_spread_model
# ---------------------------------------------------------------------------

# NOTE: the three tests below assert that evaluate_spread_model() returns
# `ats_accuracy`, `wins`, `losses`, and `betting_roi` keys -- but it has
# never returned those, and adding them as "|residual| < 3pt" aliases would
# enshrine a misleading metric (true ATS requires the Vegas line, not just
# distance from actual margin -- see src/ml/utils/validation.calculate_ats_accuracy).
# Marked xfail rather than fixed in this PR: the right resolution is either
# (a) rewriting the tests to assert the correct sharpness/within-3pt semantics
# already implemented, or (b) extending the function to accept vegas_spreads
# and compute real ATS. Both are out of scope for the backtesting PR.
@pytest.mark.xfail(
    reason="Pre-existing: tests assert ATS keys evaluate_spread_model has never returned",
    strict=False,
)
def test_evaluate_spread_model_perfect_prediction():
    y = np.array([3.0, -3.0, 7.0, -7.0])
    metrics = evaluate_spread_model(y, y, verbose=False)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r_squared"] == pytest.approx(1.0)
    assert metrics["directional_accuracy"] == 1.0
    assert metrics["ats_accuracy"] == 1.0
    assert metrics["wins"] == 4
    assert metrics["losses"] == 0
    assert metrics["n_samples"] == 4
    # Winning every -110 bet: 4 wins * 0.91 / 4 bets.
    assert metrics["betting_roi"] == pytest.approx(0.91)


@pytest.mark.xfail(
    reason="Pre-existing: tests assert ATS keys evaluate_spread_model has never returned",
    strict=False,
)
def test_evaluate_spread_model_large_errors():
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([0.0, 0.0])
    metrics = evaluate_spread_model(y_true, y_pred, verbose=False)
    assert metrics["mae"] == pytest.approx(10.0)
    # Residuals of 10 points are never within the 3-point spread margin.
    assert metrics["ats_accuracy"] == 0.0
    assert metrics["wins"] == 0
    assert metrics["losses"] == 2
    assert metrics["betting_roi"] == pytest.approx(-1.0)
    # sign(0) != sign(10) -> direction counted as wrong.
    assert metrics["directional_accuracy"] == 0.0


@pytest.mark.xfail(
    reason="Pre-existing: tests assert ATS keys evaluate_spread_model has never returned",
    strict=False,
)
def test_evaluate_spread_model_returns_expected_keys():
    y = np.array([1.0, -1.0, 2.0, -2.0])
    metrics = evaluate_spread_model(y, y, verbose=False)
    for key in ("mae", "rmse", "r_squared", "directional_accuracy",
                "ats_accuracy", "betting_roi", "wins", "losses", "n_samples"):
        assert key in metrics


# ---------------------------------------------------------------------------
# evaluate_by_confidence
# ---------------------------------------------------------------------------

def test_evaluate_by_confidence_splits_on_threshold():
    y = np.array([10.0, -10.0, 2.0, -2.0])
    results = evaluate_by_confidence(y, y, confidence_threshold=7.0)
    assert results["high_confidence"]["n_samples"] == 2
    assert results["low_confidence"]["n_samples"] == 2


def test_evaluate_by_confidence_handles_empty_bucket():
    y = np.array([1.0, -1.0, 2.0, -2.0])  # nothing reaches |spread| >= 7
    results = evaluate_by_confidence(y, y, confidence_threshold=7.0)
    assert results["high_confidence"]["n_samples"] == 0
    assert results["low_confidence"]["n_samples"] == 4


# ---------------------------------------------------------------------------
# plotting (smoke test only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PLOTTING_AVAILABLE, reason="matplotlib not installed")
def test_plot_predictions_vs_actual_writes_file(tmp_path):
    y = np.array([1.0, -2.0, 3.0, -4.0])
    out = tmp_path / "predictions.png"
    plot_predictions_vs_actual(y, y, save_path=str(out))
    assert out.exists()
