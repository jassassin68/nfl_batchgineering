"""Unit tests for src/ml/utils/validation.py.

These functions are the betting/evaluation math the system relies on, so the
tests pin down exact, hand-computed values rather than smoke-testing shapes.
"""

import numpy as np
import polars as pl
import pytest

from src.ml.utils.validation import (
    calculate_ats_accuracy,
    calculate_brier_score,
    calculate_cv_metrics,
    calculate_roi,
    train_test_split_temporal,
)

# Decimal payout of a winning -110 bet: 1 + 100/110 - 1 = 0.90909...
PROFIT_PER_UNIT = 100 / 110


# ---------------------------------------------------------------------------
# calculate_ats_accuracy
# ---------------------------------------------------------------------------

def test_ats_accuracy_all_correct():
    predictions = np.array([10.0, -10.0, 10.0, -10.0])
    vegas = np.array([0.0, 0.0, 0.0, 0.0])
    actuals = np.array([5.0, -5.0, 5.0, -5.0])  # cover direction matches model
    assert calculate_ats_accuracy(predictions, actuals, vegas) == 1.0


def test_ats_accuracy_all_wrong():
    predictions = np.array([10.0, -10.0, 10.0, -10.0])
    vegas = np.array([0.0, 0.0, 0.0, 0.0])
    actuals = np.array([-5.0, 5.0, -5.0, 5.0])  # cover direction opposite model
    assert calculate_ats_accuracy(predictions, actuals, vegas) == 0.0


def test_ats_accuracy_edge_threshold_filters_bets():
    # Only the first two games have |edge| >= 3.
    predictions = np.array([10.0, -10.0, 1.0, -1.0])
    vegas = np.array([0.0, 0.0, 0.0, 0.0])
    actuals = np.array([5.0, -5.0, 5.0, -5.0])
    # Filtered games would be wrong, but they are excluded by the threshold.
    assert calculate_ats_accuracy(predictions, actuals, vegas, edge_threshold=3.0) == 1.0


def test_ats_accuracy_no_bets_returns_half():
    predictions = np.array([1.0, -1.0])
    vegas = np.array([0.0, 0.0])
    actuals = np.array([5.0, -5.0])
    assert calculate_ats_accuracy(predictions, actuals, vegas, edge_threshold=100.0) == 0.5


# ---------------------------------------------------------------------------
# calculate_roi
# ---------------------------------------------------------------------------

def test_roi_all_wins():
    predictions = np.array([10.0, -10.0])
    vegas = np.array([0.0, 0.0])
    actuals = np.array([5.0, -5.0])
    result = calculate_roi(predictions, actuals, vegas, edge_threshold=3.0)
    assert result["total_bets"] == 2
    assert result["wins"] == 2
    assert result["losses"] == 0
    assert result["win_rate"] == 1.0
    assert result["roi"] == pytest.approx(PROFIT_PER_UNIT)


def test_roi_all_losses():
    predictions = np.array([10.0, -10.0])
    vegas = np.array([0.0, 0.0])
    actuals = np.array([-5.0, 5.0])
    result = calculate_roi(predictions, actuals, vegas, edge_threshold=3.0)
    assert result["wins"] == 0
    assert result["losses"] == 2
    assert result["roi"] == pytest.approx(-1.0)


def test_roi_push_is_not_win_or_loss():
    predictions = np.array([10.0])
    vegas = np.array([0.0])
    actuals = np.array([0.0])  # actual margin exactly equals vegas line
    result = calculate_roi(predictions, actuals, vegas, edge_threshold=3.0)
    assert result["pushes"] == 1
    assert result["wins"] == 0
    assert result["losses"] == 0
    assert result["roi"] == 0.0


def test_roi_no_bets_returns_zeroed_dict():
    predictions = np.array([1.0, -1.0])
    vegas = np.array([0.0, 0.0])
    actuals = np.array([5.0, -5.0])
    result = calculate_roi(predictions, actuals, vegas, edge_threshold=100.0)
    assert result["total_bets"] == 0
    assert result["roi"] == 0.0
    assert "win_rate" not in result  # no-bet branch omits win_rate


# ---------------------------------------------------------------------------
# calculate_brier_score
# ---------------------------------------------------------------------------

def test_brier_score_perfect():
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    outcomes = np.array([1.0, 0.0, 1.0, 0.0])
    assert calculate_brier_score(probs, outcomes) == 0.0


def test_brier_score_coin_flip():
    probs = np.array([0.5, 0.5])
    outcomes = np.array([1.0, 0.0])
    assert calculate_brier_score(probs, outcomes) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# calculate_cv_metrics
# ---------------------------------------------------------------------------

def test_cv_metrics_perfect_predictions():
    preds = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    actuals = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    seasons = [2020, 2021]
    result = calculate_cv_metrics(preds, actuals, seasons)
    assert result["aggregate"]["mae"] == 0.0
    assert result["aggregate"]["rmse"] == 0.0
    assert result["aggregate"]["n_games"] == 4
    assert result["aggregate"]["n_seasons"] == 2
    assert result["per_season"][2020]["mae"] == 0.0
    assert result["per_season"][2020]["directional_accuracy"] == 1.0


def test_cv_metrics_includes_ats_when_vegas_supplied():
    preds = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    actuals = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    seasons = [2020, 2021]
    vegas = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
    result = calculate_cv_metrics(preds, actuals, seasons, vegas_spreads=vegas)
    assert result["aggregate"]["ats_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# train_test_split_temporal
# ---------------------------------------------------------------------------

def test_temporal_split_uses_latest_season_as_test(seasons_dataframe):
    train_df, test_df = train_test_split_temporal(seasons_dataframe, test_seasons=1)
    assert test_df["season"].unique().to_list() == [2022]
    assert train_df["season"].max() == 2021
    assert train_df["season"].min() == 2015


def test_temporal_split_raises_when_too_few_seasons():
    df = pl.DataFrame({"season": [2021, 2021]})
    with pytest.raises(ValueError):
        train_test_split_temporal(df, test_seasons=1)
