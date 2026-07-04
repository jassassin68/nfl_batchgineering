"""Tests for closing-line value and bet settlement (src/betting/clv.py).

Spread convention under test: positive vegas_spread = HOME favored;
margin = home_score - away_score. CLV is positive when the line moved in
the bettor's favor after they placed the bet.
"""

from __future__ import annotations

import pytest

from src.betting.clv import (
    OUTCOME_LOSS,
    OUTCOME_PUSH,
    OUTCOME_WIN,
    clv_points,
    profit_units,
    settle_bet,
    settle_outcome,
)
from src.betting.kelly import american_to_decimal

# ---------------------------------------------------------------------------
# clv_points -- the five canonical cases confirmed in the Step E plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "at_rec, close, side, expected",
    [
        # Home bet, line moved in our favor (home got more favored).
        (3.0, 4.0, "home", 1.0),
        # Home bet, line moved against us.
        (3.0, 2.0, "home", -1.0),
        # Away bet, line moved in our favor (home got less favored).
        (3.0, 2.0, "away", 1.0),
        # Away bet, line moved against us.
        (3.0, 4.0, "away", -1.0),
        # No movement -> zero CLV either way.
        (3.0, 3.0, "home", 0.0),
        (3.0, 3.0, "away", 0.0),
    ],
)
def test_clv_points_canonical(at_rec, close, side, expected):
    assert clv_points(at_rec, close, side) == pytest.approx(expected)


def test_clv_points_sign_is_opposite_for_each_side():
    # The same line move is good for one side and bad for the other.
    home = clv_points(3.0, 5.0, "home")
    away = clv_points(3.0, 5.0, "away")
    assert home == pytest.approx(-away)
    assert home == pytest.approx(2.0)


def test_clv_points_rejects_bad_side():
    with pytest.raises(ValueError):
        clv_points(3.0, 4.0, "pass")


# ---------------------------------------------------------------------------
# settle_outcome
# ---------------------------------------------------------------------------


def test_settle_outcome_home_bet_covers():
    # Home favored by 3, wins by 7 -> covers -> home bet wins.
    assert settle_outcome(margin=7.0, vegas_spread_at_rec=3.0, side="home") == OUTCOME_WIN


def test_settle_outcome_home_bet_fails_to_cover():
    # Home favored by 3, wins by only 1 -> does not cover -> home bet loses.
    assert settle_outcome(margin=1.0, vegas_spread_at_rec=3.0, side="home") == OUTCOME_LOSS


def test_settle_outcome_away_bet_wins_when_home_underperforms():
    # Home favored by 3, wins by only 1 -> away covers -> away bet wins.
    assert settle_outcome(margin=1.0, vegas_spread_at_rec=3.0, side="away") == OUTCOME_WIN


def test_settle_outcome_away_bet_loses_when_home_covers():
    assert settle_outcome(margin=7.0, vegas_spread_at_rec=3.0, side="away") == OUTCOME_LOSS


def test_settle_outcome_push_on_the_number():
    assert settle_outcome(margin=3.0, vegas_spread_at_rec=3.0, side="home") == OUTCOME_PUSH
    assert settle_outcome(margin=3.0, vegas_spread_at_rec=3.0, side="away") == OUTCOME_PUSH


def test_settle_outcome_home_underdog_covers():
    # Negative spread = home underdog. Home loses by 1 but spread was -3,
    # so home (the dog) covers -> home bet wins.
    assert settle_outcome(margin=-1.0, vegas_spread_at_rec=-3.0, side="home") == OUTCOME_WIN


# ---------------------------------------------------------------------------
# profit_units
# ---------------------------------------------------------------------------


def test_profit_units_win_at_minus_110():
    b = american_to_decimal(-110) - 1.0  # ~0.909
    assert profit_units(OUTCOME_WIN, stake_units=10.0, odds=-110) == pytest.approx(
        10.0 * b
    )


def test_profit_units_loss_returns_negative_stake():
    assert profit_units(OUTCOME_LOSS, stake_units=10.0, odds=-110) == pytest.approx(-10.0)


def test_profit_units_push_is_zero():
    assert profit_units(OUTCOME_PUSH, stake_units=10.0, odds=-110) == 0.0


def test_profit_units_rejects_bad_outcome():
    with pytest.raises(ValueError):
        profit_units("voided", stake_units=10.0)


# ---------------------------------------------------------------------------
# settle_bet -- composition
# ---------------------------------------------------------------------------


def test_settle_bet_full_win():
    # Home favored by 3 at rec, closed at 5 (line moved toward home), home
    # wins by 7. Home bet: win, positive profit, positive CLV.
    out = settle_bet(
        margin=7.0,
        vegas_spread_at_rec=3.0,
        vegas_spread_close=5.0,
        side="home",
        stake_units=10.0,
        odds=-110,
    )
    assert out["outcome"] == OUTCOME_WIN
    assert out["profit_units"] > 0
    assert out["clv_points"] == pytest.approx(2.0)


def test_settle_bet_clv_none_when_close_missing():
    out = settle_bet(
        margin=7.0,
        vegas_spread_at_rec=3.0,
        vegas_spread_close=None,
        side="home",
        stake_units=10.0,
    )
    assert out["outcome"] == OUTCOME_WIN
    assert out["clv_points"] is None
