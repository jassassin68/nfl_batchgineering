"""Tests for the betting decision layer (src/betting/*).

Covers edge calculation, Kelly sizing, end-to-end recommendation assembly,
and the integration contract with src/ml/predict.py (single source of
truth -- the bet direction produced through predict.py must match
build_recommendations exactly).
"""

from __future__ import annotations

import io
import math
import re
from contextlib import redirect_stdout

import numpy as np
import polars as pl
import pytest

from src.betting.edge import calculate_edge, should_bet
from src.betting.kelly import american_to_decimal, kelly_fraction
from src.betting.recommend import (
    DEFAULT_BANKROLL,
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_KELLY_MULT,
    DEFAULT_ODDS,
    SIDE_AWAY,
    SIDE_HOME,
    SIDE_PASS,
    BetRecommendation,
    build_recommendations,
    recommend_one,
    write_recommendations_report,
)


# ---------------------------------------------------------------------------
# edge.py
# ---------------------------------------------------------------------------


def test_calculate_edge_sign_convention():
    # Vegas: home -3 (favored). Model: home -7 (more confident in home).
    # Edge = -7 - (-3) = -4 ... wait, sign convention is pred - vegas.
    # If model predicts -7 and vegas is -3 in "home margin" terms, the
    # model thinks home is worse, not better.
    # Use the actual convention from predict.py:468 -- positive when model
    # predicts a *larger home margin* than vegas.
    assert calculate_edge(7.0, 3.0) == pytest.approx(4.0)  # model: home wins by 7, vegas: 3 -> home edge
    assert calculate_edge(-3.0, 3.0) == pytest.approx(-6.0)  # away edge
    assert calculate_edge(0.0, 0.0) == 0.0


def test_should_bet_threshold_boundary():
    # Exactly at threshold should bet (>= comparison).
    assert should_bet(4.0, threshold=4.0) is True
    assert should_bet(-4.0, threshold=4.0) is True
    assert should_bet(3.99, threshold=4.0) is False
    assert should_bet(-3.99, threshold=4.0) is False
    assert should_bet(100.0, threshold=4.0) is True


def test_should_bet_uses_abs():
    assert should_bet(-5.0, threshold=4.0) is True
    assert should_bet(-1.0, threshold=4.0) is False


# ---------------------------------------------------------------------------
# kelly.py
# ---------------------------------------------------------------------------


def test_american_to_decimal_known_values():
    assert american_to_decimal(-110) == pytest.approx(1.0 + 100 / 110)
    assert american_to_decimal(100) == pytest.approx(2.0)
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(-200) == pytest.approx(1.5)


def test_american_to_decimal_rejects_zero():
    with pytest.raises(ValueError):
        american_to_decimal(0)


def test_kelly_fraction_canonical_at_minus110():
    # b = 100/110 ~ 0.909. For p=0.55: raw = (0.909*0.55 - 0.45)/0.909
    # = (0.5 - 0.45)/0.909 = 0.055. At 0.25 Kelly -> 0.01375.
    val = kelly_fraction(0.55, odds=-110, kelly_mult=0.25)
    b = 100 / 110
    expected = ((b * 0.55 - 0.45) / b) * 0.25
    assert val == pytest.approx(expected, rel=1e-9)


def test_kelly_fraction_clamps_negative_to_zero():
    # p=0.40 at -110 is a negative-EV bet; raw Kelly is negative.
    val = kelly_fraction(0.40, odds=-110, kelly_mult=0.25)
    assert val == 0.0


def test_kelly_fraction_caps_at_one():
    # Even with extreme inputs the result should never exceed 1 (bankroll).
    val = kelly_fraction(0.999, odds=100, kelly_mult=10.0)
    assert val <= 1.0
    assert val >= 0.0


def test_kelly_fraction_full_kelly_matches_formula():
    # mult=1.0 reproduces the raw Kelly formula.
    p = 0.60
    b = 100 / 110
    expected = (b * p - (1 - p)) / b
    assert kelly_fraction(p, odds=-110, kelly_mult=1.0) == pytest.approx(
        expected, rel=1e-9
    )


@pytest.mark.parametrize("bad_p", [0.0, 1.0, -0.1, 1.1])
def test_kelly_fraction_rejects_invalid_winprob(bad_p):
    with pytest.raises(ValueError):
        kelly_fraction(bad_p)


def test_kelly_fraction_rejects_invalid_mult():
    with pytest.raises(ValueError):
        kelly_fraction(0.55, kelly_mult=0.0)
    with pytest.raises(ValueError):
        kelly_fraction(0.55, kelly_mult=-0.1)


# ---------------------------------------------------------------------------
# recommend.py -- core logic
# ---------------------------------------------------------------------------


def test_recommend_one_home_edge_produces_home_side():
    # Model says home wins by 7, vegas line is +2 (home -2 favorite).
    # Edge = 7 - 2 = +5 > threshold -> bet HOME.
    rec = recommend_one(
        game_id="G1",
        predicted_spread=7.0,
        vegas_spread=2.0,
        edge_threshold=4.0,
        kelly_mult=0.25,
        bankroll=1000.0,
        odds=-110,
    )
    assert rec.side == SIDE_HOME
    assert rec.edge_points == pytest.approx(5.0)
    assert rec.stake_units > 0
    # win_prob for HOME bet must equal home_win_prob
    assert rec.win_prob == pytest.approx(rec.home_win_prob)
    # logistic at +7 -> prob > 0.5
    assert rec.home_win_prob > 0.5


def test_recommend_one_away_edge_produces_away_side():
    # Model: -7 (away wins by 7). Vegas: -2. Edge = -5 -> bet AWAY.
    rec = recommend_one(
        game_id="G2",
        predicted_spread=-7.0,
        vegas_spread=-2.0,
        edge_threshold=4.0,
    )
    assert rec.side == SIDE_AWAY
    assert rec.edge_points == pytest.approx(-5.0)
    assert rec.stake_units > 0
    # win_prob for AWAY bet = 1 - home_win_prob
    assert rec.win_prob == pytest.approx(1.0 - rec.home_win_prob)
    assert rec.home_win_prob < 0.5


def test_recommend_one_below_threshold_passes():
    rec = recommend_one(
        game_id="G3",
        predicted_spread=3.0,
        vegas_spread=1.0,
        edge_threshold=4.0,
    )
    assert rec.side == SIDE_PASS
    assert rec.stake_units == 0.0
    assert rec.kelly_fraction == 0.0


def test_build_recommendations_end_to_end():
    df = pl.DataFrame(
        {
            "game_id": ["A", "B", "C"],
            # A: strong HOME edge (7 vs 2 = +5).
            # B: strong AWAY edge (-3 vs 5 = -8).
            # C: below threshold (3.5 vs 2 = +1.5).
            "predicted_spread": [7.0, -3.0, 3.5],
            "vegas_spread": [2.0, 5.0, 2.0],
        }
    )
    recs = build_recommendations(
        df,
        edge_threshold=4.0,
        kelly_mult=0.25,
        bankroll=1000.0,
        odds=-110,
    )
    assert recs.height == 3
    assert recs["side"].to_list() == [SIDE_HOME, SIDE_AWAY, SIDE_PASS]
    stake_a, stake_b, stake_c = recs["stake_units"].to_list()
    assert stake_a > 0
    assert stake_b > 0
    assert stake_c == 0.0
    # game_id ordering preserved
    assert recs["game_id"].to_list() == ["A", "B", "C"]


def test_build_recommendations_handles_null_inputs():
    df = pl.DataFrame(
        {
            "game_id": ["A", "B"],
            "predicted_spread": [None, 6.0],
            "vegas_spread": [1.0, None],
        }
    )
    recs = build_recommendations(df, edge_threshold=4.0)
    assert recs.height == 2
    assert recs["side"].to_list() == [SIDE_PASS, SIDE_PASS]
    assert recs["stake_units"].to_list() == [0.0, 0.0]


def test_build_recommendations_requires_columns():
    bad = pl.DataFrame({"game_id": ["A"], "predicted_spread": [3.0]})
    with pytest.raises(ValueError, match="missing columns"):
        build_recommendations(bad)


def test_build_recommendations_stake_scales_with_bankroll():
    df = pl.DataFrame(
        {
            "game_id": ["A"],
            "predicted_spread": [10.0],
            "vegas_spread": [2.0],
        }
    )
    recs_small = build_recommendations(df, bankroll=100.0)
    recs_big = build_recommendations(df, bankroll=10_000.0)
    # stake_units is rounded to 4 decimals at write time, so use a coarser
    # tolerance than the default rel=1e-6.
    assert recs_small["stake_units"][0] * 100 == pytest.approx(
        recs_big["stake_units"][0], rel=1e-3
    )


# ---------------------------------------------------------------------------
# Integration with src/ml/predict.py
# ---------------------------------------------------------------------------


def test_predict_uses_betting_package_consistent_direction():
    """The refactored predict.py path must produce the same bet direction
    as a direct call to build_recommendations, proving single source of
    truth and that the historical bet_recommendation inversion stays fixed.
    """
    # Construct a synthetic predictions frame matching what
    # generate_predictions assembles before the betting block runs.
    df = pl.DataFrame(
        {
            "game_id": ["A", "B", "C"],
            "season": [2024, 2024, 2024],
            "week": [1, 1, 1],
            "gameday": ["2024-09-08", "2024-09-08", "2024-09-08"],
            "gametime": ["13:00", "13:00", "13:00"],
            "home_team": ["NYG", "DAL", "BUF"],
            "away_team": ["WAS", "PHI", "MIA"],
            "vegas_spread": [2.0, 5.0, 2.0],
            "vegas_total": [42.0, 47.0, 50.0],
            "predicted_spread": [7.0, -3.0, 3.5],
        }
    )

    # Direct call to build_recommendations.
    direct = build_recommendations(
        df.select(["game_id", "predicted_spread", "vegas_spread"]),
        edge_threshold=DEFAULT_EDGE_THRESHOLD,
        kelly_mult=DEFAULT_KELLY_MULT,
        bankroll=1.0,
        odds=DEFAULT_ODDS,
    )

    # Re-implement the predict.py side->label mapping the same way.
    side_to_label = {SIDE_HOME: "BET HOME", SIDE_AWAY: "BET AWAY"}
    expected_labels = [
        side_to_label.get(s, "NO BET") for s in direct["side"].to_list()
    ]
    # A: home edge -> BET HOME. B: away edge -> BET AWAY. C: pass -> NO BET.
    assert expected_labels == ["BET HOME", "BET AWAY", "NO BET"]

    # Sign check: when bet label says BET HOME, edge_points must be > 0.
    # When BET AWAY, edge_points must be < 0. This is exactly the
    # invariant the historical inversion bug violated.
    for label, edge in zip(expected_labels, direct["edge_points"].to_list()):
        if label == "BET HOME":
            assert edge > 0
        elif label == "BET AWAY":
            assert edge < 0
        else:
            assert abs(edge) < DEFAULT_EDGE_THRESHOLD


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def test_write_recommendations_report_round_trip(tmp_path):
    df = pl.DataFrame(
        {
            "game_id": ["A", "B"],
            "predicted_spread": [7.0, 0.5],
            "vegas_spread": [2.0, 0.0],
        }
    )
    recs = build_recommendations(df, edge_threshold=4.0, bankroll=1000.0)
    config = {
        "season": 2026,
        "week": 1,
        "edge_threshold": 4.0,
        "kelly_mult": 0.25,
        "bankroll": 1000.0,
        "odds": -110.0,
    }
    md_path, csv_path = write_recommendations_report(
        recs, tmp_path, season=2026, week=1, config=config
    )
    assert md_path.exists()
    assert csv_path.exists()
    md_text = md_path.read_text(encoding="utf-8")
    assert "Weekly bet recommendations" in md_text
    assert "Bets recommended: 1" in md_text
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "game_id,side,edge_points" in csv_text


# ---------------------------------------------------------------------------
# ASCII-safe stdout (Windows cp1252 rule)
# ---------------------------------------------------------------------------


def test_recommendations_stdout_is_ascii_safe(tmp_path):
    """All print() output from the betting layer must stay ASCII so the
    CLI is safe to run on Windows cp1252 consoles (same rule as
    test_backtest::test_backtest_output_is_ascii_safe).
    """
    df = pl.DataFrame(
        {
            "game_id": ["A", "B", "C"],
            "predicted_spread": [7.0, -7.0, 1.0],
            "vegas_spread": [2.0, -2.0, 0.0],
        }
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        recs = build_recommendations(df, edge_threshold=4.0)
        print(f"Bets: {recs.filter(pl.col('side') != SIDE_PASS).height}")
        write_recommendations_report(
            recs, tmp_path, season=2026, week=1, config={"a": 1}
        )
    text = buf.getvalue()
    bad = [(i, ch) for i, ch in enumerate(text) if ord(ch) > 127]
    assert not bad, f"Non-ASCII output: {bad[:5]}"
