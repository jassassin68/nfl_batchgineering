"""Tests for the walk-forward backtest harness.

Covers the parts of src/ml/backtest.py that do NOT require Snowflake or
trained models: walk-forward split correctness, per-fold metric assembly,
report formatting, and ASCII-safe stdout. The XGBoost/Elo backtest functions
themselves are exercised against a synthetic 6-season DataFrame so we can
prove temporal ordering, but without hitting Snowflake.
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.ml.backtest import (
    ATS_TARGET,
    BACKTEST_REGISTRY,
    FoldMetrics,
    _ats_accuracy,
    _aggregate,
    _summarize_fold,
    _walk_forward_season_splits,
    backtest_elo,
    backtest_xgboost,
    run_full_backtest,
    write_report,
)


def _synthetic_game_features(
    seasons: list[int], games_per_week: int = 4, weeks: int = 6, seed: int = 7
) -> pl.DataFrame:
    """Build a synthetic mart_game_prediction_features-shaped frame.

    The signal: ``home_epa_l4w - away_epa_l4w`` correlates with the realized
    margin, so a model that learns it can beat random. Vegas spreads are a
    noisy version of true margin so a smart model can produce non-trivial ATS.
    """
    rng = np.random.RandomState(seed)
    teams = [f"T{i:02d}" for i in range(2 * games_per_week)]

    rows = []
    for season in seasons:
        for week in range(2, weeks + 2):  # week >= 2 to satisfy dbt range
            shuffled = rng.permutation(teams)
            for i in range(games_per_week):
                home, away = shuffled[2 * i], shuffled[2 * i + 1]
                home_epa = rng.normal(0.05, 0.08)
                away_epa = rng.normal(0.05, 0.08)
                # Margin ~ 25 * EPA diff + home advantage + noise.
                true_margin = 25.0 * (home_epa - away_epa) + 2.5
                noise = rng.normal(0, 13.0)
                actual_margin = true_margin + noise
                home_score = max(0, int(round(21 + actual_margin / 2)))
                away_score = max(0, int(round(21 - actual_margin / 2)))
                vegas_spread = float(round(true_margin + rng.normal(0, 1.5)))
                rows.append(
                    {
                        "game_id": f"{season}_W{week:02d}_{home}_{away}",
                        "season": season,
                        "week": week,
                        "home_team": home,
                        "away_team": away,
                        "home_score": home_score,
                        "away_score": away_score,
                        "vegas_spread": vegas_spread,
                        "vegas_total": float(home_score + away_score),
                        "temp": 60.0,
                        "wind": 5.0,
                        "div_game": 0,
                        "playoff": 0,
                        "home_epa_adj": home_epa,
                        "home_epa_l4w": home_epa,
                        "home_success_rate": 0.45,
                        "home_success_l4w": 0.45,
                        "home_explosive_rate": 0.10,
                        "home_pass_epa": home_epa,
                        "home_run_epa": home_epa * 0.5,
                        "home_def_epa": -away_epa,
                        "home_def_rank": 16,
                        "home_def_pass_epa": -away_epa,
                        "home_def_run_epa": -away_epa * 0.5,
                        "home_def_epa_l4w": -away_epa,
                        "home_rz_td_rate": 0.55,
                        "home_third_conv": 0.40,
                        "home_two_min_epa": 0.1,
                        "away_epa_adj": away_epa,
                        "away_epa_l4w": away_epa,
                        "away_success_rate": 0.45,
                        "away_success_l4w": 0.45,
                        "away_explosive_rate": 0.10,
                        "away_pass_epa": away_epa,
                        "away_run_epa": away_epa * 0.5,
                        "away_def_epa": -home_epa,
                        "away_def_rank": 16,
                        "away_def_pass_epa": -home_epa,
                        "away_def_run_epa": -home_epa * 0.5,
                        "away_def_epa_l4w": -home_epa,
                        "away_rz_td_rate": 0.55,
                        "away_third_conv": 0.40,
                        "away_two_min_epa": 0.1,
                    }
                )
    return pl.DataFrame(rows)


def test_walk_forward_splits_enforce_temporal_order():
    splits = _walk_forward_season_splits([2018, 2019, 2020, 2021, 2022, 2023], 3)
    assert len(splits) == 3
    for train, test in splits:
        assert max(train) < test
        assert sorted(train) == train  # ascending


def test_walk_forward_splits_require_min_seasons():
    with pytest.raises(ValueError, match="walk-forward CV"):
        _walk_forward_season_splits([2020, 2021], 3)


def test_ats_accuracy_pushes_count_as_half():
    # 3 picks: 1 win, 1 loss, 1 push. Expect score = (1 + 0 + 0.5)/3 = 0.5.
    preds = np.array([3.0, -3.0, 3.0])
    vegas = np.array([0.0, 0.0, 0.0])
    actuals = np.array([5.0, 4.0, 0.0])  # win, loss, push
    score, n = _ats_accuracy(preds, actuals, vegas, edge_threshold=0.0)
    assert n == 3
    assert score == pytest.approx(0.5)


def test_ats_accuracy_edge_threshold_filters_bets():
    preds = np.array([1.0, 5.0, -5.0])
    vegas = np.array([0.0, 0.0, 0.0])
    actuals = np.array([0.0, 6.0, -6.0])  # win, win, win
    score, n = _ats_accuracy(preds, actuals, vegas, edge_threshold=3.0)
    # First pick has |edge|=1 < 3 -> filtered out. The remaining two both win.
    assert n == 2
    assert score == pytest.approx(1.0)


def test_summarize_fold_assembles_expected_fields():
    preds = np.array([4.0, -4.0, 1.0])
    actuals = np.array([6.0, -6.0, 0.0])
    vegas = np.array([2.0, -2.0, 0.0])
    fold = _summarize_fold(
        model_name="dummy",
        test_season=2023,
        train_seasons=[2018, 2019, 2020, 2021, 2022],
        predictions=preds,
        actuals=actuals,
        vegas_spreads=vegas,
        edge_threshold=3.0,
        odds=-110,
    )
    expected_keys = {
        "model",
        "test_season",
        "n_games",
        "train_seasons",
        "ats_accuracy_all",
        "ats_accuracy_edge",
        "n_bets_edge",
        "brier_score",
        "rmse_vs_actual",
        "rmse_vs_vegas_actual",
        "roi",
        "wins",
        "losses",
        "pushes",
        "profit_units",
        "cleared_ats_target",
    }
    assert set(asdict(fold).keys()) == expected_keys
    assert fold.train_seasons == "2018-2022"
    assert fold.n_games == 3


def test_backtest_elo_respects_temporal_ordering():
    # Six seasons so n_seasons_train=4 yields 2 folds.
    df = _synthetic_game_features(seasons=list(range(2018, 2024)), weeks=4)
    folds = backtest_elo(
        df, n_seasons_train=4, edge_threshold=3.0, odds=-110
    )
    assert len(folds) >= 2
    # The first test season must be the 5th in chronological order.
    assert folds[0].test_season == 2022
    # Every fold's train range must end before its test season.
    for f in folds:
        last_train = int(f.train_seasons.split("-")[-1])
        assert last_train < f.test_season


def test_backtest_xgboost_respects_temporal_ordering():
    df = _synthetic_game_features(seasons=list(range(2018, 2024)), weeks=4)
    folds = backtest_xgboost(
        df, n_seasons_train=4, edge_threshold=3.0, odds=-110
    )
    assert len(folds) >= 2
    for f in folds:
        last_train = int(f.train_seasons.split("-")[-1])
        assert last_train < f.test_season
        # The xgboost backtest should at minimum produce finite metrics.
        assert np.isfinite(f.rmse_vs_actual)
        assert np.isfinite(f.brier_score)


def test_run_full_backtest_invokes_every_registered_model():
    df = _synthetic_game_features(seasons=list(range(2018, 2024)), weeks=4)
    results = run_full_backtest(
        df, n_seasons_train=4, edge_threshold=3.0, odds=-110
    )
    assert set(results.keys()) == set(BACKTEST_REGISTRY.keys())
    for model_name, folds in results.items():
        assert folds, f"{model_name} produced no folds"


def test_aggregate_weights_by_game_count():
    folds = [
        FoldMetrics(
            model="dummy",
            test_season=2022,
            n_games=10,
            train_seasons="2018-2021",
            ats_accuracy_all=0.6,
            ats_accuracy_edge=0.6,
            n_bets_edge=10,
            brier_score=0.20,
            rmse_vs_actual=12.0,
            rmse_vs_vegas_actual=13.0,
            roi=0.05,
            wins=6,
            losses=4,
            pushes=0,
            profit_units=1.46,
            cleared_ats_target=True,
        ),
        FoldMetrics(
            model="dummy",
            test_season=2023,
            n_games=20,
            train_seasons="2018-2022",
            ats_accuracy_all=0.45,
            ats_accuracy_edge=0.45,
            n_bets_edge=20,
            brier_score=0.30,
            rmse_vs_actual=14.0,
            rmse_vs_vegas_actual=13.0,
            roi=-0.10,
            wins=9,
            losses=11,
            pushes=0,
            profit_units=-2.81,
            cleared_ats_target=False,
        ),
    ]
    agg = _aggregate(folds)
    # Game-weighted: (10*0.6 + 20*0.45)/30 = 0.5
    assert agg["ats_accuracy_edge"] == pytest.approx(0.5)
    assert agg["folds_clearing_ats_target"] == 1
    assert agg["total_games"] == 30


def test_write_report_round_trips_csv_and_emits_markdown(tmp_path):
    df = _synthetic_game_features(seasons=list(range(2018, 2024)), weeks=4)
    results = run_full_backtest(
        df, n_seasons_train=4, edge_threshold=3.0, odds=-110
    )
    md_path, csv_path = write_report(
        results,
        tmp_path,
        config={
            "start_season": 2018,
            "end_season": 2023,
            "models": list(BACKTEST_REGISTRY.keys()),
            "n_seasons_train": 4,
            "edge_threshold": 3.0,
            "odds": -110,
            "ats_target": ATS_TARGET,
        },
    )
    assert md_path.exists()
    assert csv_path.exists()

    md_text = md_path.read_text(encoding="utf-8")
    assert "Backtest report" in md_text
    assert "Per-model summary" in md_text
    for model_name in BACKTEST_REGISTRY:
        assert model_name in md_text

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "model,test_season" in csv_text


def test_backtest_output_is_ascii_safe():
    """Windows cp1252 console crashes on non-ASCII print(). All stdout from
    the backtest harness must stay ASCII so the script is safe to run on the
    user's machine.
    """
    df = _synthetic_game_features(seasons=list(range(2018, 2024)), weeks=4)
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_full_backtest(
            df, n_seasons_train=4, edge_threshold=3.0, odds=-110
        )
    text = buf.getvalue()
    # If any non-ASCII slipped in, locate it so the failure message is useful.
    bad = [(i, ch) for i, ch in enumerate(text) if ord(ch) > 127]
    assert not bad, f"Non-ASCII output found: {bad[:5]}"
