"""Look-ahead-bias guard tests.

The system's #1 reliability risk is temporal leakage: using future data to
predict past games silently inflates backtest results and costs real money.
These tests codify the invariant that training windows never see the test
season — if anyone weakens walk-forward CV, these fail loudly.
"""

import numpy as np
import polars as pl
import pytest

from src.ml.utils.validation import (
    train_test_split_temporal,
    walk_forward_cv,
    walk_forward_cv_arrays,
)


# ---------------------------------------------------------------------------
# walk_forward_cv (Polars DataFrame variant)
# ---------------------------------------------------------------------------

def test_walk_forward_cv_train_strictly_precedes_test(seasons_dataframe):
    folds = list(walk_forward_cv(seasons_dataframe, n_seasons_train=5))
    assert len(folds) == 3  # 8 seasons, 5 reserved for the initial window
    for train_df, test_df in folds:
        train_max = train_df["season"].max()
        test_min = test_df["season"].min()
        test_seasons = test_df["season"].unique().to_list()
        # The core invariant: no training season is >= any test season.
        assert train_max < test_min
        # And each fold tests exactly one season.
        assert len(test_seasons) == 1


def test_walk_forward_cv_rolling_window_excludes_future(seasons_dataframe):
    for train_df, test_df in walk_forward_cv(
        seasons_dataframe, n_seasons_train=5, expanding=False
    ):
        assert train_df["season"].max() < test_df["season"].min()
        # Rolling window keeps at most n_seasons_train seasons.
        assert train_df["season"].n_unique() <= 5


def test_walk_forward_cv_raises_with_insufficient_seasons():
    df = pl.DataFrame({"season": [2019, 2020, 2021]})
    with pytest.raises(ValueError):
        list(walk_forward_cv(df, n_seasons_train=5))


# ---------------------------------------------------------------------------
# walk_forward_cv_arrays (NumPy variant)
# ---------------------------------------------------------------------------

def test_walk_forward_cv_arrays_no_future_leakage():
    seasons = np.repeat(np.arange(2015, 2023), 3)
    X = np.arange(len(seasons), dtype=float).reshape(-1, 1)
    y = seasons.astype(float)

    folds = list(walk_forward_cv_arrays(X, y, seasons, n_seasons_train=5))
    assert len(folds) == 3
    for _, y_train, _, _, test_season in folds:
        # Every training label's season is strictly before the test season.
        assert np.all(y_train < test_season)


def test_walk_forward_cv_arrays_test_mask_is_single_season():
    seasons = np.repeat(np.arange(2015, 2023), 3)
    X = np.arange(len(seasons), dtype=float).reshape(-1, 1)
    y = seasons.astype(float)
    for _, _, _, y_test, test_season in walk_forward_cv_arrays(
        X, y, seasons, n_seasons_train=5
    ):
        assert np.all(y_test == test_season)


# ---------------------------------------------------------------------------
# train_test_split_temporal
# ---------------------------------------------------------------------------

def test_temporal_split_test_set_is_strictly_latest(seasons_dataframe):
    train_df, test_df = train_test_split_temporal(seasons_dataframe, test_seasons=2)
    assert sorted(test_df["season"].unique().to_list()) == [2021, 2022]
    assert train_df["season"].max() < test_df["season"].min()
