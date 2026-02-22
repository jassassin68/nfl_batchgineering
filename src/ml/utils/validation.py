"""
Walk-Forward Cross-Validation utilities for NFL prediction models.

CRITICAL: Never use random splits for time-series data.
Walk-forward CV prevents look-ahead bias by only using past data to predict future games.
"""

from typing import Iterator, Tuple, Optional, Dict, List, Any
import polars as pl
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def walk_forward_cv(
    data: pl.DataFrame,
    n_seasons_train: int = 5,
    season_col: str = 'season',
    expanding: bool = True
) -> Iterator[Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Walk-forward cross-validation for time series data.

    CRITICAL: Never use random splits - this introduces temporal leakage.

    Train on seasons [t-n, t-1], test on season t.
    Repeat for each available test season.

    Args:
        data: DataFrame with game data (must include season column)
        n_seasons_train: Minimum seasons for initial training window
        season_col: Name of season column
        expanding: If True, use all prior data. If False, rolling window.

    Yields:
        (train_df, test_df) tuples for each test season

    Raises:
        ValueError: If insufficient seasons for CV

    Example:
        >>> for train, test in walk_forward_cv(games_df, n_seasons_train=5):
        ...     model.fit(train)
        ...     preds = model.predict(test)
        ...     evaluate(preds, test['actual_spread'])
    """
    seasons = sorted(data[season_col].unique().to_list())

    if len(seasons) < n_seasons_train + 1:
        raise ValueError(
            f"Need at least {n_seasons_train + 1} seasons for CV, "
            f"got {len(seasons)}"
        )

    for i, test_season in enumerate(seasons[n_seasons_train:], start=n_seasons_train):
        if expanding:
            # Use all data before test season (expanding window)
            train_seasons = seasons[:i]
        else:
            # Rolling window of n_seasons_train
            train_seasons = seasons[i - n_seasons_train:i]

        train_df = data.filter(pl.col(season_col).is_in(train_seasons))
        test_df = data.filter(pl.col(season_col) == test_season)

        yield train_df, test_df


def walk_forward_cv_arrays(
    X: np.ndarray,
    y: np.ndarray,
    seasons: np.ndarray,
    n_seasons_train: int = 5,
    expanding: bool = True
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    """
    Walk-forward CV for numpy arrays.

    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        seasons: Season array (n_samples,) - must be sortable
        n_seasons_train: Minimum training seasons
        expanding: Use expanding or rolling window

    Yields:
        (X_train, y_train, X_test, y_test, test_season) tuples

    Raises:
        ValueError: If insufficient seasons for CV
    """
    unique_seasons = np.unique(seasons)
    unique_seasons.sort()

    if len(unique_seasons) < n_seasons_train + 1:
        raise ValueError(
            f"Need at least {n_seasons_train + 1} seasons, "
            f"got {len(unique_seasons)}"
        )

    for i, test_season in enumerate(unique_seasons[n_seasons_train:], start=n_seasons_train):
        if expanding:
            train_mask = seasons < test_season
        else:
            train_seasons = unique_seasons[i - n_seasons_train:i]
            train_mask = np.isin(seasons, train_seasons)

        test_mask = seasons == test_season

        yield (
            X[train_mask], y[train_mask],
            X[test_mask], y[test_mask],
            int(test_season)
        )


def calculate_cv_metrics(
    all_predictions: List[np.ndarray],
    all_actuals: List[np.ndarray],
    all_seasons: List[int],
    vegas_spreads: Optional[List[np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Calculate aggregate metrics across CV folds.

    Args:
        all_predictions: List of prediction arrays per fold
        all_actuals: List of actual values per fold
        all_seasons: List of test seasons
        vegas_spreads: Optional list of Vegas spread arrays per fold

    Returns:
        Dictionary with per-season and aggregate metrics
    """
    results = {
        'per_season': {},
        'aggregate': {}
    }

    all_preds_flat = np.concatenate(all_predictions)
    all_actuals_flat = np.concatenate(all_actuals)

    # Per-season metrics
    for season, preds, actuals in zip(all_seasons, all_predictions, all_actuals):
        mae = mean_absolute_error(actuals, preds)
        rmse = np.sqrt(mean_squared_error(actuals, preds))

        # Directional accuracy (did model pick correct winner?)
        pred_direction = np.sign(preds)
        actual_direction = np.sign(actuals)
        directional_acc = np.mean(pred_direction == actual_direction)

        results['per_season'][season] = {
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_acc,
            'n_games': len(preds)
        }

    # Aggregate metrics
    mae = mean_absolute_error(all_actuals_flat, all_preds_flat)
    rmse = np.sqrt(mean_squared_error(all_actuals_flat, all_preds_flat))

    pred_direction = np.sign(all_preds_flat)
    actual_direction = np.sign(all_actuals_flat)
    directional_acc = np.mean(pred_direction == actual_direction)

    results['aggregate'] = {
        'mae': mae,
        'rmse': rmse,
        'directional_accuracy': directional_acc,
        'n_games': len(all_preds_flat),
        'n_seasons': len(all_seasons)
    }

    # ATS (Against The Spread) accuracy if Vegas spreads provided
    if vegas_spreads is not None:
        all_vegas_flat = np.concatenate(vegas_spreads)
        ats_correct = calculate_ats_accuracy(
            all_preds_flat, all_actuals_flat, all_vegas_flat
        )
        results['aggregate']['ats_accuracy'] = ats_correct

    return results


def calculate_ats_accuracy(
    predictions: np.ndarray,
    actuals: np.ndarray,
    vegas_spreads: np.ndarray,
    edge_threshold: float = 0.0
) -> float:
    """
    Calculate Against-The-Spread accuracy.

    ATS accuracy measures how often the model would beat the Vegas line.

    Args:
        predictions: Model predicted spreads
        actuals: Actual game spreads (home_score - away_score)
        vegas_spreads: Vegas spread lines (positive = home favored)
        edge_threshold: Minimum edge required to place bet (0 = bet everything)

    Returns:
        ATS accuracy as float between 0 and 1
    """
    # Edge = how much model disagrees with Vegas
    edges = predictions - vegas_spreads

    # Only count games where we have sufficient edge
    bet_mask = np.abs(edges) >= edge_threshold

    if bet_mask.sum() == 0:
        return 0.5  # No bets placed

    # Did we beat the spread?
    # If model says home should win by more than Vegas, bet home
    # Home covers if actual > vegas_spread
    model_says_home = edges > 0
    home_covered = actuals > vegas_spreads

    # ATS win: model direction matches actual cover
    ats_wins = model_says_home[bet_mask] == home_covered[bet_mask]

    return np.mean(ats_wins)


def calculate_roi(
    predictions: np.ndarray,
    actuals: np.ndarray,
    vegas_spreads: np.ndarray,
    edge_threshold: float = 3.0,
    odds: float = -110
) -> Dict[str, float]:
    """
    Calculate betting ROI with edge threshold.

    Args:
        predictions: Model predicted spreads
        actuals: Actual game spreads
        vegas_spreads: Vegas spread lines
        edge_threshold: Minimum points of edge to bet
        odds: American odds (default -110)

    Returns:
        Dictionary with ROI metrics
    """
    # Convert American odds to decimal
    if odds < 0:
        decimal_odds = 1 + (100 / abs(odds))
    else:
        decimal_odds = 1 + (odds / 100)

    profit_per_unit = decimal_odds - 1  # ~0.91 for -110

    edges = predictions - vegas_spreads
    bet_mask = np.abs(edges) >= edge_threshold

    if bet_mask.sum() == 0:
        return {
            'roi': 0.0,
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'pushes': 0,
            'profit': 0.0
        }

    model_says_home = edges[bet_mask] > 0
    actual_margin = actuals[bet_mask] - vegas_spreads[bet_mask]

    # Determine outcomes
    # Win if betting home and margin > 0, or betting away and margin < 0
    wins = np.sum(
        (model_says_home & (actual_margin > 0)) |
        (~model_says_home & (actual_margin < 0))
    )
    losses = np.sum(
        (model_says_home & (actual_margin < 0)) |
        (~model_says_home & (actual_margin > 0))
    )
    pushes = np.sum(actual_margin == 0)

    total_bets = len(model_says_home)
    profit = wins * profit_per_unit - losses * 1.0  # Lose 1 unit per loss

    return {
        'roi': profit / total_bets if total_bets > 0 else 0.0,
        'total_bets': total_bets,
        'wins': int(wins),
        'losses': int(losses),
        'pushes': int(pushes),
        'profit': profit,
        'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0.0
    }


def calculate_brier_score(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray
) -> float:
    """
    Calculate Brier Score for probability predictions.

    Brier Score measures calibration of probabilistic predictions.
    Lower is better. Perfect = 0, Random = 0.25.

    Args:
        predicted_probs: Predicted win probabilities [0, 1]
        actual_outcomes: Binary outcomes (1 = home win, 0 = away win)

    Returns:
        Brier score (lower is better)
    """
    return np.mean((predicted_probs - actual_outcomes) ** 2)


def train_test_split_temporal(
    data: pl.DataFrame,
    test_seasons: int = 1,
    season_col: str = 'season'
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Simple temporal train/test split.

    Uses last N seasons as test set.

    Args:
        data: DataFrame with game data
        test_seasons: Number of seasons for test set
        season_col: Name of season column

    Returns:
        (train_df, test_df) tuple
    """
    seasons = sorted(data[season_col].unique().to_list())

    if len(seasons) <= test_seasons:
        raise ValueError(
            f"Need more than {test_seasons} seasons, got {len(seasons)}"
        )

    train_seasons = seasons[:-test_seasons]
    test_season_list = seasons[-test_seasons:]

    train_df = data.filter(pl.col(season_col).is_in(train_seasons))
    test_df = data.filter(pl.col(season_col).is_in(test_season_list))

    return train_df, test_df
