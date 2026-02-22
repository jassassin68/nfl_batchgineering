"""
Feature engineering utilities for NFL prediction models.
Handles feature selection, creation, and transformation.
"""

import polars as pl
from typing import List, Dict, Optional
import numpy as np


def select_spread_features(df: pl.DataFrame) -> List[str]:
    """
    Select relevant features for spread prediction from game features dataset.

    Args:
        df: Polars DataFrame with game-level features

    Returns:
        List of column names to use as features
    """

    # Core offensive EPA features (home team)
    home_offensive_features = [
        'home_epa_adj',
        'home_epa_l4w',
        'home_success_rate',
        'home_success_l4w',
        'home_explosive_rate',
        'home_pass_epa',
        'home_run_epa',
    ]

    # Core offensive EPA features (away team)
    away_offensive_features = [
        'away_epa_adj',
        'away_epa_l4w',
        'away_success_rate',
        'away_success_l4w',
        'away_explosive_rate',
        'away_pass_epa',
        'away_run_epa',
    ]

    # Defensive features (home team)
    home_defensive_features = [
        'home_def_epa',
        'home_def_rank',
        'home_def_pass_epa',
        'home_def_run_epa',
        'home_def_epa_l4w',
    ]

    # Defensive features (away team)
    away_defensive_features = [
        'away_def_epa',
        'away_def_rank',
        'away_def_pass_epa',
        'away_def_run_epa',
        'away_def_epa_l4w',
    ]

    # Situational features (home team)
    home_situational_features = [
        'home_rz_td_rate',
        'home_third_conv',
        'home_two_min_epa',
    ]

    # Situational features (away team)
    away_situational_features = [
        'away_rz_td_rate',
        'away_third_conv',
        'away_two_min_epa',
    ]

    # Game context features
    context_features = [
        'temp',
        'wind',
        'div_game',
        'playoff',
    ]

    # Vegas lines are NOT used as training features (would create circular logic)
    # They are only used for evaluation/edge calculation in predict.py and validation.py
    vegas_features = []  # Intentionally empty

    # Combine all features
    all_features = (
        home_offensive_features +
        away_offensive_features +
        home_defensive_features +
        away_defensive_features +
        home_situational_features +
        away_situational_features +
        context_features +
        vegas_features
    )

    # Filter to only features that exist in the DataFrame
    available_features = [f for f in all_features if f in df.columns]

    print(f"Selected {len(available_features)} features out of {len(all_features)} possible")

    if len(available_features) < len(all_features):
        missing = set(all_features) - set(available_features)
        print(f"Warning: Missing {len(missing)} features: {missing}")

    return available_features


def create_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create interaction and derived features from base features.

    Args:
        df: Polars DataFrame with base game features

    Returns:
        DataFrame with additional derived features
    """

    # Create matchup advantage features (only if base columns exist)
    derived_exprs = []

    # Offensive matchup advantages
    if 'home_epa_l4w' in df.columns and 'away_def_epa' in df.columns:
        derived_exprs.append(
            (pl.col('home_epa_l4w') - pl.col('away_def_epa')).alias('home_off_vs_away_def')
        )

    if 'away_epa_l4w' in df.columns and 'home_def_epa' in df.columns:
        derived_exprs.append(
            (pl.col('away_epa_l4w') - pl.col('home_def_epa')).alias('away_off_vs_home_def')
        )

    # Overall EPA differential
    if 'home_epa_l4w' in df.columns and 'away_epa_l4w' in df.columns:
        derived_exprs.append(
            (pl.col('home_epa_l4w') - pl.col('away_epa_l4w')).alias('epa_differential')
        )

    # Defensive EPA differential (lower is better)
    if 'home_def_epa' in df.columns and 'away_def_epa' in df.columns:
        derived_exprs.append(
            (pl.col('away_def_epa') - pl.col('home_def_epa')).alias('def_epa_differential')
        )

    # Success rate differential
    if 'home_success_l4w' in df.columns and 'away_success_l4w' in df.columns:
        derived_exprs.append(
            (pl.col('home_success_l4w') - pl.col('away_success_l4w')).alias('success_rate_diff')
        )

    # Explosive play differential
    if 'home_explosive_rate' in df.columns and 'away_explosive_rate' in df.columns:
        derived_exprs.append(
            (pl.col('home_explosive_rate') - pl.col('away_explosive_rate')).alias('explosive_rate_diff')
        )

    # Weather impact (high wind favors running teams)
    if 'wind' in df.columns and 'home_run_epa' in df.columns and 'away_pass_epa' in df.columns:
        derived_exprs.append(
            (pl.col('wind') * (pl.col('home_run_epa') - pl.col('away_pass_epa'))).alias('wind_run_advantage')
        )

    # Division rivalry intensity
    if 'div_game' in df.columns and 'home_epa_l4w' in df.columns:
        derived_exprs.append(
            (pl.col('div_game') * pl.col('home_epa_l4w')).alias('div_game_quality')
        )

    # Implied spread from EPA differential (for comparison purposes)
    # Rule of thumb: 0.1 EPA per play ≈ 2.5 points
    # NOTE: This does NOT use Vegas lines - it's purely EPA-based
    if 'home_epa_l4w' in df.columns and 'away_epa_l4w' in df.columns:
        derived_exprs.append(
            ((pl.col('home_epa_l4w') - pl.col('away_epa_l4w')) * 25.0).alias('implied_epa_spread')
        )

    # NOTE: Vegas-dependent derived features (epa_vegas_spread_diff, implied_epa_total)
    # have been intentionally removed to avoid circular logic in training

    # If we have derived expressions, add them to the DataFrame
    if derived_exprs:
        df = df.with_columns(derived_exprs)
        print(f"Created {len(derived_exprs)} derived features")

    return df


def create_target_variable(df: pl.DataFrame, target_type: str = 'spread') -> pl.DataFrame:
    """
    Create target variable for prediction.

    Args:
        df: DataFrame with home_score and away_score
        target_type: Type of target ('spread' or 'total')

    Returns:
        DataFrame with target variable added
    """

    if target_type == 'spread':
        # Spread = home_score - away_score (positive means home team won by that margin)
        df = df.with_columns(
            (pl.col('home_score') - pl.col('away_score')).alias('actual_spread')
        )
    elif target_type == 'total':
        # Total = home_score + away_score
        df = df.with_columns(
            (pl.col('home_score') + pl.col('away_score')).alias('actual_total')
        )
    else:
        raise ValueError(f"Unknown target_type: {target_type}. Use 'spread' or 'total'.")

    return df


def prepare_training_data(
    df: pl.DataFrame,
    target_type: str = 'spread',
    include_derived_features: bool = True
) -> tuple[pl.DataFrame, List[str], str]:
    """
    Full pipeline to prepare data for model training.

    Args:
        df: Raw game features DataFrame
        target_type: Type of prediction ('spread' or 'total')
        include_derived_features: Whether to create interaction features

    Returns:
        Tuple of (prepared_df, feature_columns, target_column)
    """

    # Create target variable
    df = create_target_variable(df, target_type)
    target_col = f'actual_{target_type}'

    # Create derived features if requested
    if include_derived_features:
        df = create_derived_features(df)

    # Select features
    base_features = select_spread_features(df)

    # Add derived features if they were created
    derived_features = []
    if include_derived_features:
        potential_derived = [
            'home_off_vs_away_def',
            'away_off_vs_home_def',
            'epa_differential',
            'def_epa_differential',
            'success_rate_diff',
            'explosive_rate_diff',
            'wind_run_advantage',
            'div_game_quality',
            'implied_epa_spread',
            # NOTE: epa_vegas_spread_diff and implied_epa_total removed (used Vegas lines)
        ]
        derived_features = [f for f in potential_derived if f in df.columns]

    all_features = base_features + derived_features

    # Remove rows with NULL in target or key features
    required_cols = all_features + [target_col]
    df = df.filter(
        pl.all_horizontal([pl.col(c).is_not_null() for c in required_cols])
    )

    print(f"\nData preparation complete:")
    print(f"  - Total features: {len(all_features)}")
    print(f"  - Base features: {len(base_features)}")
    print(f"  - Derived features: {len(derived_features)}")
    print(f"  - Target: {target_col}")
    print(f"  - Rows after cleaning: {len(df)}")

    return df, all_features, target_col


def get_feature_importance_names() -> Dict[str, str]:
    """
    Returns human-readable descriptions for feature names.
    Useful for interpreting model feature importance.

    Returns:
        Dictionary mapping feature names to descriptions
    """

    return {
        # Home offensive
        'home_epa_adj': 'Home Team EPA (Weekly)',
        'home_epa_l4w': 'Home Team EPA (4-Week)',
        'home_success_rate': 'Home Success Rate',
        'home_success_l4w': 'Home Success Rate (4-Week)',
        'home_explosive_rate': 'Home Explosive Play Rate',
        'home_pass_epa': 'Home Passing EPA',
        'home_run_epa': 'Home Rushing EPA',

        # Away offensive
        'away_epa_adj': 'Away Team EPA (Weekly)',
        'away_epa_l4w': 'Away Team EPA (4-Week)',
        'away_success_rate': 'Away Success Rate',
        'away_success_l4w': 'Away Success Rate (4-Week)',
        'away_explosive_rate': 'Away Explosive Play Rate',
        'away_pass_epa': 'Away Passing EPA',
        'away_run_epa': 'Away Rushing EPA',

        # Home defensive
        'home_def_epa': 'Home Defense EPA Allowed',
        'home_def_rank': 'Home Defense Rank',
        'home_def_pass_epa': 'Home Pass Defense EPA',
        'home_def_run_epa': 'Home Run Defense EPA',
        'home_def_epa_l4w': 'Home Defense EPA (4-Week)',

        # Away defensive
        'away_def_epa': 'Away Defense EPA Allowed',
        'away_def_rank': 'Away Defense Rank',
        'away_def_pass_epa': 'Away Pass Defense EPA',
        'away_def_run_epa': 'Away Run Defense EPA',
        'away_def_epa_l4w': 'Away Defense EPA (4-Week)',

        # Situational
        'home_rz_td_rate': 'Home Red Zone TD Rate',
        'home_third_conv': 'Home 3rd Down Conversion',
        'home_two_min_epa': 'Home Two-Minute Drill EPA',
        'away_rz_td_rate': 'Away Red Zone TD Rate',
        'away_third_conv': 'Away 3rd Down Conversion',
        'away_two_min_epa': 'Away Two-Minute Drill EPA',

        # Context
        'temp': 'Temperature',
        'wind': 'Wind Speed',
        'div_game': 'Division Game',
        'playoff': 'Playoff Game',

        # NOTE: Vegas lines (vegas_spread, vegas_total, vegas_home_win_prob)
        # are NOT used as training features to avoid circular logic

        # Derived
        'home_off_vs_away_def': 'Home Offense vs Away Defense Matchup',
        'away_off_vs_home_def': 'Away Offense vs Home Defense Matchup',
        'epa_differential': 'EPA Differential (Home - Away)',
        'def_epa_differential': 'Defensive EPA Differential',
        'success_rate_diff': 'Success Rate Differential',
        'explosive_rate_diff': 'Explosive Play Rate Differential',
        'wind_run_advantage': 'Wind-Adjusted Run Advantage',
        'div_game_quality': 'Division Game Quality Factor',
        'implied_epa_spread': 'EPA-Implied Spread Estimate',
    }
