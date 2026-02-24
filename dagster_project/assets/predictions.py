"""Prediction asset: generate weekly NFL spread predictions."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from dagster import asset, AssetKey, Config, MaterializeResult, MetadataValue

from dagster_project.constants import PROJECT_ROOT, MODEL_DIR, DATA_DIR

sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.predict import (
    load_upcoming_games,
    load_ensemble_model,
    generate_predictions,
    write_to_snowflake,
)


class PredictionConfig(Config):
    """Run config for the weekly predictions asset."""

    week: int
    season: int


@asset(
    group_name="predictions",
    compute_kind="python",
    deps=[AssetKey("mart_upcoming_game_predictions")],
)
def weekly_predictions(context, config: PredictionConfig) -> MaterializeResult:
    """Generate spread predictions for upcoming games using the trained XGBoost model.

    Depends on the dbt mart_upcoming_game_predictions asset being materialized
    first. Loads the pre-trained model from the model directory, generates
    predictions, writes results to Snowflake and CSV.
    """
    week = config.week
    season = config.season
    context.log.info(f"Generating predictions for Week {week}, Season {season}")

    # 1. Load upcoming games from Snowflake mart
    games_df = load_upcoming_games(week, season)
    if games_df.is_empty():
        context.log.warning(f"No upcoming games for Week {week}, Season {season}")
        return MaterializeResult(
            metadata={
                "games_predicted": MetadataValue.int(0),
                "bets_recommended": MetadataValue.int(0),
            }
        )

    context.log.info(f"Loaded {len(games_df)} upcoming games")

    # 2. Load trained models (XGBoost-only when only xgboost/ subdir exists)
    model_dir = str(MODEL_DIR)
    context.log.info(f"Loading models from {model_dir}")
    models = load_ensemble_model(model_dir)

    if not models:
        raise RuntimeError(f"No models found in {model_dir}")

    context.log.info(f"Loaded models: {list(models.keys())}")

    # 3. Generate predictions (reuses existing logic from predict.py)
    results = generate_predictions(games_df, models)

    if results.is_empty():
        raise RuntimeError("Prediction generation returned empty results")

    # 4. Write to Snowflake
    write_to_snowflake(results, week, season)
    context.log.info(f"Wrote {len(results)} predictions to Snowflake")

    # 5. Write CSV output
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / f"predictions_week{week}.csv"
    results.write_csv(str(csv_path))
    context.log.info(f"Wrote predictions to {csv_path}")

    # 6. Compute summary metadata
    bets_recommended = 0
    if "bet_recommendation" in results.columns:
        bets_recommended = len(
            results.filter(pl.col("bet_recommendation") != "NO BET")
        )

    return MaterializeResult(
        metadata={
            "games_predicted": MetadataValue.int(len(results)),
            "bets_recommended": MetadataValue.int(bets_recommended),
            "csv_path": MetadataValue.path(str(csv_path)),
            "week": MetadataValue.int(week),
            "season": MetadataValue.int(season),
        }
    )
