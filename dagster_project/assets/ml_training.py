"""ML training asset: train XGBoost spread predictor (manual trigger only)."""

import sys
from argparse import Namespace

from dagster import asset, AssetKey, Config, MaterializeResult, MetadataValue

from dagster_project.constants import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.train_spread_model import main as train_main


class TrainingConfig(Config):
    """Run config for the model training asset."""

    min_season: int = 2015
    max_depth: int = 5
    learning_rate: float = 0.05
    n_estimators: int = 300
    early_stopping: int = 50
    include_derived: bool = True
    output_dir: str = "ml_models"
    limit: int = 0


@asset(
    group_name="ml_training",
    compute_kind="python",
    deps=[AssetKey("mart_game_prediction_features")],
    automation_condition=None,
)
def trained_xgboost_model(context, config: TrainingConfig) -> MaterializeResult:
    """Train the XGBoost spread predictor model.

    NOT part of the weekly pipeline -- triggered manually or at start of season.
    Depends on the dbt mart_game_prediction_features table containing historical
    game data with features.

    Wraps src/ml/train_spread_model.py's main() function.
    """
    context.log.info(
        f"Starting XGBoost training: seasons >= {config.min_season}, "
        f"depth={config.max_depth}, lr={config.learning_rate}, "
        f"n_estimators={config.n_estimators}"
    )

    args = Namespace(
        min_season=config.min_season,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        early_stopping=config.early_stopping,
        include_derived=config.include_derived,
        output_dir=config.output_dir,
        limit=config.limit if config.limit > 0 else None,
    )

    train_main(args)

    context.log.info(f"Training complete. Artifacts saved to {config.output_dir}/")

    return MaterializeResult(
        metadata={
            "output_dir": MetadataValue.path(config.output_dir),
            "min_season": MetadataValue.int(config.min_season),
            "max_depth": MetadataValue.int(config.max_depth),
            "learning_rate": MetadataValue.float(config.learning_rate),
            "n_estimators": MetadataValue.int(config.n_estimators),
        }
    )
