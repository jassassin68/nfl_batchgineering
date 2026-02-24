"""Dagster definitions entry point for the NFL prediction system."""

from dotenv import load_dotenv
from dagster import Definitions

from dagster_project.assets.ingestion import raw_nflverse_data, raw_schedules
from dagster_project.assets.dbt_assets import nfl_dbt_assets
from dagster_project.assets.predictions import weekly_predictions
from dagster_project.assets.ml_training import trained_xgboost_model
from dagster_project.resources.dbt_resource import dbt_resource
from dagster_project.jobs.weekly_pipeline import weekly_prediction_job
from dagster_project.schedules.weekly_schedule import weekly_prediction_schedule

# Load .env so Snowflake credentials and other env vars are available
load_dotenv()

defs = Definitions(
    assets=[
        raw_nflverse_data,
        raw_schedules,
        nfl_dbt_assets,
        weekly_predictions,
        trained_xgboost_model,
    ],
    resources={
        "dbt": dbt_resource,
    },
    jobs=[weekly_prediction_job],
    schedules=[weekly_prediction_schedule],
)
