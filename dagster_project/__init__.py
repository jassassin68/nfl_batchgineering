"""Dagster definitions entry point for the NFL prediction system."""

from dotenv import load_dotenv
from dagster import Definitions

from dagster_project.assets.ingestion import raw_nflverse_data, raw_schedules
from dagster_project.assets.dbt_assets import nfl_dbt_assets
from dagster_project.assets.predictions import weekly_predictions, test_predictions
from dagster_project.assets.preflight import pipeline_preflight
from dagster_project.assets.ml_training import trained_xgboost_model
from dagster_project.assets.validation import model_validation_report
from dagster_project.assets.recommendations import weekly_bet_recommendations
from dagster_project.resources.dbt_resource import dbt_resource
from dagster_project.jobs.weekly_pipeline import weekly_prediction_job
from dagster_project.jobs.test_pipeline import test_prediction_job
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
        pipeline_preflight,
        test_predictions,
        model_validation_report,
        weekly_bet_recommendations,
    ],
    resources={
        "dbt": dbt_resource,
    },
    jobs=[weekly_prediction_job, test_prediction_job],
    schedules=[weekly_prediction_schedule],
)
