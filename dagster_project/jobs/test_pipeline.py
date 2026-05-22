"""Test-mode pipeline job: preflight validation then historical replay.

Replays a known completed week (supplied as run config) through the model so
the end-to-end pipeline can be verified deterministically -- e.g. during the
offseason when no upcoming games exist. Preflight runs first and fails the run
loudly if the requested week's data is missing or malformed.
"""

from dagster import AssetSelection, define_asset_job

test_prediction_job = define_asset_job(
    name="test_prediction_job",
    selection=AssetSelection.assets("pipeline_preflight", "test_predictions"),
    description=(
        "Test-mode pipeline: validate a historical week via preflight, then "
        "replay it through the ensemble and score predictions vs actual "
        "results. Requires test_week and test_season in run config."
    ),
)
