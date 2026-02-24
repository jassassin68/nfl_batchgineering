"""Weekly prediction job: materializes the full pipeline from ingestion to predictions."""

from dagster import define_asset_job, AssetSelection

# Select the full chain: ingestion -> dbt -> predictions
# Excludes the training asset (manual-only)
weekly_prediction_job = define_asset_job(
    name="weekly_prediction_job",
    selection=(
        AssetSelection.groups("ingestion")
        | AssetSelection.groups("staging")
        | AssetSelection.groups("intermediate")
        | AssetSelection.groups("marts")
        | AssetSelection.groups("predictions")
    ),
    description=(
        "Weekly NFL prediction pipeline: load latest nflverse data, "
        "run dbt transforms, and generate spread predictions."
    ),
)
