"""Reconciliation job: settle completed bets into ML.BET_RESULTS."""

from dagster import AssetSelection, define_asset_job

# Just the reconciliation asset. It reads final scores from Snowflake
# (mart_game_prediction_features, refreshed by the weekly pipeline) rather
# than re-materializing the marts, so the selection is intentionally narrow.
reconciliation_job = define_asset_job(
    name="reconciliation_job",
    selection=AssetSelection.groups("reconciliation"),
    description=(
        "Settle recorded bets against final game results: compute outcome, "
        "profit, and CLV and write PRODUCTION_ANALYTICS.ML.BET_RESULTS."
    ),
)
