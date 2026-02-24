"""Ingestion assets: load nflverse data to Snowflake RAW.NFLVERSE.

The per-year datasets (play_by_play, rosters, etc.) are loaded as a single
multi_asset that produces one output per dbt source table. This ensures
Dagster's asset graph correctly wires each dbt staging model to the specific
source table it depends on.

The schedules dataset is a separate asset because it uses a different loader
(single-file, full replace) and maps to a distinct dbt source.
"""

import sys
from datetime import datetime

from dagster import (
    AssetKey,
    AssetSpec,
    MaterializeResult,
    asset,
    multi_asset,
)

from dagster_project.constants import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.training_data_loader import TrainingDataLoader

# The 6 per-year datasets, each mapping to a dbt source table under nfl_verse_raw.
# The asset keys use the default dagster-dbt convention: ["nfl_verse_raw", "<table>"]
PER_YEAR_DATASETS = list(TrainingDataLoader.DATASET_URLS.keys())


def _current_season_year() -> int:
    """Derive the current NFL season year.

    NFL seasons span two calendar years (e.g., the 2025 season runs Sep 2025
    through Feb 2026). If we're in Jan/Feb, the active season started the
    previous calendar year.
    """
    now = datetime.now()
    if now.month <= 2:
        return now.year - 1
    return now.year


@multi_asset(
    specs=[
        AssetSpec(
            key=AssetKey(["nfl_verse_raw", dataset]),
            group_name="ingestion",
        )
        for dataset in PER_YEAR_DATASETS
    ],
    compute_kind="python",
    can_subset=False,
)
def raw_nflverse_data(context):
    """Load per-year nflverse datasets for the current season to Snowflake.

    Loads play_by_play, rosters, player_summary_stats, team_summary_stats,
    injuries, and play_by_play_participation for the current NFL season year.
    Historical years don't change, so only the current year is refreshed weekly.

    Produces one asset output per dataset, matching the dbt source table keys.
    """
    season_year = _current_season_year()
    years = [season_year]

    context.log.info(
        f"Loading {len(PER_YEAR_DATASETS)} datasets for season {season_year}"
    )

    with TrainingDataLoader() as loader:
        for dataset in PER_YEAR_DATASETS:
            context.log.info(f"Loading {dataset} for {years}...")
            result = loader.bulk_load_dataset(dataset, years)

            if not result.success:
                raise RuntimeError(
                    f"Failed to load {dataset}: {result.error_message}"
                )

            context.log.info(
                f"Loaded {dataset}: {result.rows_loaded:,} rows "
                f"in {result.duration_seconds:.1f}s"
            )

            yield MaterializeResult(
                asset_key=AssetKey(["nfl_verse_raw", dataset]),
                metadata={
                    "rows_loaded": result.rows_loaded,
                    "season_year": season_year,
                    "duration_s": round(result.duration_seconds, 1),
                },
            )


@asset(
    key=AssetKey(["nfl_verse_raw", "schedules"]),
    group_name="ingestion",
    compute_kind="python",
)
def raw_schedules() -> MaterializeResult:
    """Load the schedules single-file dataset to Snowflake.

    Schedules contain all seasons in one file and are fully replaced on each
    load. This ensures future game schedules and Vegas lines are up to date.
    """
    with TrainingDataLoader() as loader:
        result = loader.load_single_file_dataset("schedules")

    if not result.success:
        raise RuntimeError(f"Failed to load schedules: {result.error_message}")

    return MaterializeResult(
        metadata={
            "rows_loaded": result.rows_loaded,
            "duration_s": round(result.duration_seconds, 1),
        },
    )
