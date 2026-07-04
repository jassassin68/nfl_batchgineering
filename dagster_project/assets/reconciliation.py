"""Bet result reconciliation asset.

Settles recorded bets (PRODUCTION_ANALYTICS.ML.BETS) against final game
results once games complete, writing outcome / profit / CLV to
PRODUCTION_ANALYTICS.ML.BET_RESULTS. Runs Tuesday mornings after the weekend
slate resolves.

Thin wrapper: all settlement math lives in src/betting/clv.py and the SQL
orchestration in src/betting/reconcile.py -- one definition, imported here.

Note: NO ``from __future__ import annotations`` here -- Dagster's pythonic
Config introspection cannot resolve PEP 563 stringified annotations and
raises DagsterInvalidPythonicConfigDefinitionError.
"""

import sys

from dagster import AssetKey, Config, MaterializeResult, MetadataValue, asset

from dagster_project.constants import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from src.betting.reconcile import reconcile_bets


class ReconciliationConfig(Config):
    """Run config for the bet_result_reconciliation asset."""

    # Only settle bets recommended at least this many hours ago, so a game
    # still in progress is never settled early.
    min_age_hours: int = 24


@asset(
    group_name="reconciliation",
    compute_kind="python",
    deps=[AssetKey("mart_game_prediction_features")],
)
def bet_result_reconciliation(
    context, config: ReconciliationConfig
) -> MaterializeResult:
    """Settle completed bets and write ML.BET_RESULTS.

    Scans ML.BETS for rows older than ``min_age_hours`` without a matching
    ML.BET_RESULTS row, joins to final scores + kickoff line on
    mart_game_prediction_features, and persists the settled outcome, profit,
    and CLV for each.
    """
    context.log.info(
        f"Reconciling bets older than {config.min_age_hours}h without results"
    )

    summary = reconcile_bets(min_age_hours=config.min_age_hours)

    context.log.info(
        f"Reconciliation: {summary['candidates']} candidates, "
        f"{summary['resolved']} newly resolved"
    )

    return MaterializeResult(
        metadata={
            "candidates": MetadataValue.int(summary["candidates"]),
            "resolved": MetadataValue.int(summary["resolved"]),
            "min_age_hours": MetadataValue.int(config.min_age_hours),
        }
    )
