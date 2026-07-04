"""Weekly bet recommendations asset.

Materializes sized bet recommendations for an upcoming NFL week and writes
a markdown + CSV pair under reports/recommendations/. Calls
src.betting.recommend.build_recommendations so the asset and CLI produce
identical artifacts.

Note: NO ``from __future__ import annotations`` here -- Dagster's pythonic
Config introspection cannot resolve PEP 563 stringified annotations and
raises DagsterInvalidPythonicConfigDefinitionError.
"""

import sys

import polars as pl
from dagster import AssetKey, Config, MaterializeResult, MetadataValue, asset

from dagster_project.constants import MODEL_DIR, PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from src.betting.ledger import DEFAULT_MODEL_VERSION, record_bets
from src.betting.recommend import (
    DEFAULT_BANKROLL,
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_KELLY_MULT,
    DEFAULT_ODDS,
    SIDE_PASS,
    build_recommendations,
    write_recommendations_report,
)
from src.ml.predict import (
    generate_predictions,
    load_ensemble_model,
    load_upcoming_games,
)


class RecommendationsConfig(Config):
    """Run config for the weekly_bet_recommendations asset."""

    week: int
    season: int
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD
    kelly_mult: float = DEFAULT_KELLY_MULT
    bankroll: float = DEFAULT_BANKROLL
    odds: float = DEFAULT_ODDS
    # Tag separating champion vs challenger results in the bet ledger.
    # Hand-maintained until Step F adds proper model versioning.
    model_version: str = DEFAULT_MODEL_VERSION
    # When False, recommendations are computed and reported but NOT written
    # to ML.BETS (useful for dry-runs / backfills).
    record_to_ledger: bool = True


@asset(
    group_name="recommendations",
    compute_kind="python",
    deps=[AssetKey("mart_upcoming_game_predictions")],
)
def weekly_bet_recommendations(
    context, config: RecommendationsConfig
) -> MaterializeResult:
    """Build sized bet recommendations for the configured week.

    Re-runs predictions rather than reading ML.PREDICTIONS to guarantee
    sizing is derived from the freshest model output.
    """
    context.log.info(
        f"Recommendations: season={config.season} week={config.week} "
        f"edge>={config.edge_threshold} kelly_mult={config.kelly_mult} "
        f"bankroll={config.bankroll} odds={config.odds}"
    )

    games_df = load_upcoming_games(config.week, config.season)
    if games_df.is_empty():
        context.log.warning(
            f"No upcoming games for season {config.season} week {config.week}"
        )
        return MaterializeResult(
            metadata={
                "bets_count": MetadataValue.int(0),
                "games_considered": MetadataValue.int(0),
            }
        )

    models = load_ensemble_model(str(MODEL_DIR))
    if not models:
        raise RuntimeError(f"No models found in {MODEL_DIR}")
    context.log.info(f"Loaded models: {list(models.keys())}")

    predictions = generate_predictions(games_df, models)
    if predictions.is_empty():
        raise RuntimeError("Prediction generation returned empty results.")
    context.log.info(f"Generated {predictions.height} predictions")

    recs = build_recommendations(
        predictions,
        edge_threshold=config.edge_threshold,
        kelly_mult=config.kelly_mult,
        bankroll=config.bankroll,
        odds=config.odds,
    )

    report_dir = PROJECT_ROOT / "reports" / "recommendations"
    run_config = {
        "season": config.season,
        "week": config.week,
        "edge_threshold": config.edge_threshold,
        "kelly_mult": config.kelly_mult,
        "bankroll": config.bankroll,
        "odds": config.odds,
    }
    md_path, csv_path = write_recommendations_report(
        recs, report_dir, config.season, config.week, run_config
    )
    context.log.info(f"Wrote {md_path}")
    context.log.info(f"Wrote {csv_path}")

    bets = recs.filter(pl.col("side") != SIDE_PASS)
    n_bets = bets.height

    # Persist every non-pass recommendation to the bet ledger so production
    # CLV/ROI can be measured (Step E). staked defaults to True.
    bets_recorded = 0
    if config.record_to_ledger:
        bets_recorded = record_bets(
            recs,
            season=config.season,
            week=config.week,
            model_version=config.model_version,
            bankroll=config.bankroll,
            odds=config.odds,
        )
        context.log.info(
            f"Recorded {bets_recorded} bets to ML.BETS "
            f"(model_version={config.model_version})"
        )
    else:
        context.log.info("record_to_ledger=False -- skipped ML.BETS write")
    total_stake = float(bets["stake_units"].sum()) if n_bets else 0.0
    max_stake = float(bets["stake_units"].max()) if n_bets else 0.0
    avg_edge = (
        float(bets["edge_points"].abs().mean()) if n_bets else 0.0
    )

    return MaterializeResult(
        metadata={
            "games_considered": MetadataValue.int(recs.height),
            "bets_count": MetadataValue.int(n_bets),
            "bets_recorded": MetadataValue.int(bets_recorded),
            "model_version": MetadataValue.text(config.model_version),
            "total_stake": MetadataValue.float(round(total_stake, 4)),
            "max_stake": MetadataValue.float(round(max_stake, 4)),
            "avg_edge_pts": MetadataValue.float(round(avg_edge, 3)),
            "bankroll": MetadataValue.float(float(config.bankroll)),
            "edge_threshold": MetadataValue.float(float(config.edge_threshold)),
            "kelly_mult": MetadataValue.float(float(config.kelly_mult)),
            "markdown_path": MetadataValue.path(str(md_path)),
            "csv_path": MetadataValue.path(str(csv_path)),
        }
    )
