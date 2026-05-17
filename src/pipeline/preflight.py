"""Deterministic preflight validation for the NFL prediction pipeline.

Runs as the first step of every pipeline run. It answers a fixed set of
questions about data availability and, from the answers, constructs the
execution plan (which steps run vs are skipped). If any question cannot be
answered affirmatively it raises PreflightError -- failing loudly rather than
letting the pipeline proceed on missing or malformed data.

TEST mode replays a historical week from mart_game_prediction_features, so the
heavy ingestion/dbt steps are skipped (the data is already loaded and
transformed). Ingestion is never auto-run -- missing raw data is a hard stop.
"""

from __future__ import annotations

from src.pipeline.config import (
    CheckResult,
    ExecutionPlan,
    PipelineMode,
    PipelineRunConfig,
    StepDecision,
)
from src.pipeline.snowflake import marts_table, query_df

# Core model feature columns that must exist and be non-null for the requested
# week. A representative subset of the 34 spread-model features -- if these are
# populated the dbt feature build completed correctly.
REQUIRED_FEATURE_COLUMNS = [
    "home_epa_adj",
    "home_epa_l4w",
    "home_def_epa",
    "away_epa_adj",
    "away_epa_l4w",
    "away_def_epa",
]


class PreflightError(RuntimeError):
    """Raised when preflight validation fails -- the pipeline must not proceed."""


def run_preflight(config: PipelineRunConfig) -> ExecutionPlan:
    """Validate inputs/data and construct the execution plan.

    Args:
        config: the pipeline run configuration (mode, week, season).

    Returns:
        ExecutionPlan with validation checks and per-step run/skip decisions.

    Raises:
        PreflightError: if a required question cannot be answered affirmatively.
    """
    if config.mode is PipelineMode.TEST:
        return _preflight_test(config)
    return _preflight_prod(config)


def _preflight_test(config: PipelineRunConfig) -> ExecutionPlan:
    """Validate a TEST-mode run against a historical week."""
    plan = ExecutionPlan(config=config)
    table = marts_table("mart_game_prediction_features")
    season, week = config.season, config.week

    # Q1: does this week/season exist in the historical feature mart?
    count_df = query_df(
        f"select count(*) as n from {table} "
        f"where season = {season} and week = {week}"
    )
    n_games = int(count_df["n"][0]) if len(count_df) else 0
    week_exists = n_games > 0
    plan.checks.append(
        CheckResult(
            "week_exists_in_history",
            week_exists,
            f"{n_games} game(s) in {table} for season {season}, week {week}"
            if week_exists
            else f"no games in {table} for season {season}, week {week}",
        )
    )
    if not week_exists:
        available = query_df(
            f"select distinct season, week from {table} order by season, week"
        )
        raise PreflightError(
            plan.render()
            + f"\n\nFAIL: season {season} week {week} is not in the historical "
            f"data.\nAvailable (season, week): "
            f"{available.to_dicts() if len(available) else 'none'}"
        )

    # Q2: are the required model feature columns present and non-null?
    null_exprs = ", ".join(
        f"sum(case when {col} is null then 1 else 0 end) as {col}_nulls"
        for col in REQUIRED_FEATURE_COLUMNS
    )
    nulls_df = query_df(
        f"select {null_exprs} from {table} "
        f"where season = {season} and week = {week}"
    )
    null_cols = [
        col
        for col in REQUIRED_FEATURE_COLUMNS
        if int(nulls_df[f"{col}_nulls"][0] or 0) > 0
    ]
    features_ok = not null_cols
    plan.checks.append(
        CheckResult(
            "required_features_present",
            features_ok,
            "all required model feature columns are populated"
            if features_ok
            else f"null values found in feature columns: {null_cols}",
        )
    )
    if not features_ok:
        raise PreflightError(
            plan.render()
            + f"\n\nFAIL: required feature columns contain nulls: {null_cols}. "
            "Re-run dbt to rebuild the feature mart."
        )

    # All questions answered affirmatively -- construct the step plan.
    plan.steps = [
        StepDecision(
            "ingestion",
            False,
            "TEST mode: historical raw data already loaded for this season",
        ),
        StepDecision(
            "dbt_build",
            False,
            "TEST mode: feature mart already built (validated above)",
        ),
        StepDecision(
            "predict",
            True,
            f"generate predictions for {n_games} historical game(s)",
        ),
    ]
    return plan


def _preflight_prod(config: PipelineRunConfig) -> ExecutionPlan:
    """Construct the plan for a PROD run (full live pipeline)."""
    plan = ExecutionPlan(config=config)
    plan.checks.append(
        CheckResult(
            "prod_mode",
            True,
            "production run: full ingestion + dbt + predict pipeline",
        )
    )
    plan.steps = [
        StepDecision("ingestion", True, "load latest nflverse data"),
        StepDecision("dbt_build", True, "refresh staging/intermediate/marts"),
        StepDecision("predict", True, "generate predictions for upcoming games"),
    ]
    return plan
