"""Preflight asset: deterministic validation gate for the test pipeline.

Runs first in the test_prediction_job. Validates that the requested historical
week/season exists with usable feature data, then emits the execution plan.
Raises (failing the run loudly) when validation cannot pass.
"""

import sys

from dagster import Config, MetadataValue, asset

from dagster_project.constants import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.config import PipelineMode, PipelineRunConfig
from src.pipeline.preflight import PreflightError, run_preflight


class TestModeConfig(Config):
    """Run config for a test-mode pipeline run."""

    week: int
    season: int


@asset(group_name="preflight", compute_kind="python")
def pipeline_preflight(context, config: TestModeConfig) -> dict:
    """Validate the requested test week and construct the execution plan.

    Deterministically answers: does this week/season exist in the historical
    feature mart, and are the required model feature columns populated. On
    failure it raises PreflightError so the run stops loudly. On success it
    returns the plan for downstream assets to consume.
    """
    run_config = PipelineRunConfig(
        mode=PipelineMode.TEST, week=config.week, season=config.season
    )

    try:
        plan = run_preflight(run_config)
    except PreflightError as exc:
        context.log.error(f"Preflight validation failed:\n{exc}")
        raise

    context.log.info("\n" + plan.render())

    checks_text = "\n".join(
        f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}"
        for c in plan.checks
    )
    context.add_output_metadata(
        {
            "mode": "TEST",
            "week": config.week,
            "season": config.season,
            "checks": MetadataValue.text(checks_text),
            "steps_to_run": MetadataValue.text(", ".join(plan.steps_to_run())),
            "steps_skipped": MetadataValue.text(", ".join(plan.steps_skipped())),
        }
    )

    return {
        "mode": "test",
        "week": config.week,
        "season": config.season,
        "steps_to_run": plan.steps_to_run(),
    }
