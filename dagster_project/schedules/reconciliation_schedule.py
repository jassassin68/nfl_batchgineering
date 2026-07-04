"""Tuesday schedule for settling the weekend's bets.

Runs a couple of hours before ``weekly_prediction_schedule`` (which fires at
08:00) so the prior week's results are settled into ML.BET_RESULTS before the
next week's recommendations are generated. Skips the offseason.
"""

from datetime import datetime

from dagster import (
    DefaultScheduleStatus,
    RunConfig,
    RunRequest,
    ScheduleEvaluationContext,
    SkipReason,
    schedule,
)

from dagster_project.assets.reconciliation import ReconciliationConfig
from dagster_project.jobs.reconciliation_pipeline import reconciliation_job

# NFL regular season months (September through January, plus early February).
# Mirrors weekly_schedule.NFL_SEASON_MONTHS.
NFL_SEASON_MONTHS = {9, 10, 11, 12, 1, 2}


@schedule(
    job=reconciliation_job,
    cron_schedule="0 6 * * 2",  # Every Tuesday at 6:00 AM
    default_status=DefaultScheduleStatus.STOPPED,
)
def reconciliation_schedule(context: ScheduleEvaluationContext):
    """Trigger bet reconciliation on Tuesdays during the NFL season."""
    scheduled_time = context.scheduled_execution_time
    if scheduled_time is None:
        scheduled_time = datetime.now()

    if scheduled_time.month not in NFL_SEASON_MONTHS:
        return SkipReason(
            f"Off-season month ({scheduled_time.strftime('%B')}). "
            f"Bet reconciliation only runs September through February."
        )

    return RunRequest(
        run_config=RunConfig(
            ops={"bet_result_reconciliation": ReconciliationConfig()}
        ),
        tags={"job": "reconciliation"},
    )
