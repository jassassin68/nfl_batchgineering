"""Tuesday schedule for weekly NFL predictions during the regular season."""

from datetime import datetime, timedelta

from dagster import (
    DefaultScheduleStatus,
    RunConfig,
    RunRequest,
    ScheduleEvaluationContext,
    SkipReason,
    schedule,
)

from dagster_project.assets.predictions import PredictionConfig
from dagster_project.jobs.weekly_pipeline import weekly_prediction_job

# NFL regular season months (September through January, plus early February)
NFL_SEASON_MONTHS = {9, 10, 11, 12, 1, 2}


def _derive_nfl_week(execution_date: datetime) -> tuple[int, int]:
    """Derive the NFL week number and season from a Tuesday execution date.

    The NFL season starts on the Thursday after Labor Day (first Monday in Sep).
    Tuesday predictions target the upcoming week's games (Thursday-Monday).
    Week 1 starts the first full week of September.

    Returns:
        (week, season) tuple
    """
    year = execution_date.year
    month = execution_date.month

    # Determine season year: Jan/Feb belong to the previous year's season
    if month <= 2:
        season = year - 1
    else:
        season = year

    # Approximate Week 1 start: first Tuesday of September in the season year
    # NFL Week 1 is typically the Thursday after Labor Day
    sep_1 = datetime(season, 9, 1)
    # Find the first Tuesday in September
    days_until_tuesday = (1 - sep_1.weekday()) % 7  # Tuesday = 1
    first_tuesday = sep_1 + timedelta(days=days_until_tuesday)

    # Calculate weeks elapsed since first Tuesday of September
    days_elapsed = (execution_date - first_tuesday).days
    week = max(1, (days_elapsed // 7) + 1)

    # Cap at 18 for regular season
    week = min(week, 18)

    return week, season


@schedule(
    job=weekly_prediction_job,
    cron_schedule="0 8 * * 2",  # Every Tuesday at 8:00 AM
    default_status=DefaultScheduleStatus.STOPPED,
)
def weekly_prediction_schedule(context: ScheduleEvaluationContext):
    """Trigger the weekly prediction pipeline on Tuesdays during NFL season.

    Skips execution during the offseason (March through August).
    Passes the derived NFL week and season as run config to the predictions asset.
    """
    scheduled_time = context.scheduled_execution_time
    if scheduled_time is None:
        scheduled_time = datetime.now()

    current_month = scheduled_time.month

    if current_month not in NFL_SEASON_MONTHS:
        return SkipReason(
            f"Off-season month ({scheduled_time.strftime('%B')}). "
            f"NFL predictions only run September through February."
        )

    week, season = _derive_nfl_week(scheduled_time)

    return RunRequest(
        run_config=RunConfig(
            ops={"weekly_predictions": PredictionConfig(week=week, season=season)}
        ),
        tags={"nfl_week": str(week), "nfl_season": str(season)},
    )
