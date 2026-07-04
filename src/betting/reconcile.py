"""Settle recorded bets against final game results.

Scans ``PRODUCTION_ANALYTICS.ML.BETS`` for rows that are old enough to have
completed and do not yet have a matching ``BET_RESULTS`` row, joins to final
scores and the kickoff (closing) line on ``mart_game_prediction_features``,
settles each via :func:`src.betting.clv.settle_bet`, and writes ``BET_RESULTS``.

This is the library entry point used by the ``bet_result_reconciliation``
Dagster asset -- the settlement math lives in src/betting/clv.py, the SQL
orchestration lives here, and the asset is a thin wrapper.

Closing line: the nflverse spread at a game's first play is the kickoff line,
exposed as ``vegas_spread`` on ``mart_game_prediction_features``; no separate
odds feed is required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from src.betting.clv import settle_bet

BETS_TABLE = "PRODUCTION_ANALYTICS.ML.BETS"
RESULTS_TABLE = "PRODUCTION_ANALYTICS.ML.BET_RESULTS"

_CREATE_RESULTS_SQL = f"""
    CREATE TABLE IF NOT EXISTS {RESULTS_TABLE} (
        bet_id              VARCHAR        NOT NULL,
        game_id             VARCHAR        NOT NULL,
        home_score          NUMBER,
        away_score          NUMBER,
        margin              FLOAT,
        vegas_spread_close  FLOAT,
        outcome             VARCHAR,
        profit_units        FLOAT,
        clv_points          FLOAT,
        resolved_at         TIMESTAMP_NTZ,
        PRIMARY KEY (bet_id)
    )
"""

_INSERT_RESULTS_SQL = f"""
    INSERT INTO {RESULTS_TABLE}
        (bet_id, game_id, home_score, away_score, margin,
         vegas_spread_close, outcome, profit_units, clv_points, resolved_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def _unresolved_query(marts_table_name: str, min_age_hours: int) -> str:
    """SQL selecting completed, not-yet-resolved bets joined to final scores."""
    return f"""
        SELECT
            b.bet_id,
            b.game_id,
            b.side,
            b.vegas_spread_at_rec,
            b.stake_units,
            b.odds_at_rec,
            m.home_score,
            m.away_score,
            m.vegas_spread AS vegas_spread_close
        FROM {BETS_TABLE} b
        JOIN {marts_table_name} m
            ON b.game_id = m.game_id
        LEFT JOIN {RESULTS_TABLE} r
            ON b.bet_id = r.bet_id
        WHERE r.bet_id IS NULL
          AND m.home_score IS NOT NULL
          AND m.away_score IS NOT NULL
          AND b.recommended_at < DATEADD('hour', -{int(min_age_hours)}, CURRENT_TIMESTAMP())
    """


def _settle_row(row: dict, resolved_at: datetime) -> tuple:
    """Settle one candidate row into a BET_RESULTS insert tuple."""
    home = float(row["home_score"])
    away = float(row["away_score"])
    margin = home - away
    at_rec = float(row["vegas_spread_at_rec"])
    close = (
        None
        if row["vegas_spread_close"] is None
        else float(row["vegas_spread_close"])
    )
    odds = float(row["odds_at_rec"]) if row["odds_at_rec"] is not None else -110.0

    settled = settle_bet(
        margin=margin,
        vegas_spread_at_rec=at_rec,
        vegas_spread_close=close,
        side=row["side"],
        stake_units=float(row["stake_units"]),
        odds=odds,
    )
    return (
        str(row["bet_id"]),
        str(row["game_id"]),
        int(home),
        int(away),
        margin,
        close,
        settled["outcome"],
        settled["profit_units"],
        settled["clv_points"],
        resolved_at.replace(tzinfo=None),  # TIMESTAMP_NTZ expects naive UTC
    )


def reconcile_bets(
    conn=None,
    *,
    min_age_hours: int = 24,
    resolved_at: Optional[datetime] = None,
) -> dict:
    """Settle all eligible bets and write ``BET_RESULTS``.

    Args:
        conn: Optional open Snowflake connection (injected by tests). When
            None, opens one via ``get_snowflake_connection`` and closes it.
        min_age_hours: Only settle bets recommended at least this many hours
            ago, so in-progress games are never settled early.
        resolved_at: Timestamp to stamp results with (defaults to now, UTC).

    Returns:
        Summary dict: ``{"candidates": int, "resolved": int}``.
    """
    if resolved_at is None:
        resolved_at = datetime.now(timezone.utc)

    owns_conn = conn is None
    if owns_conn:
        from src.ml.predict import get_snowflake_connection

        conn = get_snowflake_connection()

    try:
        from src.pipeline.snowflake import marts_table

        marts_name = marts_table("mart_game_prediction_features")

        cursor = conn.cursor()
        cursor.execute(_CREATE_RESULTS_SQL)

        cursor.execute(_unresolved_query(marts_name, min_age_hours))
        columns = [desc[0].lower() for desc in cursor.description]
        candidates = [dict(zip(columns, r)) for r in cursor.fetchall()]

        rows: List[tuple] = [_settle_row(c, resolved_at) for c in candidates]
        if rows:
            cursor.executemany(_INSERT_RESULTS_SQL, rows)
        conn.commit()
        cursor.close()
    finally:
        if owns_conn:
            conn.close()

    return {"candidates": len(candidates), "resolved": len(rows)}
