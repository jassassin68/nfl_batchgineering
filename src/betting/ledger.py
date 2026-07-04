"""Persist sized bet recommendations to the production bet ledger.

Every non-pass recommendation produced by
``src.betting.recommend.build_recommendations`` is written to
``PRODUCTION_ANALYTICS.ML.BETS`` so the system's real-money edge can be
measured later via CLV and ROI (see dagster_project/assets/reconciliation.py
and dbt_project/models/3_marts/mart_bet_performance.sql).

Reuses ``get_snowflake_connection`` from src/ml/predict.py -- one source of
truth for Snowflake key-pair auth. Bets are written with ``staked = TRUE`` by
default; the user flips ``staked`` to FALSE in Snowflake only when they decide
*not* to take a recommended bet (default-yes is safer than default-no for
measuring system performance).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

import polars as pl

from src.betting.recommend import SIDE_PASS

BETS_TABLE = "PRODUCTION_ANALYTICS.ML.BETS"

# Default model tag. Proper model versioning is deferred to Step F
# (see .claude/plans/step-f-model-iteration.md); until then this is a
# hand-maintained string bumped whenever the ensemble changes.
DEFAULT_MODEL_VERSION = "step_d_xgb_v1"

_CREATE_BETS_SQL = f"""
    CREATE TABLE IF NOT EXISTS {BETS_TABLE} (
        bet_id              VARCHAR        NOT NULL,
        game_id             VARCHAR        NOT NULL,
        season              NUMBER         NOT NULL,
        week                NUMBER         NOT NULL,
        recommended_at      TIMESTAMP_NTZ  NOT NULL,
        side                VARCHAR        NOT NULL,
        predicted_spread    FLOAT,
        vegas_spread_at_rec FLOAT,
        edge_points         FLOAT,
        win_prob            FLOAT,
        kelly_fraction      FLOAT,
        stake_units         FLOAT,
        bankroll_at_rec     FLOAT,
        odds_at_rec         FLOAT,
        model_version       VARCHAR        NOT NULL,
        staked              BOOLEAN        DEFAULT TRUE,
        PRIMARY KEY (bet_id)
    )
"""

_INSERT_BETS_SQL = f"""
    INSERT INTO {BETS_TABLE}
        (bet_id, game_id, season, week, recommended_at, side,
         predicted_spread, vegas_spread_at_rec, edge_points, win_prob,
         kelly_fraction, stake_units, bankroll_at_rec, odds_at_rec,
         model_version, staked)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def _to_bet_rows(
    recs: pl.DataFrame,
    *,
    season: int,
    week: int,
    model_version: str,
    bankroll: float,
    odds: float,
    recommended_at: datetime,
) -> List[tuple]:
    """Turn non-pass recommendations into INSERT-ready row tuples.

    Pure function (no Snowflake) so it can be unit-tested directly. One
    tuple per recommendation whose ``side`` is not 'pass'; column order
    matches ``_INSERT_BETS_SQL``.
    """
    stamp = recommended_at.replace(tzinfo=None)  # TIMESTAMP_NTZ expects naive UTC
    rows: List[tuple] = []
    for r in recs.iter_rows(named=True):
        if r["side"] == SIDE_PASS:
            continue
        rows.append(
            (
                str(uuid.uuid4()),
                str(r["game_id"]),
                int(season),
                int(week),
                stamp,
                str(r["side"]),
                _clean(r.get("predicted_spread")),
                _clean(r.get("vegas_spread")),
                _clean(r.get("edge_points")),
                _clean(r.get("win_prob")),
                _clean(r.get("kelly_fraction")),
                _clean(r.get("stake_units")),
                float(bankroll),
                float(odds),
                str(model_version),
                True,
            )
        )
    return rows


def _clean(value) -> Optional[float]:
    """Coerce a value to float, mapping NaN/None to None (SQL NULL)."""
    if value is None:
        return None
    f = float(value)
    if f != f:  # NaN
        return None
    return f


def record_bets(
    recs: pl.DataFrame,
    *,
    season: int,
    week: int,
    model_version: str = DEFAULT_MODEL_VERSION,
    bankroll: float,
    odds: float,
    recommended_at: Optional[datetime] = None,
    conn=None,
) -> int:
    """Persist non-pass recommendations to ML.BETS. Returns rows written.

    Re-materializing a week is idempotent: existing rows for the same
    ``(season, week, model_version)`` are deleted before the new batch is
    inserted, mirroring the delete-then-insert pattern in
    ``src.ml.predict.write_to_snowflake``. (Note: a manual ``staked=FALSE``
    edit is therefore reset if the week is re-run.)

    Args:
        recs: Output of ``build_recommendations`` (may include 'pass' rows,
            which are skipped).
        season: NFL season.
        week: NFL week.
        model_version: Tag separating champion vs challenger results.
        bankroll: Bankroll used for sizing (stored as ``bankroll_at_rec``).
        odds: American odds used for sizing (stored as ``odds_at_rec``).
        recommended_at: Timestamp to stamp rows with (defaults to now, UTC).
        conn: Optional open Snowflake connection (injected by tests). When
            None, opens one via ``get_snowflake_connection`` and closes it.

    Returns:
        Number of bet rows inserted.
    """
    if recommended_at is None:
        recommended_at = datetime.now(timezone.utc)

    rows = _to_bet_rows(
        recs,
        season=season,
        week=week,
        model_version=model_version,
        bankroll=bankroll,
        odds=odds,
        recommended_at=recommended_at,
    )

    owns_conn = conn is None
    if owns_conn:
        from src.ml.predict import get_snowflake_connection

        conn = get_snowflake_connection()

    try:
        cursor = conn.cursor()
        cursor.execute(_CREATE_BETS_SQL)
        cursor.execute(
            f"DELETE FROM {BETS_TABLE} "
            f"WHERE season = {int(season)} AND week = {int(week)} "
            f"AND model_version = %s",
            (str(model_version),),
        )
        if rows:
            cursor.executemany(_INSERT_BETS_SQL, rows)
        conn.commit()
        cursor.close()
    finally:
        if owns_conn:
            conn.close()

    return len(rows)
