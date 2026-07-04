-- ============================================================================
-- MART: Bet Performance
-- ============================================================================
-- Purpose: Production scorecard for the betting system. One row per rolling
--   window x model_version, so champion vs challenger results stay separable.
-- Grain: One row per (rolling_window, model_version).
-- Sources:
--   - ml.bets         (recorded recommendations; staked = TRUE only)
--   - ml.bet_results  (settled outcomes: outcome, profit, CLV)
-- Notes:
--   - win_rate excludes pushes (denominator = wins + losses).
--   - roi = sum(profit_units) / sum(stake_units).
--   - brier uses the model's win_prob against the realized win/loss (pushes
--     excluded). Lower is better -- CLAUDE.md's primary metric.
--   - avg_clv / clv_positive_rate measure closing-line value, the real edge
--     signal that stabilizes far faster than win rate.
-- ============================================================================

{{ config(materialized='view', tags=['marts', 'betting', 'performance']) }}

WITH staked_bets AS (
    SELECT
        bet_id,
        model_version,
        season,
        win_prob,
        stake_units
    FROM {{ source('ml', 'bets') }}
    WHERE staked = TRUE
),

results AS (
    SELECT
        bet_id,
        outcome,
        profit_units,
        clv_points,
        resolved_at
    FROM {{ source('ml', 'bet_results') }}
),

-- One row per settled, staked bet with everything needed to score it.
settled AS (
    SELECT
        b.model_version,
        b.season,
        b.win_prob,
        b.stake_units,
        r.outcome,
        r.profit_units,
        r.clv_points,
        r.resolved_at
    FROM staked_bets b
    JOIN results r ON b.bet_id = r.bet_id
),

current_season AS (
    SELECT MAX(season) AS cur_season FROM settled
),

-- Replicate each bet into every rolling-window bucket it belongs to, so a
-- single GROUP BY produces all windows at once.
expanded AS (
    SELECT
        s.*,
        w.rolling_window
    FROM settled s
    CROSS JOIN current_season cs
    CROSS JOIN (
        SELECT 'last_4_weeks'   AS rolling_window
        UNION ALL SELECT 'last_8_weeks'
        UNION ALL SELECT 'season_to_date'
        UNION ALL SELECT 'all_time'
    ) w
    WHERE
        w.rolling_window = 'all_time'
        OR (w.rolling_window = 'last_4_weeks'
            AND s.resolved_at >= DATEADD('day', -28, CURRENT_TIMESTAMP()))
        OR (w.rolling_window = 'last_8_weeks'
            AND s.resolved_at >= DATEADD('day', -56, CURRENT_TIMESTAMP()))
        OR (w.rolling_window = 'season_to_date'
            AND s.season = cs.cur_season)
)

SELECT
    rolling_window,
    model_version,

    COUNT(*) AS n_bets,
    SUM(CASE WHEN outcome = 'win'  THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) AS losses,
    SUM(CASE WHEN outcome = 'push' THEN 1 ELSE 0 END) AS pushes,

    -- Win rate excludes pushes.
    DIV0(
        SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END),
        NULLIF(SUM(CASE WHEN outcome IN ('win', 'loss') THEN 1 ELSE 0 END), 0)
    ) AS win_rate,

    -- ROI in stake units.
    DIV0(SUM(profit_units), NULLIF(SUM(stake_units), 0)) AS roi,

    -- Brier score vs realized outcome (pushes excluded).
    AVG(
        CASE WHEN outcome IN ('win', 'loss')
             THEN POWER(win_prob - (CASE WHEN outcome = 'win' THEN 1 ELSE 0 END), 2)
        END
    ) AS brier,

    -- Closing-line value.
    AVG(clv_points) AS avg_clv,
    DIV0(
        SUM(CASE WHEN clv_points > 0 THEN 1 ELSE 0 END),
        NULLIF(COUNT(clv_points), 0)
    ) AS clv_positive_rate

FROM expanded
GROUP BY 1, 2
