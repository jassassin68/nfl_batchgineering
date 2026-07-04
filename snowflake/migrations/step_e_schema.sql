-- ============================================================================
-- Step E: bet ledger schema
-- ============================================================================
-- Production tables that record placed bets and their settled results so the
-- system's real-money edge can be measured via CLV and ROI.
--
-- These tables live in PRODUCTION_ANALYTICS.ML alongside ML.PREDICTIONS and are
-- written by the Python recording/reconciliation layer (src/betting/ledger.py,
-- dagster_project/assets/reconciliation.py), NOT by dbt. dbt reads them as a
-- source (see dbt_project/models/3_marts/_bets_sources.yml) to build
-- mart_bet_performance.
--
-- Idempotent: safe to run repeatedly. The Python layer also issues
-- CREATE TABLE IF NOT EXISTS, so applying this migration up front is optional
-- but keeps the schema documented in one place under source control.
--
-- Closing lines: there is no separate BET_CLOSING_LINES table. The nflverse
-- spread at the first play of a game is the kickoff (closing) line, surfaced
-- as vegas_spread on mart_game_prediction_features; reconciliation reads it
-- from there.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS PRODUCTION_ANALYTICS.ML;

-- One row per recommended bet, captured at recommendation time. ------------
CREATE TABLE IF NOT EXISTS PRODUCTION_ANALYTICS.ML.BETS (
    bet_id              VARCHAR        NOT NULL,   -- uuid4, primary key
    game_id             VARCHAR        NOT NULL,
    season              NUMBER         NOT NULL,
    week                NUMBER         NOT NULL,
    recommended_at      TIMESTAMP_NTZ  NOT NULL,   -- UTC
    side                VARCHAR        NOT NULL,   -- 'home' | 'away'
    predicted_spread    FLOAT,
    vegas_spread_at_rec FLOAT,                     -- line when recommended
    edge_points         FLOAT,
    win_prob            FLOAT,                     -- prob of the side bet
    kelly_fraction      FLOAT,
    stake_units         FLOAT,
    bankroll_at_rec     FLOAT,
    odds_at_rec         FLOAT,
    model_version       VARCHAR        NOT NULL,   -- champion vs challenger key
    staked              BOOLEAN        DEFAULT TRUE,
    PRIMARY KEY (bet_id)
);

-- One row per resolved bet. clv_points stays populated once the closing -----
-- line is available (it always is, since it comes from completed-game pbp). --
CREATE TABLE IF NOT EXISTS PRODUCTION_ANALYTICS.ML.BET_RESULTS (
    bet_id              VARCHAR        NOT NULL,   -- FK -> BETS.bet_id
    game_id             VARCHAR        NOT NULL,
    home_score          NUMBER,
    away_score          NUMBER,
    margin              FLOAT,                     -- home_score - away_score
    vegas_spread_close  FLOAT,                     -- kickoff line
    outcome             VARCHAR,                   -- 'win' | 'loss' | 'push'
    profit_units        FLOAT,                     -- same units as stake_units
    clv_points          FLOAT,                     -- (close - at_rec) * side_mult
    resolved_at         TIMESTAMP_NTZ,             -- UTC
    PRIMARY KEY (bet_id)
);
