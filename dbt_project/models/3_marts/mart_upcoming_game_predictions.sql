-- ============================================================================
-- MART: Upcoming Game Predictions
-- ============================================================================
-- Purpose: ML-ready feature dataset for predicting upcoming NFL games
-- Grain: One row per upcoming game (games not yet played)
-- Uses same feature structure as mart_game_prediction_features for ML compatibility
-- ============================================================================

{{ config(
    materialized='view',
    tags=['marts', 'predictions', 'upcoming']
) }}

WITH base_features AS (
    SELECT *
    FROM {{ ref('int_upcoming_games') }}
)

SELECT
    -- ========================================================================
    -- IDENTIFIERS & GAME CONTEXT
    -- ========================================================================
    game_id,
    season,
    week,
    gameday,
    gametime,
    home_team,
    away_team,
    NULL AS home_score,  -- Future game - no score yet
    NULL AS away_score,  -- Future game - no score yet
    COALESCE(temp, 72) AS temp,  -- Default to 72 for domes
    COALESCE(wind, 0) AS wind,   -- Default to 0 for domes
    COALESCE(roof, 'outdoors') AS roof,
    surface,
    COALESCE(div_game, 0) AS div_game,
    CASE WHEN week >= 18 THEN 1 ELSE 0 END AS playoff,

    -- Vegas lines and betting context
    vegas_spread,
    vegas_total,
    -- Calculate implied home win probability from spread
    CASE
        WHEN vegas_spread::number < 0 THEN 0.50 - (ABS(vegas_spread) * 0.033)
        WHEN vegas_spread::number > 0 THEN 0.50 + (ABS(vegas_spread) * 0.033)
        ELSE 0.50
    END AS vegas_home_win_prob,
    CASE
        WHEN ABS(vegas_spread) <= 2.5 THEN 'Pick-em'
        WHEN ABS(vegas_spread) <= 6.5 THEN 'Small'
        WHEN ABS(vegas_spread) <= 10.5 THEN 'Medium'
        ELSE 'Large'
    END AS spread_category,

    -- Game context
    home_rest,
    away_rest,
    home_qb_name,
    away_qb_name,
    home_coach,
    away_coach,
    stadium,

    -- ========================================================================
    -- HOME TEAM FEATURES
    -- ========================================================================
    home_epa_adj,
    home_epa_l4w,
    home_success_rate,
    home_success_l4w,
    home_explosive_rate,
    home_pass_epa,
    home_run_epa,
    home_rz_epa,
    home_pass_rate,
    home_weather_epa,
    home_matchup_score,
    home_efficiency_score,
    home_epa_trend,
    home_volatility,
    home_tier,
    home_weather_resistance,
    home_def_epa,
    home_def_rank,
    home_def_tier,
    home_def_pass_epa,
    home_def_run_epa,
    home_def_epa_l4w,
    home_def_success_allowed,
    home_rz_td_rate,
    home_third_conv,
    home_two_min_epa,

    -- ========================================================================
    -- AWAY TEAM FEATURES
    -- ========================================================================
    away_epa_adj,
    away_epa_l4w,
    away_success_rate,
    away_success_l4w,
    away_explosive_rate,
    away_pass_epa,
    away_run_epa,
    away_rz_epa,
    away_pass_rate,
    away_weather_epa,
    away_matchup_score,
    away_efficiency_score,
    away_epa_trend,
    away_volatility,
    away_tier,
    away_weather_resistance,
    away_def_epa,
    away_def_rank,
    away_def_tier,
    away_def_pass_epa,
    away_def_run_epa,
    away_def_epa_l4w,
    away_def_success_allowed,
    away_rz_td_rate,
    away_third_conv,
    away_two_min_epa,

    -- ============================================================
    -- DERIVED FEATURES (same as mart_game_prediction_features)
    -- ============================================================

    -- EPA Differentials (positive = home team advantage)
    (home_epa_adj - away_epa_adj) AS epa_differential,
    (home_epa_l4w - away_epa_l4w) AS epa_l4w_differential,
    (home_weather_epa - away_weather_epa) AS weather_adj_epa_diff,

    -- Success Rate Differentials
    (home_success_rate - away_success_rate) AS success_rate_differential,
    (home_success_l4w - away_success_l4w) AS success_rate_l4w_diff,

    -- Explosive Play Differentials
    (home_explosive_rate - away_explosive_rate) AS explosive_rate_diff,

    -- Matchup Advantages (offense vs opponent defense)
    (home_epa_adj - away_def_epa) AS home_off_vs_away_def,
    (away_epa_adj - home_def_epa) AS away_off_vs_home_def,
    ((home_epa_adj - away_def_epa) - (away_epa_adj - home_def_epa)) AS net_matchup_advantage,

    -- Efficiency Score Differential
    (home_efficiency_score - away_efficiency_score) AS efficiency_score_diff,

    -- Situational Differentials
    (home_rz_td_rate - away_rz_td_rate) AS rz_td_rate_diff,
    (home_third_conv - away_third_conv) AS third_down_diff,

    -- ============================================================
    -- SIMPLE BASELINE PREDICTIONS
    -- ============================================================

    -- Simple EPA-based spread prediction
    ((home_epa_adj - away_epa_adj) * 14) +
    CASE
        WHEN COALESCE(roof, 'outdoors') IN ('dome', 'closed') THEN 1.5
        WHEN COALESCE(roof, 'outdoors') = 'outdoors' AND COALESCE(temp, 72) BETWEEN 45 AND 75 THEN 2.0
        ELSE 1.75
    END AS simple_spread_prediction,

    -- Base total from offensive efficiency
    42 + ((home_epa_adj + away_epa_adj) * 25) - ((home_def_epa + away_def_epa) * 15) AS base_total_prediction

FROM base_features
WHERE home_epa_adj IS NOT NULL
  AND away_epa_adj IS NOT NULL
