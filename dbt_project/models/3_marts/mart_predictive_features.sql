-- models/3_marts/mart_predictive_features.sql
-- Unified predictive features combining offensive, defensive, and situational metrics
-- Maps intermediate model columns to expected feature names for downstream use

{{ config(
    materialized='table',
    tags=['marts', 'predictions', 'team_level']
) }}

WITH offensive_features AS (
    SELECT
        team,
        season,
        week,

        -- Map to expected column names
        rolling_4wk_epa_per_play AS epa_per_play_l4w,
        avg_epa_per_play AS epa_per_play_adj,  -- Using weekly avg as "adjusted"
        rolling_4wk_success_rate AS success_rate_l4w,
        success_rate,
        rolling_4wk_explosive_rate AS explosive_play_rate,
        rolling_4wk_pass_epa AS pass_epa_adj,
        rolling_4wk_rush_epa AS run_epa_adj,
        rolling_4wk_red_zone_td_rate AS redzone_epa_adj,

        -- Calculate pass rate
        CAST(pass_attempts AS FLOAT) / NULLIF(total_plays, 0) AS pass_rate,

        -- Placeholder features (can be enhanced later)
        rolling_4wk_epa_per_play AS weather_adjusted_epa_projection,  -- Simplified for now
        rolling_4wk_epa_per_play AS offensive_matchup_score,  -- Simplified for now
        rolling_4wk_success_rate AS composite_efficiency_score,  -- Using success rate

        -- EPA trend (current week vs 4-week average)
        avg_epa_per_play - rolling_4wk_epa_per_play AS epa_trend,

        -- Volatility placeholder (can calculate std dev in future)
        0.1 AS epa_rolling_volatility,  -- Placeholder constant

        -- Efficiency tier classification
        CASE
            WHEN rolling_4wk_epa_per_play >= 0.15 THEN 'Elite'
            WHEN rolling_4wk_epa_per_play >= 0.05 THEN 'Good'
            WHEN rolling_4wk_epa_per_play >= -0.05 THEN 'Average'
            ELSE 'Poor'
        END AS efficiency_tier,

        -- Weather resistance tier (placeholder - can enhance with actual weather data)
        CASE
            WHEN rolling_4wk_epa_per_play >= 0.1 THEN 'High'
            WHEN rolling_4wk_epa_per_play >= 0.0 THEN 'Medium'
            ELSE 'Low'
        END AS weather_resistance_tier

    FROM {{ ref('int_team_offensive_metrics') }}
),

defensive_features AS (
    SELECT
        team,
        season,
        week,

        -- Map defensive metrics to expected names
        rolling_4wk_epa_allowed AS season_def_epa_allowed,
        defensive_epa_rank AS def_epa_rank,
        rolling_4wk_pass_epa_allowed AS season_def_pass_epa,
        rolling_4wk_rush_epa_allowed AS season_def_run_epa,
        rolling_4wk_epa_allowed AS def_epa_allowed_l4w,
        rolling_4wk_success_rate_allowed AS season_def_success_rate_allowed,

        -- Defensive strength tier
        CASE
            WHEN defensive_epa_rank <= 8 THEN 'Elite'
            WHEN defensive_epa_rank <= 16 THEN 'Good'
            WHEN defensive_epa_rank <= 24 THEN 'Average'
            ELSE 'Poor'
        END AS def_strength_tier

    FROM {{ ref('int_team_defensive_strength') }}
)

-- Join offensive and defensive features
SELECT
    o.team,
    o.season,
    o.week,

    -- Offensive features
    o.epa_per_play_adj,
    o.epa_per_play_l4w,
    o.success_rate,
    o.success_rate_l4w,
    o.explosive_play_rate,
    o.pass_epa_adj,
    o.run_epa_adj,
    o.redzone_epa_adj,
    o.pass_rate,
    o.weather_adjusted_epa_projection,
    o.offensive_matchup_score,
    o.composite_efficiency_score,
    o.epa_trend,
    o.epa_rolling_volatility,
    o.efficiency_tier,
    o.weather_resistance_tier,

    -- Defensive features
    d.season_def_epa_allowed,
    d.def_epa_rank,
    d.def_strength_tier,
    d.season_def_pass_epa,
    d.season_def_run_epa,
    d.def_epa_allowed_l4w,
    d.season_def_success_rate_allowed

FROM offensive_features o
LEFT JOIN defensive_features d
    ON o.team = d.team
    AND o.season = d.season
    AND o.week = d.week
