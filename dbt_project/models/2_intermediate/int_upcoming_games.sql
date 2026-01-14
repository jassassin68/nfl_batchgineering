-- ============================================================================
-- INTERMEDIATE: Upcoming Games with Latest Team Features
-- ============================================================================
-- Purpose: Join upcoming games (not yet played) with latest team features
-- Grain: One row per upcoming game
-- Dependencies:
--   - stgnv_schedules (game schedule with future games)
--   - mart_predictive_features (team offensive + defensive metrics)
--   - int_situational_efficiency (red zone, third down, two-minute drill)
-- ============================================================================

{{ config(
    materialized='view',
    tags=['intermediate', 'predictions', 'upcoming']
) }}

-- Get only upcoming games (where scores are NULL)
WITH upcoming_schedule AS (
    SELECT
        game_id,
        season,
        week,
        gameday,
        gametime,
        home_team,
        away_team,
        spread_line AS vegas_spread,
        total_line AS vegas_total,
        home_rest,
        away_rest,
        div_game,
        roof,
        surface,
        temp,
        wind,
        stadium,
        home_qb_name,
        away_qb_name,
        home_coach,
        away_coach
    FROM {{ ref('stgnv_schedules') }}
    WHERE home_score IS NULL  -- Future games have no score
      AND game_type = 'REG'   -- Regular season only (adjust as needed)
),

-- Get the latest week of features for each team in the current season
latest_team_features AS (
    SELECT *
    FROM (
        SELECT
            season,
            week,
            team,
            epa_per_play_adj,
            epa_per_play_l4w,
            success_rate,
            success_rate_l4w,
            explosive_play_rate,
            pass_epa_adj,
            run_epa_adj,
            redzone_epa_adj,
            pass_rate,
            weather_adjusted_epa_projection,
            offensive_matchup_score,
            composite_efficiency_score,
            epa_trend,
            epa_rolling_volatility,
            efficiency_tier,
            weather_resistance_tier,
            season_def_epa_allowed,
            def_epa_rank,
            def_strength_tier,
            season_def_pass_epa,
            season_def_run_epa,
            def_epa_allowed_l4w,
            season_def_success_rate_allowed,
            ROW_NUMBER() OVER (PARTITION BY team, season ORDER BY week DESC) AS rn
        FROM {{ ref('mart_predictive_features') }}
    ) ranked
    WHERE rn = 1
),

-- Get latest situational features for each team
latest_situational AS (
    SELECT *
    FROM (
        SELECT
            season,
            week,
            team,
            red_zone_epa AS rz_epa,
            red_zone_success_rate AS rz_success,
            red_zone_td_rate AS rz_td_rate,
            third_down_conversion_rate AS third_conv_rate,
            third_down_epa AS third_epa,
            two_minute_drill_epa AS two_min_epa,
            two_minute_success_rate AS two_min_success,
            ROW_NUMBER() OVER (PARTITION BY team, season ORDER BY week DESC) AS rn
        FROM {{ ref('int_situational_efficiency') }}
    ) ranked
    WHERE rn = 1
)

-- Join upcoming games with home team features
SELECT
    s.game_id,
    s.season,
    s.week,
    s.gameday,
    s.gametime,
    s.home_team,
    s.away_team,
    s.vegas_spread,
    s.vegas_total,
    s.home_rest,
    s.away_rest,
    s.div_game,
    s.roof,
    s.surface,
    s.temp,
    s.wind,
    s.stadium,
    s.home_qb_name,
    s.away_qb_name,
    s.home_coach,
    s.away_coach,

    -- Home team offensive features
    htf.epa_per_play_adj AS home_epa_adj,
    htf.epa_per_play_l4w AS home_epa_l4w,
    htf.success_rate AS home_success_rate,
    htf.success_rate_l4w AS home_success_l4w,
    htf.explosive_play_rate AS home_explosive_rate,
    htf.pass_epa_adj AS home_pass_epa,
    htf.run_epa_adj AS home_run_epa,
    htf.redzone_epa_adj AS home_rz_epa,
    htf.pass_rate AS home_pass_rate,
    htf.weather_adjusted_epa_projection AS home_weather_epa,
    htf.offensive_matchup_score AS home_matchup_score,
    htf.composite_efficiency_score AS home_efficiency_score,
    htf.epa_trend AS home_epa_trend,
    htf.epa_rolling_volatility AS home_volatility,
    htf.efficiency_tier AS home_tier,
    htf.weather_resistance_tier AS home_weather_resistance,

    -- Home team defensive features
    htf.season_def_epa_allowed AS home_def_epa,
    htf.def_epa_rank AS home_def_rank,
    htf.def_strength_tier AS home_def_tier,
    htf.season_def_pass_epa AS home_def_pass_epa,
    htf.season_def_run_epa AS home_def_run_epa,
    htf.def_epa_allowed_l4w AS home_def_epa_l4w,
    htf.season_def_success_rate_allowed AS home_def_success_allowed,

    -- Home team situational
    hs.rz_td_rate AS home_rz_td_rate,
    hs.third_conv_rate AS home_third_conv,
    hs.two_min_epa AS home_two_min_epa,

    -- Away team offensive features
    atf.epa_per_play_adj AS away_epa_adj,
    atf.epa_per_play_l4w AS away_epa_l4w,
    atf.success_rate AS away_success_rate,
    atf.success_rate_l4w AS away_success_l4w,
    atf.explosive_play_rate AS away_explosive_rate,
    atf.pass_epa_adj AS away_pass_epa,
    atf.run_epa_adj AS away_run_epa,
    atf.redzone_epa_adj AS away_rz_epa,
    atf.pass_rate AS away_pass_rate,
    atf.weather_adjusted_epa_projection AS away_weather_epa,
    atf.offensive_matchup_score AS away_matchup_score,
    atf.composite_efficiency_score AS away_efficiency_score,
    atf.epa_trend AS away_epa_trend,
    atf.epa_rolling_volatility AS away_volatility,
    atf.efficiency_tier AS away_tier,
    atf.weather_resistance_tier AS away_weather_resistance,

    -- Away team defensive features
    atf.season_def_epa_allowed AS away_def_epa,
    atf.def_epa_rank AS away_def_rank,
    atf.def_strength_tier AS away_def_tier,
    atf.season_def_pass_epa AS away_def_pass_epa,
    atf.season_def_run_epa AS away_def_run_epa,
    atf.def_epa_allowed_l4w AS away_def_epa_l4w,
    atf.season_def_success_rate_allowed AS away_def_success_allowed,

    -- Away team situational
    aws.rz_td_rate AS away_rz_td_rate,
    aws.third_conv_rate AS away_third_conv,
    aws.two_min_epa AS away_two_min_epa

FROM upcoming_schedule s
LEFT JOIN latest_team_features htf
    ON s.home_team = htf.team
    AND s.season = htf.season
LEFT JOIN latest_situational hs
    ON s.home_team = hs.team
    AND s.season = hs.season
LEFT JOIN latest_team_features atf
    ON s.away_team = atf.team
    AND s.season = atf.season
LEFT JOIN latest_situational aws
    ON s.away_team = aws.team
    AND s.season = aws.season
