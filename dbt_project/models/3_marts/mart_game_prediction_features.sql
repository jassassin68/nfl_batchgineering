-- models/marts/mart_game_prediction_features.sql
-- Game-level features combining both teams for spread and totals prediction

{{ config(
    materialized='table',
    indexes=[
        {'columns': ['game_id']},
        {'columns': ['season', 'week']},
        {'columns': ['home_team', 'away_team']}
    ]
) }}

WITH game_schedule AS (
    -- Derive game schedule from play-by-play data since schedule source doesn't exist
    SELECT DISTINCT
        game_id,
        season,
        week,
        game_date AS gameday,
        NULL AS gametime,  -- Not available in play-by-play
        home_team,
        away_team,
        -- Get final scores from play-by-play
        MAX(CASE WHEN posteam = home_team THEN posteam_score_post
                 WHEN defteam = home_team THEN defteam_score_post
                 END) AS home_score,
        MAX(CASE WHEN posteam = away_team THEN posteam_score_post
                 WHEN defteam = away_team THEN defteam_score_post
                 END) AS away_score
    FROM {{ ref('int_plays_cleaned') }}
    WHERE season >= 2020  -- Adjust based on your needs
    GROUP BY 1, 2, 3, 4, 5, 6, 7
),

-- Get weather for each game from play-by-play data
game_weather AS (
    SELECT DISTINCT
        game_id,
        temp,
        wind,
        roof,
        surface
    FROM {{ ref('int_plays_cleaned') }}
    WHERE temp IS NOT NULL OR roof IN ('dome', 'closed')
),

-- Get division game flag
game_context AS (
    SELECT 
        gs.game_id,
        gs.season,
        gs.week,
        gs.gameday,
        gs.gametime,
        gs.home_team,
        gs.away_team,
        gs.home_score,
        gs.away_score,
        
        -- Weather
        COALESCE(gw.temp, 72) AS temp,  -- Default to 72 for domes
        COALESCE(gw.wind, 0) AS wind,   -- Default to 0 for domes
        COALESCE(gw.roof, 'outdoors') AS roof,
        gw.surface,
        
        -- Division game indicator (simplified - you may have a divisions table)
        CASE 
            WHEN gs.home_team IN ('BAL', 'CIN', 'CLE', 'PIT') 
                AND gs.away_team IN ('BAL', 'CIN', 'CLE', 'PIT') THEN 1
            WHEN gs.home_team IN ('BUF', 'MIA', 'NE', 'NYJ') 
                AND gs.away_team IN ('BUF', 'MIA', 'NE', 'NYJ') THEN 1
            WHEN gs.home_team IN ('HOU', 'IND', 'JAX', 'TEN') 
                AND gs.away_team IN ('HOU', 'IND', 'JAX', 'TEN') THEN 1
            WHEN gs.home_team IN ('DEN', 'KC', 'LV', 'LAC') 
                AND gs.away_team IN ('DEN', 'KC', 'LV', 'LAC') THEN 1
            WHEN gs.home_team IN ('DAL', 'NYG', 'PHI', 'WAS') 
                AND gs.away_team IN ('DAL', 'NYG', 'PHI', 'WAS') THEN 1
            WHEN gs.home_team IN ('CHI', 'DET', 'GB', 'MIN') 
                AND gs.away_team IN ('CHI', 'DET', 'GB', 'MIN') THEN 1
            WHEN gs.home_team IN ('ATL', 'CAR', 'NO', 'TB') 
                AND gs.away_team IN ('ATL', 'CAR', 'NO', 'TB') THEN 1
            WHEN gs.home_team IN ('ARI', 'LAR', 'SF', 'SEA') 
                AND gs.away_team IN ('ARI', 'LAR', 'SF', 'SEA') THEN 1
            ELSE 0
        END AS div_game,
        
        -- Playoff game
        CASE WHEN gs.week >= 18 THEN 1 ELSE 0 END AS playoff
        
    FROM game_schedule gs
    LEFT JOIN game_weather gw ON gs.game_id = gw.game_id
),

-- Get team offensive features (when they play at home)
home_team_offense AS (
    SELECT 
        pf.season,
        pf.week,
        pf.team,
        
        -- Core offensive metrics
        pf.epa_per_play_adj AS off_epa_adj,
        pf.epa_per_play_l4w AS off_epa_l4w,
        pf.success_rate AS off_success_rate,
        pf.success_rate_l4w AS off_success_l4w,
        pf.explosive_play_rate AS off_explosive_rate,
        pf.pass_epa_adj AS off_pass_epa,
        pf.run_epa_adj AS off_run_epa,
        pf.redzone_epa_adj AS off_rz_epa,
        pf.pass_rate AS off_pass_rate,
        pf.weather_adjusted_epa_projection AS off_weather_epa,
        pf.offensive_matchup_score AS off_matchup_score,
        pf.composite_efficiency_score AS off_efficiency_score,
        pf.epa_trend AS off_epa_trend,
        pf.epa_rolling_volatility AS off_volatility,
        pf.efficiency_tier AS off_tier,
        pf.weather_resistance_tier AS off_weather_resistance
        
    FROM {{ ref('mart_predictive_features') }} pf
),

-- Get team defensive features
team_defense AS (
    SELECT 
        team,
        season,
        week,
        season_def_epa_allowed AS def_epa_allowed,
        def_epa_rank AS def_rank,
        def_strength_tier AS def_tier,
        season_def_pass_epa AS def_pass_epa,
        season_def_run_epa AS def_run_epa,
        def_epa_allowed_l4w AS def_epa_l4w,
        season_def_success_rate_allowed AS def_success_allowed
    FROM {{ ref('int_team_defensive_strength') }}
),

-- Get situational features
team_situational AS (
    SELECT 
        season,
        week,
        team,
        
        -- Red zone efficiency
        MAX(CASE WHEN situation_type = 'red_zone' THEN epa_per_play END) AS rz_epa,
        MAX(CASE WHEN situation_type = 'red_zone' THEN success_rate END) AS rz_success,
        MAX(CASE WHEN situation_type = 'red_zone' THEN td_rate END) AS rz_td_rate,
        
        -- Third down efficiency
        MAX(CASE WHEN situation_type = 'third_down' THEN conversion_rate END) AS third_conv_rate,
        MAX(CASE WHEN situation_type = 'third_down' THEN epa_per_play END) AS third_epa,
        
        -- Two minute efficiency
        MAX(CASE WHEN situation_type = 'two_minute' THEN epa_per_play END) AS two_min_epa,
        MAX(CASE WHEN situation_type = 'two_minute' THEN success_rate END) AS two_min_success
        
    FROM {{ ref('int_situational_efficiency') }}
    GROUP BY season, week, team
),

-- Combine all home team features
home_team_combined AS (
    SELECT 
        gc.game_id,
        gc.season,
        gc.week,
        gc.gameday,
        gc.gametime,
        gc.home_team,
        gc.away_team,
        gc.home_score,
        gc.away_score,
        gc.temp,
        gc.wind,
        gc.roof,
        gc.surface,
        gc.div_game,
        gc.playoff,
        
        -- Home team offensive features
        ho.off_epa_adj AS home_epa_adj,
        ho.off_epa_l4w AS home_epa_l4w,
        ho.off_success_rate AS home_success_rate,
        ho.off_success_l4w AS home_success_l4w,
        ho.off_explosive_rate AS home_explosive_rate,
        ho.off_pass_epa AS home_pass_epa,
        ho.off_run_epa AS home_run_epa,
        ho.off_rz_epa AS home_rz_epa,
        ho.off_pass_rate AS home_pass_rate,
        ho.off_weather_epa AS home_weather_epa,
        ho.off_matchup_score AS home_matchup_score,
        ho.off_efficiency_score AS home_efficiency_score,
        ho.off_epa_trend AS home_epa_trend,
        ho.off_volatility AS home_volatility,
        ho.off_tier AS home_tier,
        ho.off_weather_resistance AS home_weather_resistance,
        
        -- Home team defensive features
        hd.def_epa_allowed AS home_def_epa,
        hd.def_rank AS home_def_rank,
        hd.def_tier AS home_def_tier,
        hd.def_pass_epa AS home_def_pass_epa,
        hd.def_run_epa AS home_def_run_epa,
        hd.def_epa_l4w AS home_def_epa_l4w,
        
        -- Home team situational
        hs.rz_epa AS home_situation_rz_epa,
        hs.rz_success AS home_situation_rz_success,
        hs.rz_td_rate AS home_rz_td_rate,
        hs.third_conv_rate AS home_third_conv,
        hs.third_epa AS home_third_epa,
        hs.two_min_epa AS home_two_min_epa
        
    FROM game_context gc
    LEFT JOIN home_team_offense ho
        ON gc.home_team = ho.team
        AND gc.season = ho.season
        AND gc.week = ho.week
    LEFT JOIN team_defense hd
        ON gc.home_team = hd.team
        AND gc.season = hd.season
        AND gc.week = hd.week
    LEFT JOIN team_situational hs
        ON gc.home_team = hs.team
        AND gc.season = hs.season
        AND gc.week = hs.week
),

-- Add away team features
combined_features AS (
    SELECT 
        htc.*,
        
        -- Away team offensive features
        ao.off_epa_adj AS away_epa_adj,
        ao.off_epa_l4w AS away_epa_l4w,
        ao.off_success_rate AS away_success_rate,
        ao.off_success_l4w AS away_success_l4w,
        ao.off_explosive_rate AS away_explosive_rate,
        ao.off_pass_epa AS away_pass_epa,
        ao.off_run_epa AS away_run_epa,
        ao.off_rz_epa AS away_rz_epa,
        ao.off_pass_rate AS away_pass_rate,
        ao.off_weather_epa AS away_weather_epa,
        ao.off_matchup_score AS away_matchup_score,
        ao.off_efficiency_score AS away_efficiency_score,
        ao.off_epa_trend AS away_epa_trend,
        ao.off_volatility AS away_volatility,
        ao.off_tier AS away_tier,
        ao.off_weather_resistance AS away_weather_resistance,
        
        -- Away team defensive features
        ad.def_epa_allowed AS away_def_epa,
        ad.def_rank AS away_def_rank,
        ad.def_tier AS away_def_tier,
        ad.def_pass_epa AS away_def_pass_epa,
        ad.def_run_epa AS away_def_run_epa,
        ad.def_epa_l4w AS away_def_epa_l4w,
        
        -- Away team situational
        aws.rz_epa AS away_situation_rz_epa,
        aws.rz_success AS away_situation_rz_success,
        aws.rz_td_rate AS away_rz_td_rate,
        aws.third_conv_rate AS away_third_conv,
        aws.third_epa AS away_third_epa,
        aws.two_min_epa AS away_two_min_epa
        
    FROM home_team_combined htc
    LEFT JOIN home_team_offense ao
        ON htc.away_team = ao.team
        AND htc.season = ao.season
        AND htc.week = ao.week
    LEFT JOIN team_defense ad
        ON htc.away_team = ad.team
        AND htc.season = ad.season
        AND htc.week = ad.week
    LEFT JOIN team_situational aws
        ON htc.away_team = aws.team
        AND htc.season = aws.season
        AND htc.week = aws.week
)

SELECT 
    game_id,
    season,
    week,
    gameday,
    gametime,
    home_team,
    away_team,
    home_score,
    away_score,
    temp,
    wind,
    roof,
    surface,
    div_game,
    playoff,
    
    -- Home team features
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
    home_rz_td_rate,
    home_third_conv,
    home_two_min_epa,
    
    -- Away team features
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
    away_rz_td_rate,
    away_third_conv,
    away_two_min_epa,
    
    -- ============================================================
    -- SPREAD PREDICTION FEATURES
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
    -- TOTALS (OVER/UNDER) PREDICTION FEATURES
    -- ============================================================
    
    -- Combined Offensive Efficiency
    (home_epa_adj + away_epa_adj) AS combined_offensive_epa,
    (home_efficiency_score + away_efficiency_score) AS combined_efficiency_score,
    (home_explosive_rate + away_explosive_rate) AS combined_explosive_rate,
    
    -- Combined Defensive Strength (lower EPA allowed = better defense = lower scoring)
    (home_def_epa + away_def_epa) AS combined_defensive_epa,
    
    -- Pace/Volume Indicators
    (home_pass_rate + away_pass_rate) / 2.0 AS avg_pass_rate,
    
    -- Red Zone Efficiency (affects scoring)
    (COALESCE(home_rz_td_rate, 0) + COALESCE(away_rz_td_rate, 0)) AS combined_rz_efficiency,
    
    -- ============================================================
    -- WEATHER IMPACT FEATURES
    -- ============================================================
    
    -- Weather Impact on Totals
    CASE 
        WHEN wind >= 20 THEN -4.5  -- Very high wind severely reduces scoring
        WHEN wind >= 15 THEN -3.5  -- High wind reduces scoring
        WHEN temp <= 20 THEN -3.0  -- Extreme cold reduces scoring
        WHEN temp <= 32 THEN -2.0  -- Cold weather reduces scoring
        WHEN temp >= 90 THEN -2.0  -- Extreme heat reduces scoring
        WHEN temp >= 85 THEN -1.5  -- Hot weather slight reduction
        ELSE 0
    END AS weather_totals_adjustment,
    
    -- Weather impact on spread (affects passing more than running)
    CASE 
        WHEN wind >= 15 THEN 
            (home_pass_rate - away_pass_rate) * -0.5  -- Hurts pass-heavy team
        ELSE 0
    END AS weather_spread_adjustment,
    
    -- ============================================================
    -- GAME CONTEXT FEATURES
    -- ============================================================
    
    -- Division game (typically lower scoring, more competitive)
    div_game AS is_division_game,
    playoff AS is_playoff_game,
    
    -- Home Field Advantage (modern NFL ~1.5-2 points)
    CASE 
        WHEN roof IN ('dome', 'closed') THEN 1.5  -- Indoor venues
        WHEN roof = 'open' THEN 2.0               -- Outdoor retractable (open)
        WHEN roof = 'outdoors' AND temp BETWEEN 45 AND 75 THEN 2.0  -- Ideal outdoor
        WHEN roof = 'outdoors' AND (temp < 45 OR temp > 75) THEN 1.75  -- Adverse outdoor
        ELSE 1.75  -- Average
    END AS estimated_home_field_advantage,
    
    -- ============================================================
    -- VOLATILITY/CONSISTENCY MEASURES
    -- ============================================================
    
    (home_volatility + away_volatility) AS combined_volatility,
    ABS(home_epa_trend - away_epa_trend) AS trend_divergence,
    
    -- ============================================================
    -- PREDICTION MODELS (Simple Baseline)
    -- ============================================================
    
    -- SPREAD PREDICTIONS
    -- Simple EPA-based spread prediction
    ((home_epa_adj - away_epa_adj) * 14) + 
    CASE 
        WHEN roof IN ('dome', 'closed') THEN 1.5
        WHEN roof = 'outdoors' AND temp BETWEEN 45 AND 75 THEN 2.0
        ELSE 1.75
    END AS simple_spread_prediction,
    
    -- Enhanced spread prediction
    ((home_epa_adj - away_epa_adj) * 12) + 
    ((home_success_rate - away_success_rate) * 8) + 
    ((home_off_vs_away_def - away_off_vs_home_def) * 0.5) +
    CASE 
        WHEN roof IN ('dome', 'closed') THEN 1.5
        ELSE 2.0
    END +
    CASE WHEN wind >= 15 THEN (home_pass_rate - away_pass_rate) * -0.5 ELSE 0 END AS enhanced_spread_prediction,
    
    -- TOTALS PREDICTIONS
    -- Base total from offensive efficiency
    42 + ((home_epa_adj + away_epa_adj) * 25) - ((home_def_epa + away_def_epa) * 15) AS base_total_prediction,
    
    -- Weather-adjusted total
    42 + ((home_epa_adj + away_epa_adj) * 25) - ((home_def_epa + away_def_epa) * 15) + 
    CASE 
        WHEN wind >= 20 THEN -4.5
        WHEN wind >= 15 THEN -3.5
        WHEN temp <= 20 THEN -3.0
        WHEN temp <= 32 THEN -2.0
        WHEN temp >= 90 THEN -2.0
        WHEN temp >= 85 THEN -1.5
        ELSE 0
    END AS weather_adj_total_prediction,
    
    -- Enhanced total with all factors
    42 + 
    ((home_epa_adj + away_epa_adj) * 20) - 
    ((home_def_epa + away_def_epa) * 12) +
    ((home_explosive_rate + away_explosive_rate) * 15) +
    ((COALESCE(home_rz_td_rate, 0) + COALESCE(away_rz_td_rate, 0)) * 8) +
    CASE 
        WHEN wind >= 20 THEN -4.5
        WHEN wind >= 15 THEN -3.5
        WHEN temp <= 20 THEN -3.0
        WHEN temp <= 32 THEN -2.0
        WHEN temp >= 90 THEN -2.0
        WHEN temp >= 85 THEN -1.5
        ELSE 0
    END +
    CASE WHEN div_game = 1 THEN -2.5 ELSE 0 END AS enhanced_total_prediction,
    
    -- ============================================================
    -- BINARY CLASSIFICATION FEATURES (for ML models)
    -- ============================================================
    
    CASE WHEN (home_epa_adj - away_epa_adj) > 0.1 THEN 1 ELSE 0 END AS home_strong_advantage,
    CASE WHEN ABS(home_epa_adj - away_epa_adj) <= 0.05 THEN 1 ELSE 0 END AS close_matchup,
    CASE WHEN (home_epa_adj + away_epa_adj) > 0.2 THEN 1 ELSE 0 END AS high_scoring_expected,
    CASE WHEN wind >= 15 THEN 1 ELSE 0 END AS high_wind_game,
    CASE WHEN temp <= 32 THEN 1 ELSE 0 END AS cold_weather_game,
    CASE WHEN roof IN ('dome', 'closed') THEN 1 ELSE 0 END AS indoor_game,
    
    -- ============================================================
    -- NORMALIZED FEATURES (0-1 scale for ML)
    -- ============================================================
    
    -- Assumes EPA differential range of -0.4 to +0.4
    ((home_epa_adj - away_epa_adj) + 0.4) / 0.8 AS epa_diff_normalized,
    
    -- Assumes combined offensive EPA range of -0.2 to +0.4
    LEAST(((home_epa_adj + away_epa_adj) + 0.2) / 0.6, 1.0) AS total_offense_normalized,
    
    -- Assumes combined defensive EPA range of -0.2 to +0.4
    GREATEST(((home_def_epa + away_def_epa) + 0.2) / 0.6, 0.0) AS total_defense_normalized,
    
    -- Success rate already 0-1
    (home_success_rate + away_success_rate) / 2.0 AS avg_success_rate,
    
    -- ============================================================
    -- CONFIDENCE INDICATORS
    -- ============================================================
    
    CASE 
        WHEN ABS((home_epa_adj - away_epa_adj) * 14) >= 7 THEN 'High'
        WHEN ABS((home_epa_adj - away_epa_adj) * 14) >= 3 THEN 'Medium'
        ELSE 'Low'
    END AS spread_confidence,
    
    CASE 
        WHEN ABS(42 + ((home_epa_adj + away_epa_adj) * 25) - 47) >= 6 THEN 'High'
        WHEN ABS(42 + ((home_epa_adj + away_epa_adj) * 25) - 47) >= 3 THEN 'Medium'
        ELSE 'Low'
    END AS total_confidence

FROM combined_features
WHERE home_epa_adj IS NOT NULL 
  AND away_epa_adj IS NOT NULL  -- Ensure both teams have data