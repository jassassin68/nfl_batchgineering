-- models/marts/mart_weather_impact_metrics.sql
-- Weather-adjusted team efficiency metrics

{{ config(
    materialized='table'
) }}

WITH weather_conditions_raw AS (
    -- Get weather data by game
    SELECT
        game_id,
        home_team,
        away_team,
        MAX(CAST(wind AS FLOAT)) AS max_wind,
        MAX(CAST(temp AS FLOAT)) AS max_temp
    FROM {{ ref('stgnv_play_by_play') }}
    WHERE wind IS NOT NULL OR temp IS NOT NULL
    GROUP BY game_id, home_team, away_team
),

weather_conditions AS (
    -- Compute weather condition flags from aggregated data
    SELECT
        game_id,
        home_team,
        away_team,
        max_wind,
        max_temp,

        -- Define adverse weather conditions
        CASE
            WHEN max_wind >= 15 THEN 1
            WHEN max_temp <= 32 THEN 1
            WHEN max_temp >= 85 THEN 1
            ELSE 0
        END AS adverse_weather,

        -- Weather severity categories
        CASE
            WHEN max_wind >= 20 OR max_temp <= 20 OR max_temp >= 90 THEN 'Severe'
            WHEN max_wind >= 15 OR max_temp <= 32 OR max_temp >= 85 THEN 'Moderate'
            ELSE 'Mild'
        END AS weather_severity,

        -- Specific weather impacts
        CASE WHEN max_wind >= 15 THEN 1 ELSE 0 END AS high_wind,
        CASE WHEN max_temp <= 32 THEN 1 ELSE 0 END AS cold_weather,
        CASE WHEN max_temp >= 85 THEN 1 ELSE 0 END AS hot_weather

    FROM weather_conditions_raw
),

plays_with_weather AS (
    SELECT
        p.posteam,
        p.season,
        p.game_id,
        p.play_type,
        p.down,
        p.yardline_100,
        p.epa,
        p.is_successful_play,
        p.is_explosive_play,
        p.pass_attempt,
        p.rush_attempt,
        w.adverse_weather,
        w.weather_severity,
        w.high_wind,
        w.cold_weather,
        w.hot_weather

    FROM {{ ref('int_plays_cleaned') }} p
    INNER JOIN weather_conditions w
        ON p.game_id = w.game_id
    WHERE p.posteam IS NOT NULL
        AND (p.pass_attempt = 1 OR p.rush_attempt = 1)  -- Only offensive plays
),

weather_impact_by_team AS (
    SELECT
        posteam AS team,
        season,
        adverse_weather,
        weather_severity,

        -- EPA metrics by weather condition
        AVG(epa) AS epa_per_play,
        AVG(CAST(is_successful_play AS FLOAT)) AS success_rate,
        AVG(CAST(is_explosive_play AS FLOAT)) AS explosive_rate,

        -- Play type performance in weather
        AVG(CASE WHEN pass_attempt = 1 THEN epa END) AS pass_epa,
        AVG(CASE WHEN rush_attempt = 1 THEN epa END) AS run_epa,

        -- Passing specific weather impacts
        AVG(CASE WHEN pass_attempt = 1 THEN CAST(is_successful_play AS FLOAT) END) AS pass_success_rate,
        SUM(pass_attempt) AS pass_attempts,
        SUM(rush_attempt) AS run_attempts,

        -- Weather-specific situational performance
        AVG(CASE WHEN yardline_100 <= 20 THEN epa END) AS redzone_epa,
        AVG(CASE WHEN down >= 3 THEN epa END) AS third_down_epa,

        COUNT(*) AS total_plays,
        COUNT(DISTINCT game_id) AS games_in_condition

    FROM plays_with_weather
    WHERE epa IS NOT NULL
    GROUP BY posteam, season, adverse_weather, weather_severity
),

weather_impact_comparison AS (
    SELECT 
        team,
        season,
        
        -- Good weather performance (baseline)
        MAX(CASE WHEN adverse_weather = 0 THEN epa_per_play END) AS epa_good_weather,
        MAX(CASE WHEN adverse_weather = 0 THEN success_rate END) AS success_rate_good_weather,
        MAX(CASE WHEN adverse_weather = 0 THEN pass_epa END) AS pass_epa_good_weather,
        MAX(CASE WHEN adverse_weather = 0 THEN run_epa END) AS run_epa_good_weather,
        
        -- Adverse weather performance
        MAX(CASE WHEN adverse_weather = 1 THEN epa_per_play END) AS epa_adverse_weather,
        MAX(CASE WHEN adverse_weather = 1 THEN success_rate END) AS success_rate_adverse_weather,
        MAX(CASE WHEN adverse_weather = 1 THEN pass_epa END) AS pass_epa_adverse_weather,
        MAX(CASE WHEN adverse_weather = 1 THEN run_epa END) AS run_epa_adverse_weather,
        
        -- Game counts for sample size validation
        MAX(CASE WHEN adverse_weather = 0 THEN games_in_condition END) AS games_good_weather,
        MAX(CASE WHEN adverse_weather = 1 THEN games_in_condition END) AS games_adverse_weather,
        
        -- Play counts
        SUM(CASE WHEN adverse_weather = 0 THEN total_plays END) AS plays_good_weather,
        SUM(CASE WHEN adverse_weather = 1 THEN total_plays END) AS plays_adverse_weather
        
    FROM weather_impact_by_team
    GROUP BY team, season
),

weather_adjustments AS (
    SELECT 
        *,
        
        -- Calculate weather impact differentials
        (epa_good_weather - epa_adverse_weather) AS epa_weather_impact,
        (success_rate_good_weather - success_rate_adverse_weather) AS success_weather_impact,
        (pass_epa_good_weather - pass_epa_adverse_weather) AS pass_weather_impact,
        (run_epa_good_weather - run_epa_adverse_weather) AS run_weather_impact,
        
        -- Weather adjustment factors (for adjusting future performance)
        CASE 
            WHEN games_adverse_weather >= 2 THEN -- Minimum sample size
                epa_good_weather - epa_adverse_weather
            ELSE 0  -- Use league average if insufficient data
        END AS epa_weather_adjustment,
        
        -- Team weather resistance rating
        CASE 
            WHEN games_adverse_weather < 2 THEN 'Insufficient_Data'
            WHEN (epa_good_weather - epa_adverse_weather) <= 0.05 THEN 'Weather_Resistant'
            WHEN (epa_good_weather - epa_adverse_weather) <= 0.15 THEN 'Moderate_Impact'
            ELSE 'Weather_Vulnerable'
        END AS weather_resistance_tier
        
    FROM weather_impact_comparison
),

-- Calculate league-wide weather impacts for teams with insufficient data
league_weather_averages AS (
    SELECT 
        season,
        AVG(epa_weather_impact) AS league_avg_weather_impact,
        AVG(pass_weather_impact) AS league_avg_pass_weather_impact,
        AVG(run_weather_impact) AS league_avg_run_weather_impact,
        STDDEV(epa_weather_impact) AS league_std_weather_impact
    FROM weather_adjustments
    WHERE games_adverse_weather >= 2
    GROUP BY season
)

SELECT 
    w.*,
    l.league_avg_weather_impact,
    l.league_avg_pass_weather_impact,  
    l.league_avg_run_weather_impact,
    
    -- Final weather adjustment (use team-specific if available, else league average)
    COALESCE(
        CASE WHEN w.games_adverse_weather >= 2 THEN w.epa_weather_adjustment END,
        l.league_avg_weather_impact
    ) AS final_weather_adjustment,
    
    -- Weather-adjusted EPA projections
    COALESCE(w.epa_good_weather, w.epa_adverse_weather + l.league_avg_weather_impact) AS weather_neutral_epa,
    
    -- Confidence in weather adjustment
    CASE 
        WHEN w.games_adverse_weather >= 4 THEN 'High'
        WHEN w.games_adverse_weather >= 2 THEN 'Medium'
        ELSE 'Low'
    END AS weather_adjustment_confidence
    
FROM weather_adjustments w
LEFT JOIN league_weather_averages l
    ON w.season = l.season