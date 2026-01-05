-- models/intermediate/int_situational_efficiency.sql
-- Advanced situational efficiency metrics for predictive modeling

{{ config(
    materialized='table',
    indexes=[
        {'columns': ['team', 'season', 'week']},
        {'columns': ['situation_type', 'season']}
    ]
) }}

WITH situational_plays AS (
    SELECT 
        *,
        
        -- Define situational contexts
        CASE 
            WHEN yardline_100 <= 20 THEN 'red_zone'
            WHEN yardline_100 <= 10 THEN 'goal_line'  
            WHEN yardline_100 >= 80 THEN 'own_redzone'
            ELSE 'field'
        END AS part_of_field_zone,
        
        CASE 
            WHEN down = 3 THEN 'third_down'
            WHEN down = 4 THEN 'fourth_down'
            WHEN down <= 2 THEN 'early_down'
        END AS internal_down_situation,
        
        CASE 
            WHEN ydstogo = 1 THEN 'short_yardage'
            WHEN ydstogo BETWEEN 2 AND 7 THEN 'medium_yardage' 
            WHEN ydstogo BETWEEN 8 AND 15 THEN 'long_yardage'
            WHEN ydstogo > 15 THEN 'very_long'
        END AS distance_situation,
        
        CASE 
            WHEN ABS(score_differential) <= 3 THEN 'one_score'
            WHEN ABS(score_differential) <= 7 THEN 'close_game'
            WHEN ABS(score_differential) <= 14 THEN 'moderate_game'
            ELSE 'blowout'
        END AS score_situation,
        
        CASE 
            WHEN half_seconds_remaining <= 120 THEN 'two_minute'
            WHEN half_seconds_remaining <= 300 THEN 'five_minute'
            ELSE 'normal_time'
        END AS time_situation
        
    FROM {{ ref('int_plays_cleaned') }}
    WHERE epa IS NOT NULL
),

-- Red Zone Efficiency
red_zone_metrics AS (
    SELECT 
        posteam AS team,
        season,
        week,
        'red_zone' AS situation_type,
        
        COUNT(*) AS plays,
        AVG(epa) AS epa_per_play,
        AVG(is_successful_play::float) AS success_rate,
        SUM(touchdown)::float / NULLIF(COUNT(DISTINCT drive), 0) AS td_rate,
        SUM(touchdown)::float / NULLIF(COUNT(DISTINCT drive), 0) AS conversion_rate
        
    FROM situational_plays 
    WHERE part_of_field_zone = 'red_zone'
    GROUP BY posteam, season, week
),

-- Third Down Efficiency  
third_down_metrics AS (
    SELECT 
        posteam AS team,
        season,
        week,
        'third_down' AS situation_type,
        
        COUNT(*) AS plays,
        AVG(epa) AS epa_per_play,
        AVG(is_successful_play::float) AS success_rate,
        SUM(touchdown)::float / NULLIF(COUNT(DISTINCT drive), 0) AS td_rate,
        AVG(first_down::float) AS conversion_rate
        
    FROM situational_plays
    WHERE internal_down_situation = 'third_down'
    GROUP BY posteam, season, week
),

-- Two Minute Drill Efficiency
two_minute_metrics AS (
    SELECT 
        posteam AS team,
        season, 
        week,
        'two_minute' AS situation_type,
        
        COUNT(*) AS plays,
        AVG(epa) AS epa_per_play,
        AVG(is_successful_play::float) AS success_rate,
        SUM(touchdown)::float / NULLIF(COUNT(DISTINCT drive), 0) AS td_rate,
        AVG(first_down::float) AS conversion_rate
        
    FROM situational_plays
    WHERE time_situation = 'two_minute'
    GROUP BY posteam, season, week
),

-- Short Yardage Efficiency
short_yardage_metrics AS (
    SELECT 
        posteam AS team,
        season,
        week,
        'short_yardage' AS situation_type,
        
        COUNT(*) AS plays,
        AVG(epa) AS epa_per_play,
        AVG(is_successful_play::float) AS success_rate,
        SUM(touchdown)::float / NULLIF(COUNT(DISTINCT drive), 0) AS td_rate,
        AVG(first_down::float) AS conversion_rate
        
    FROM situational_plays
    WHERE ydstogo = 1 
    GROUP BY posteam, season, week
),

-- Pivot situational metrics to wide format
team_situational_summary AS (
    SELECT 
        team,
        season,
        week,
        
        -- Red Zone Metrics
        MAX(CASE WHEN situation_type = 'red_zone' THEN plays END) AS red_zone_plays,
        MAX(CASE WHEN situation_type = 'red_zone' THEN td_rate END) AS red_zone_td_rate,
        MAX(CASE WHEN situation_type = 'red_zone' THEN epa_per_play END) AS red_zone_epa,
        MAX(CASE WHEN situation_type = 'red_zone' THEN success_rate END) AS red_zone_success_rate,
        
        -- Third Down Metrics  
        MAX(CASE WHEN situation_type = 'third_down' THEN plays END) AS third_down_attempts,
        MAX(CASE WHEN situation_type = 'third_down' THEN conversion_rate END) AS third_down_conversion_rate,
        MAX(CASE WHEN situation_type = 'third_down' THEN plays * conversion_rate END) AS third_down_conversions,
        MAX(CASE WHEN situation_type = 'third_down' THEN epa_per_play END) AS third_down_epa,
        
        -- Two Minute Drill Metrics
        MAX(CASE WHEN situation_type = 'two_minute' THEN plays END) AS two_minute_drill_plays,
        MAX(CASE WHEN situation_type = 'two_minute' THEN epa_per_play END) AS two_minute_drill_epa,
        MAX(CASE WHEN situation_type = 'two_minute' THEN success_rate END) AS two_minute_success_rate,
        
        -- Short Yardage Metrics
        MAX(CASE WHEN situation_type = 'short_yardage' THEN plays END) AS short_yardage_plays,
        MAX(CASE WHEN situation_type = 'short_yardage' THEN success_rate END) AS short_yardage_success_rate,
        MAX(CASE WHEN situation_type = 'short_yardage' THEN conversion_rate END) AS short_yardage_conversion_rate
        
    FROM (
        SELECT * FROM red_zone_metrics
        UNION ALL
        SELECT * FROM third_down_metrics  
        UNION ALL
        SELECT * FROM two_minute_metrics
        UNION ALL
        SELECT * FROM short_yardage_metrics
    ) combined
    GROUP BY team, season, week
)

SELECT 
    team,
    season,
    week,
    
    -- Red Zone Metrics
    COALESCE(red_zone_plays, 0) AS red_zone_plays,
    COALESCE(red_zone_td_rate, 0) AS red_zone_td_rate,
    COALESCE(red_zone_epa, 0) AS red_zone_epa,
    COALESCE(red_zone_success_rate, 0) AS red_zone_success_rate,
    
    -- Third Down Metrics
    COALESCE(third_down_attempts, 0) AS third_down_attempts,
    COALESCE(third_down_conversions, 0) AS third_down_conversions,
    COALESCE(third_down_conversion_rate, 0) AS third_down_conversion_rate,
    COALESCE(third_down_epa, 0) AS third_down_epa,
    
    -- Two Minute Drill Metrics
    COALESCE(two_minute_drill_plays, 0) AS two_minute_drill_plays,
    COALESCE(two_minute_drill_epa, 0) AS two_minute_drill_epa,
    COALESCE(two_minute_success_rate, 0) AS two_minute_success_rate,
    
    -- Short Yardage Metrics
    COALESCE(short_yardage_plays, 0) AS short_yardage_plays,
    COALESCE(short_yardage_success_rate, 0) AS short_yardage_success_rate,
    COALESCE(short_yardage_conversion_rate, 0) AS short_yardage_conversion_rate,
    
    -- Rolling 4-week averages
    AVG(red_zone_td_rate) OVER (
        PARTITION BY team, season 
        ORDER BY week 
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS rolling_4wk_red_zone_td_rate,
    
    AVG(third_down_conversion_rate) OVER (
        PARTITION BY team, season
        ORDER BY week
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW  
    ) AS rolling_4wk_third_down_rate,
    
    AVG(two_minute_drill_epa) OVER (
        PARTITION BY team, season
        ORDER BY week
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW  
    ) AS rolling_4wk_two_min_epa,
    
    AVG(short_yardage_success_rate) OVER (
        PARTITION BY team, season
        ORDER BY week
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW  
    ) AS rolling_4wk_short_yardage_rate
    
FROM team_situational_summary