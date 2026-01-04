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
        END AS field_zone,
        
        CASE 
            WHEN down = 3 THEN 'third_down'
            WHEN down = 4 THEN 'fourth_down'
            WHEN down <= 2 THEN 'early_down'
        END AS down_situation,
        
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
        
    FROM {{ ref('stgnv_play_by_play') }}
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
        AVG(success::float) AS success_rate,
        SUM(touchdown) AS touchdowns,
        COUNT(DISTINCT drive) AS drives,
        SUM(touchdown)::float / NULLIF(COUNT(DISTINCT drive), 0) AS td_rate,
        
        -- Goal line specific (inside 5)
        AVG(CASE WHEN yardline_100 <= 5 THEN epa END) AS goal_line_epa,
        AVG(CASE WHEN yardline_100 <= 5 THEN success::float END) AS goal_line_success,
        
        -- Red zone by play type
        AVG(CASE WHEN play_type = 'pass' THEN epa END) AS rz_pass_epa,
        AVG(CASE WHEN play_type = 'run' THEN epa END) AS rz_run_epa,
        COUNT(CASE WHEN play_type = 'pass' THEN 1 END)::float / 
            NULLIF(COUNT(*), 0) AS rz_pass_rate
        
    FROM situational_plays 
    WHERE field_zone = 'red_zone'
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
        AVG(success::float) AS success_rate,
        AVG(first_down::float) AS conversion_rate,
        
        -- Third down by distance
        AVG(CASE WHEN ydstogo = 1 THEN first_down::float END) AS third_and_1_rate,
        AVG(CASE WHEN ydstogo BETWEEN 2 AND 3 THEN first_down::float END) AS third_and_short_rate,
        AVG(CASE WHEN ydstogo BETWEEN 4 AND 6 THEN first_down::float END) AS third_and_medium_rate,
        AVG(CASE WHEN ydstogo >= 7 THEN first_down::float END) AS third_and_long_rate,
        
        -- Third down EPA by distance
        AVG(CASE WHEN ydstogo <= 3 THEN epa END) AS third_short_epa,
        AVG(CASE WHEN ydstogo BETWEEN 4 AND 6 THEN epa END) AS third_medium_epa,
        AVG(CASE WHEN ydstogo >= 7 THEN epa END) AS third_long_epa
        
    FROM situational_plays
    WHERE down_situation = 'third_down'
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
        AVG(success::float) AS success_rate,
        AVG(explosive_play::float) AS explosive_rate,
        
        -- Two minute by half
        AVG(CASE WHEN qtr = 2 THEN epa END) AS first_half_2min_epa,
        AVG(CASE WHEN qtr = 4 THEN epa END) AS second_half_2min_epa,
        
        -- Pressure situations (trailing in 2-minute drill)
        AVG(CASE WHEN score_differential < 0 THEN epa END) AS trailing_2min_epa,
        AVG(CASE WHEN score_differential > 0 THEN epa END) AS leading_2min_epa
        
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
        AVG(success::float) AS success_rate,
        AVG(first_down::float) AS conversion_rate,
        
        -- Short yardage by play type
        AVG(CASE WHEN play_type = 'run' THEN success::float END) AS short_yardage_run_rate,
        AVG(CASE WHEN play_type = 'pass' THEN success::float END) AS short_yardage_pass_rate,
        
        COUNT(CASE WHEN play_type = 'run' THEN 1 END)::float / 
            NULLIF(COUNT(*), 0) AS short_yardage_run_frequency
        
    FROM situational_plays
    WHERE ydstogo = 1 
    GROUP BY posteam, season, week
),

-- Combine all situational metrics
combined_situational AS (
    SELECT * FROM red_zone_metrics
    UNION ALL
    SELECT * FROM third_down_metrics  
    UNION ALL
    SELECT * FROM two_minute_metrics
    UNION ALL
    SELECT * FROM short_yardage_metrics
)

SELECT 
    *,
    
    -- Rolling averages for situational metrics
    AVG(epa_per_play) OVER (
        PARTITION BY team, season, situation_type 
        ORDER BY week 
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS epa_l4w,
    
    AVG(success_rate) OVER (
        PARTITION BY team, season, situation_type
        ORDER BY week
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW  
    ) AS success_rate_l4w,
    
    -- Situational rankings
    RANK() OVER (
        PARTITION BY season, week, situation_type
        ORDER BY epa_per_play DESC
    ) AS situation_epa_rank,
    
    -- Sample size indicator
    CASE 
        WHEN plays >= 10 THEN 'sufficient'
        WHEN plays >= 5 THEN 'moderate'  
        ELSE 'low'
    END AS sample_size_quality
    
FROM combined_situational