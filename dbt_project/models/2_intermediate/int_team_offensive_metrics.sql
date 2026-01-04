-- models/2_intermediate/int_team_offensive_metrics.sql
-- Calculate rolling offensive performance metrics by team
-- Grain: One row per team per season per week

{{ config(
    materialized='table',
    tags=['intermediate', 'team_level', 'offensive']
) }}

WITH team_weekly_plays AS (
    -- Aggregate play-level metrics to team-week level
    SELECT
        posteam AS team,
        season,
        week,
        season_type,

        -- Play counts by type
        COUNT(*) AS total_plays,
        SUM(pass_attempt) AS pass_attempts,
        SUM(rush_attempt) AS rush_attempts,

        -- EPA metrics
        AVG(epa) AS avg_epa_per_play,
        AVG(CASE WHEN pass_attempt = 1 THEN epa END) AS avg_pass_epa,
        AVG(CASE WHEN rush_attempt = 1 THEN epa END) AS avg_rush_epa,
        SUM(epa) AS total_epa,

        -- Success rate (% of plays with positive EPA)
        AVG(CASE WHEN epa > 0 THEN 1.0 ELSE 0.0 END) AS success_rate,
        AVG(CASE WHEN pass_attempt = 1 AND epa > 0 THEN 1.0 ELSE 0.0 END) AS pass_success_rate,
        AVG(CASE WHEN rush_attempt = 1 AND epa > 0 THEN 1.0 ELSE 0.0 END) AS rush_success_rate,

        -- Explosive play rate
        AVG(CASE WHEN is_explosive_play = 1 THEN 1.0 ELSE 0.0 END) AS explosive_play_rate,
        AVG(CASE WHEN pass_attempt = 1 AND yards_gained >= 15 THEN 1.0 ELSE 0.0 END) AS explosive_pass_rate,
        AVG(CASE WHEN rush_attempt = 1 AND yards_gained >= 10 THEN 1.0 ELSE 0.0 END) AS explosive_rush_rate,

        -- Yards
        AVG(yards_gained) AS avg_yards_per_play,
        SUM(yards_gained) AS total_yards,
        AVG(CASE WHEN pass_attempt = 1 THEN yards_gained END) AS avg_yards_per_pass,
        AVG(CASE WHEN rush_attempt = 1 THEN yards_gained END) AS avg_yards_per_rush,

        -- Third down efficiency
        SUM(CASE WHEN down = 3 THEN 1 ELSE 0 END) AS third_down_attempts,
        SUM(third_down_converted) AS third_down_conversions,
        AVG(CASE WHEN down = 3 THEN third_down_converted ELSE NULL END) AS third_down_conversion_rate,

        -- Turnovers
        SUM(is_turnover) AS turnovers,
        AVG(CASE WHEN pass_attempt = 1 OR rush_attempt = 1 THEN is_turnover ELSE NULL END) AS turnover_rate,

        -- Touchdowns
        SUM(touchdown) AS touchdowns,
        SUM(pass_touchdown) AS pass_touchdowns,
        SUM(rush_touchdown) AS rush_touchdowns

    FROM {{ ref('int_plays_cleaned') }}
    WHERE
        posteam IS NOT NULL
        AND season_type = 'REG'  -- Focus on regular season for consistency
        AND (pass_attempt = 1 OR rush_attempt = 1)  -- Only offensive plays
    GROUP BY 1, 2, 3, 4
),

red_zone_efficiency AS (
    -- Calculate red zone touchdown rate separately
    SELECT
        posteam AS team,
        season,
        week,

        SUM(CASE WHEN yardline_100 <= 20 THEN 1 ELSE 0 END) AS red_zone_plays,
        SUM(CASE WHEN yardline_100 <= 20 AND touchdown = 1 THEN 1 ELSE 0 END) AS red_zone_touchdowns,
        AVG(CASE WHEN yardline_100 <= 20 THEN touchdown ELSE NULL END) AS red_zone_td_rate,
        AVG(CASE WHEN yardline_100 <= 10 THEN touchdown ELSE NULL END) AS goal_line_td_rate

    FROM {{ ref('int_plays_cleaned') }}
    WHERE
        posteam IS NOT NULL
        AND season_type = 'REG'
        AND (pass_attempt = 1 OR rush_attempt = 1)
    GROUP BY 1, 2, 3
),

drive_metrics AS (
    -- Calculate drive-level efficiency metrics
    SELECT
        posteam AS team,
        season,
        week,

        COUNT(DISTINCT drive) AS total_drives,
        AVG(CASE WHEN drive_ended_with_score = 1 THEN 1.0 ELSE 0.0 END) AS drive_scoring_rate,

        -- Points and yards per drive (approximate from play data)
        SUM(touchdown * 6 + field_goal_attempt * 3) / NULLIF(COUNT(DISTINCT drive), 0) AS approx_points_per_drive,
        SUM(yards_gained) / NULLIF(COUNT(DISTINCT drive), 0) AS avg_yards_per_drive

    FROM {{ ref('int_plays_cleaned') }}
    WHERE
        posteam IS NOT NULL
        AND season_type = 'REG'
        AND drive IS NOT NULL
    GROUP BY 1, 2, 3
),

combined_metrics AS (
    -- Join all metrics together
    SELECT
        p.*,
        r.red_zone_plays,
        r.red_zone_touchdowns,
        r.red_zone_td_rate,
        r.goal_line_td_rate,
        d.total_drives,
        d.drive_scoring_rate,
        d.approx_points_per_drive,
        d.avg_yards_per_drive

    FROM team_weekly_plays p
    LEFT JOIN red_zone_efficiency r
        ON p.team = r.team
        AND p.season = r.season
        AND p.week = r.week
    LEFT JOIN drive_metrics d
        ON p.team = d.team
        AND p.season = d.season
        AND p.week = d.week
),

rolling_metrics AS (
    -- Calculate rolling 4-week averages for key metrics
    SELECT
        *,

        -- Rolling EPA metrics (4-week window)
        AVG(avg_epa_per_play) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_epa_per_play,

        AVG(avg_pass_epa) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_pass_epa,

        AVG(avg_rush_epa) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_rush_epa,

        -- Rolling success rates (4-week window)
        AVG(success_rate) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_success_rate,

        AVG(pass_success_rate) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_pass_success_rate,

        AVG(rush_success_rate) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_rush_success_rate,

        -- Rolling explosive play rate (4-week window)
        AVG(explosive_play_rate) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_explosive_rate,

        -- Rolling third down efficiency (4-week window)
        AVG(third_down_conversion_rate) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_third_down_rate,

        -- Rolling red zone efficiency (4-week window)
        AVG(red_zone_td_rate) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_red_zone_td_rate,

        -- Rolling turnover rate (4-week window)
        AVG(turnover_rate) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_turnover_rate,

        -- Rolling yards per play (4-week window)
        AVG(avg_yards_per_play) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_yards_per_play,

        -- Rolling points per drive (4-week window)
        AVG(approx_points_per_drive) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_points_per_drive,

        -- Sample size quality indicator
        SUM(total_plays) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_play_count

    FROM combined_metrics
)

SELECT
    -- Identifiers
    team,
    season,
    week,
    season_type,

    -- Raw weekly metrics
    total_plays,
    pass_attempts,
    rush_attempts,
    avg_epa_per_play,
    avg_pass_epa,
    avg_rush_epa,
    total_epa,
    success_rate,
    pass_success_rate,
    rush_success_rate,
    explosive_play_rate,
    explosive_pass_rate,
    explosive_rush_rate,
    avg_yards_per_play,
    total_yards,
    avg_yards_per_pass,
    avg_yards_per_rush,
    third_down_attempts,
    third_down_conversions,
    third_down_conversion_rate,
    turnovers,
    turnover_rate,
    touchdowns,
    pass_touchdowns,
    rush_touchdowns,
    red_zone_plays,
    red_zone_touchdowns,
    red_zone_td_rate,
    goal_line_td_rate,
    total_drives,
    drive_scoring_rate,
    approx_points_per_drive,
    avg_yards_per_drive,

    -- Rolling 4-week metrics (key features for predictions)
    rolling_4wk_epa_per_play,
    rolling_4wk_pass_epa,
    rolling_4wk_rush_epa,
    rolling_4wk_success_rate,
    rolling_4wk_pass_success_rate,
    rolling_4wk_rush_success_rate,
    rolling_4wk_explosive_rate,
    rolling_4wk_third_down_rate,
    rolling_4wk_red_zone_td_rate,
    rolling_4wk_turnover_rate,
    rolling_4wk_yards_per_play,
    rolling_4wk_points_per_drive,
    rolling_4wk_play_count

FROM rolling_metrics
ORDER BY team, season, week
