-- models/2_intermediate/int_team_defensive_strength.sql
-- Calculate rolling defensive performance metrics by team
-- Grain: One row per team per season per week
-- Note: Lower EPA allowed is better defense, higher success rate allowed is worse defense

{{ config(
    materialized='table',
    tags=['intermediate', 'team_level', 'defensive']
) }}

WITH team_weekly_defensive_plays AS (
    -- Aggregate play-level metrics to team-week level for defensive performance
    SELECT
        defteam AS team,
        season,
        week,
        season_type,

        -- Play counts faced
        COUNT(*) AS total_plays_faced,
        SUM(pass_attempt) AS pass_attempts_faced,
        SUM(rush_attempt) AS rush_attempts_faced,

        -- EPA allowed (lower is better for defense)
        AVG(epa) AS avg_epa_allowed,
        AVG(CASE WHEN pass_attempt = 1 THEN epa END) AS avg_pass_epa_allowed,
        AVG(CASE WHEN rush_attempt = 1 THEN epa END) AS avg_rush_epa_allowed,
        SUM(epa) AS total_epa_allowed,

        -- Success rate allowed (lower is better for defense)
        AVG(CASE WHEN epa > 0 THEN 1.0 ELSE 0.0 END) AS success_rate_allowed,
        AVG(CASE WHEN pass_attempt = 1 AND epa > 0 THEN 1.0 ELSE 0.0 END) AS pass_success_rate_allowed,
        AVG(CASE WHEN rush_attempt = 1 AND epa > 0 THEN 1.0 ELSE 0.0 END) AS rush_success_rate_allowed,

        -- Explosive play rate allowed (lower is better for defense)
        AVG(CASE WHEN is_explosive_play = 1 THEN 1.0 ELSE 0.0 END) AS explosive_play_rate_allowed,
        AVG(CASE WHEN pass_attempt = 1 AND yards_gained >= 15 THEN 1.0 ELSE 0.0 END) AS explosive_pass_rate_allowed,
        AVG(CASE WHEN rush_attempt = 1 AND yards_gained >= 10 THEN 1.0 ELSE 0.0 END) AS explosive_rush_rate_allowed,

        -- Yards allowed (lower is better for defense)
        AVG(yards_gained) AS avg_yards_allowed_per_play,
        SUM(yards_gained) AS total_yards_allowed,
        AVG(CASE WHEN pass_attempt = 1 THEN yards_gained END) AS avg_yards_allowed_per_pass,
        AVG(CASE WHEN rush_attempt = 1 THEN yards_gained END) AS avg_yards_allowed_per_rush,

        -- Third down defense (lower conversion rate is better)
        SUM(CASE WHEN down = 3 THEN 1 ELSE 0 END) AS third_down_attempts_faced,
        SUM(third_down_converted) AS third_down_conversions_allowed,
        AVG(CASE WHEN down = 3 THEN third_down_converted ELSE NULL END) AS third_down_conversion_rate_allowed,

        -- Forced turnovers (higher is better for defense)
        SUM(is_turnover) AS forced_turnovers,
        AVG(CASE WHEN pass_attempt = 1 OR rush_attempt = 1 THEN is_turnover ELSE NULL END) AS forced_turnover_rate,

        -- Sacks and QB pressure (higher is better for defense)
        SUM(sack) AS sacks,
        AVG(CASE WHEN pass_attempt = 1 OR sack = 1 THEN sack ELSE NULL END) AS sack_rate,

        -- Touchdowns allowed (lower is better)
        SUM(touchdown) AS touchdowns_allowed,
        SUM(pass_touchdown) AS pass_touchdowns_allowed,
        SUM(rush_touchdown) AS rush_touchdowns_allowed

    FROM {{ ref('int_plays_cleaned') }}
    WHERE
        defteam IS NOT NULL
        AND season_type = 'REG'  -- Focus on regular season
        AND (pass_attempt = 1 OR rush_attempt = 1)  -- Only offensive plays
    GROUP BY 1, 2, 3, 4
),

red_zone_defense AS (
    -- Calculate red zone touchdown rate allowed
    SELECT
        defteam AS team,
        season,
        week,

        SUM(CASE WHEN yardline_100 <= 20 THEN 1 ELSE 0 END) AS red_zone_plays_faced,
        SUM(CASE WHEN yardline_100 <= 20 AND touchdown = 1 THEN 1 ELSE 0 END) AS red_zone_touchdowns_allowed,
        AVG(CASE WHEN yardline_100 <= 20 THEN touchdown ELSE NULL END) AS red_zone_td_rate_allowed,
        AVG(CASE WHEN yardline_100 <= 10 THEN touchdown ELSE NULL END) AS goal_line_td_rate_allowed

    FROM {{ ref('int_plays_cleaned') }}
    WHERE
        defteam IS NOT NULL
        AND season_type = 'REG'
        AND (pass_attempt = 1 OR rush_attempt = 1)
    GROUP BY 1, 2, 3
),

drive_defense_metrics AS (
    -- Calculate drive-level defensive metrics
    SELECT
        defteam AS team,
        season,
        week,

        COUNT(DISTINCT drive) AS total_drives_faced,
        AVG(CASE WHEN drive_ended_with_score = 1 THEN 1.0 ELSE 0.0 END) AS drive_scoring_rate_allowed,

        -- Points and yards allowed per drive (approximate)
        SUM(touchdown * 6 + field_goal_attempt * 3) / NULLIF(COUNT(DISTINCT drive), 0) AS approx_points_allowed_per_drive,
        SUM(yards_gained) / NULLIF(COUNT(DISTINCT drive), 0) AS avg_yards_allowed_per_drive

    FROM {{ ref('int_plays_cleaned') }}
    WHERE
        defteam IS NOT NULL
        AND season_type = 'REG'
        AND drive IS NOT NULL
    GROUP BY 1, 2, 3
),

combined_defensive_metrics AS (
    -- Join all defensive metrics together
    SELECT
        p.*,
        r.red_zone_plays_faced,
        r.red_zone_touchdowns_allowed,
        r.red_zone_td_rate_allowed,
        r.goal_line_td_rate_allowed,
        d.total_drives_faced,
        d.drive_scoring_rate_allowed,
        d.approx_points_allowed_per_drive,
        d.avg_yards_allowed_per_drive

    FROM team_weekly_defensive_plays p
    LEFT JOIN red_zone_defense r
        ON p.team = r.team
        AND p.season = r.season
        AND p.week = r.week
    LEFT JOIN drive_defense_metrics d
        ON p.team = d.team
        AND p.season = d.season
        AND p.week = d.week
),

rolling_defensive_metrics AS (
    -- Calculate rolling 4-week averages for defensive metrics
    SELECT
        *,

        -- Rolling EPA allowed (4-week window) - lower is better
        AVG(avg_epa_allowed) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_epa_allowed,

        AVG(avg_pass_epa_allowed) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_pass_epa_allowed,

        AVG(avg_rush_epa_allowed) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_rush_epa_allowed,

        -- Rolling success rates allowed (4-week window) - lower is better
        AVG(success_rate_allowed) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_success_rate_allowed,

        AVG(pass_success_rate_allowed) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_pass_success_allowed,

        AVG(rush_success_rate_allowed) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_rush_success_allowed,

        -- Rolling explosive play rate allowed (4-week window) - lower is better
        AVG(explosive_play_rate_allowed) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_explosive_rate_allowed,

        -- Rolling third down defense (4-week window) - lower is better
        AVG(third_down_conversion_rate_allowed) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_third_down_allowed,

        -- Rolling red zone defense (4-week window) - lower is better
        AVG(red_zone_td_rate_allowed) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_red_zone_td_allowed,

        -- Rolling forced turnover rate (4-week window) - higher is better
        AVG(forced_turnover_rate) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_forced_turnover_rate,

        -- Rolling sack rate (4-week window) - higher is better
        AVG(sack_rate) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_sack_rate,

        -- Rolling yards allowed per play (4-week window) - lower is better
        AVG(avg_yards_allowed_per_play) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_yards_allowed_per_play,

        -- Rolling points allowed per drive (4-week window) - lower is better
        AVG(approx_points_allowed_per_drive) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_points_allowed_per_drive,

        -- Sample size quality indicator
        SUM(total_plays_faced) OVER (
            PARTITION BY team
            ORDER BY season, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4wk_defensive_play_count

    FROM combined_defensive_metrics
),

final_with_rankings AS (
    -- Add defensive rankings within each season-week
    SELECT
        *,

        -- Defensive rankings (1 = best defense)
        RANK() OVER (
            PARTITION BY season, week
            ORDER BY rolling_4wk_epa_allowed ASC  -- Lower EPA allowed = better rank
        ) AS defensive_epa_rank,

        RANK() OVER (
            PARTITION BY season, week
            ORDER BY rolling_4wk_yards_allowed_per_play ASC
        ) AS yards_allowed_rank,

        RANK() OVER (
            PARTITION BY season, week
            ORDER BY rolling_4wk_points_allowed_per_drive ASC
        ) AS points_allowed_rank,

        RANK() OVER (
            PARTITION BY season, week
            ORDER BY rolling_4wk_forced_turnover_rate DESC  -- Higher turnover rate = better rank
        ) AS forced_turnover_rank

    FROM rolling_defensive_metrics
)

SELECT
    -- Identifiers
    team,
    season,
    week,
    season_type,

    -- Raw weekly metrics
    total_plays_faced,
    pass_attempts_faced,
    rush_attempts_faced,
    avg_epa_allowed,
    avg_pass_epa_allowed,
    avg_rush_epa_allowed,
    total_epa_allowed,
    success_rate_allowed,
    pass_success_rate_allowed,
    rush_success_rate_allowed,
    explosive_play_rate_allowed,
    explosive_pass_rate_allowed,
    explosive_rush_rate_allowed,
    avg_yards_allowed_per_play,
    total_yards_allowed,
    avg_yards_allowed_per_pass,
    avg_yards_allowed_per_rush,
    third_down_attempts_faced,
    third_down_conversions_allowed,
    third_down_conversion_rate_allowed,
    forced_turnovers,
    forced_turnover_rate,
    sacks,
    sack_rate,
    touchdowns_allowed,
    pass_touchdowns_allowed,
    rush_touchdowns_allowed,
    red_zone_plays_faced,
    red_zone_touchdowns_allowed,
    red_zone_td_rate_allowed,
    goal_line_td_rate_allowed,
    total_drives_faced,
    drive_scoring_rate_allowed,
    approx_points_allowed_per_drive,
    avg_yards_allowed_per_drive,

    -- Rolling 4-week metrics (key features for predictions)
    rolling_4wk_epa_allowed,
    rolling_4wk_pass_epa_allowed,
    rolling_4wk_rush_epa_allowed,
    rolling_4wk_success_rate_allowed,
    rolling_4wk_pass_success_allowed,
    rolling_4wk_rush_success_allowed,
    rolling_4wk_explosive_rate_allowed,
    rolling_4wk_third_down_allowed,
    rolling_4wk_red_zone_td_allowed,
    rolling_4wk_forced_turnover_rate,
    rolling_4wk_sack_rate,
    rolling_4wk_yards_allowed_per_play,
    rolling_4wk_points_allowed_per_drive,
    rolling_4wk_defensive_play_count,

    -- Rankings (1 = best defense in that week)
    defensive_epa_rank,
    yards_allowed_rank,
    points_allowed_rank,
    forced_turnover_rank

FROM final_with_rankings
ORDER BY team, season, week
