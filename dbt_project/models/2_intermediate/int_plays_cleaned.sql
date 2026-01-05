-- models/2_intermediate/int_plays_cleaned.sql
-- Clean and enhance play-by-play data with calculated fields and classifications
-- Grain: One row per play (excludes invalid plays like spikes, kneels, and timeouts)

{{ config(
    materialized='view',
    tags=['intermediate', 'play_level']
) }}

WITH base_plays AS (
    SELECT
        -- Identifiers
        play_id,
        game_id,
        game_date,
        season,
        CAST(week AS INT) AS week,
        season_type,
        home_team,
        away_team,
        posteam,
        posteam_type,
        defteam,

        -- Game situation
        CAST(down AS INT) AS down,
        CAST(ydstogo AS INT) AS ydstogo,
        CAST(yardline_100 AS INT) AS yardline_100,
        CAST(qtr AS INT) AS qtr,
        CAST(half_seconds_remaining AS INT) AS half_seconds_remaining,
        CAST(game_seconds_remaining AS INT) AS game_seconds_remaining,
        game_half,

        -- Score context
        CAST(posteam_score AS INT) AS posteam_score,
        CAST(defteam_score AS INT) AS defteam_score,
        CAST(score_differential AS INT) AS score_differential,
        CAST(posteam_score_post AS INT) AS posteam_score_post,
        CAST(defteam_score_post AS INT) AS defteam_score_post,

        -- Play details
        play_type,
        CAST(yards_gained AS INT) AS yards_gained,
        'desc' AS play_description,

        -- Play outcomes (boolean flags)
        CAST(CASE WHEN first_down_rush = '1' OR first_down_pass = '1' THEN 1 ELSE 0 END AS INT) AS first_down,
        CAST(CASE WHEN touchdown = '1' THEN 1 ELSE 0 END AS INT) AS touchdown,
        CAST(CASE WHEN pass_touchdown = '1' THEN 1 ELSE 0 END AS INT) AS pass_touchdown,
        CAST(CASE WHEN rush_touchdown = '1' THEN 1 ELSE 0 END AS INT) AS rush_touchdown,
        CAST(CASE WHEN interception = '1' THEN 1 ELSE 0 END AS INT) AS interception,
        CAST(CASE WHEN fumble_lost = '1' THEN 1 ELSE 0 END AS INT) AS fumble_lost,
        CAST(CASE WHEN fumble = '1' THEN 1 ELSE 0 END AS INT) AS fumble,
        CAST(CASE WHEN safety = '1' THEN 1 ELSE 0 END AS INT) AS safety,
        CAST(CASE WHEN sack = '1' THEN 1 ELSE 0 END AS INT) AS sack,
        CAST(CASE WHEN penalty = '1' THEN 1 ELSE 0 END AS INT) AS penalty,
        CAST(CASE WHEN complete_pass = '1' THEN 1 ELSE 0 END AS INT) AS complete_pass,
        CAST(CASE WHEN incomplete_pass = '1' THEN 1 ELSE 0 END AS INT) AS incomplete_pass,

        -- Attempt flags
        CAST(CASE WHEN pass_attempt = '1' THEN 1 ELSE 0 END AS INT) AS pass_attempt,
        CAST(CASE WHEN rush_attempt = '1' THEN 1 ELSE 0 END AS INT) AS rush_attempt,
        CAST(CASE WHEN field_goal_attempt = '1' THEN 1 ELSE 0 END AS INT) AS field_goal_attempt,
        CAST(CASE WHEN punt_attempt = '1' THEN 1 ELSE 0 END AS INT) AS punt_attempt,
        CAST(CASE WHEN two_point_attempt = '1' THEN 1 ELSE 0 END AS INT) AS two_point_attempt,
        CAST(CASE WHEN extra_point_attempt = '1' THEN 1 ELSE 0 END AS INT) AS extra_point_attempt,

        -- QB flags
        CAST(CASE WHEN qb_kneel = '1' THEN 1 ELSE 0 END AS INT) AS qb_kneel,
        CAST(CASE WHEN qb_spike = '1' THEN 1 ELSE 0 END AS INT) AS qb_spike,
        CAST(CASE WHEN qb_scramble = '1' THEN 1 ELSE 0 END AS INT) AS qb_scramble,
        CAST(CASE WHEN qb_dropback = '1' THEN 1 ELSE 0 END AS INT) AS qb_dropback,

        -- Play style
        CAST(CASE WHEN shotgun = '1' THEN 1 ELSE 0 END AS INT) AS shotgun,
        CAST(CASE WHEN no_huddle = '1' THEN 1 ELSE 0 END AS INT) AS no_huddle,
        pass_length,
        pass_location,
        run_location,
        run_gap,

        -- Air yards and YAC (for passing plays)
        CAST(air_yards AS FLOAT) AS air_yards,
        CAST(yards_after_catch AS FLOAT) AS yards_after_catch,

        -- Advanced metrics (EPA and WPA)
        CAST(ep AS FLOAT) AS ep,
        CAST(epa AS FLOAT) AS epa,
        CAST(wpa AS FLOAT) AS wpa,
        CAST(wp AS FLOAT) AS wp,

        -- Down conversion metrics
        CAST(CASE WHEN third_down_converted = '1' THEN 1 ELSE 0 END AS INT) AS third_down_converted,
        CAST(CASE WHEN third_down_failed = '1' THEN 1 ELSE 0 END AS INT) AS third_down_failed,
        CAST(CASE WHEN fourth_down_converted = '1' THEN 1 ELSE 0 END AS INT) AS fourth_down_converted,
        CAST(CASE WHEN fourth_down_failed = '1' THEN 1 ELSE 0 END AS INT) AS fourth_down_failed,

        -- Drive information
        CAST(drive AS INT) AS drive,
        CAST(drive_play_count AS INT) AS drive_play_count,
        CAST(drive_first_downs AS INT) AS drive_first_downs,
        fixed_drive_result AS drive_result,
        CAST(drive_yards_penalized AS INT) AS drive_yards_penalized,
        CAST(drive_start_yard_line AS INT) AS drive_start_yard_line,
        CAST(drive_end_yard_line AS INT) AS drive_end_yard_line,
        CAST(CASE WHEN drive_ended_with_score = '1' THEN 1 ELSE 0 END AS INT) AS drive_ended_with_score,

        -- Weather
        CAST(temp AS FLOAT) AS temp,
        CAST(wind AS FLOAT) AS wind,
        roof,
        surface,

        -- Player IDs for tracking
        passer_player_id,
        passer_player_name,
        rusher_player_id,
        rusher_player_name,
        receiver_player_id,
        receiver_player_name

    FROM {{ ref('stgnv_play_by_play') }}
    WHERE
        -- Filter out invalid/administrative plays
        CAST(CASE WHEN qb_kneel = '1' THEN 1 ELSE 0 END AS INT) = 0  -- Exclude kneels
        AND CAST(CASE WHEN qb_spike = '1' THEN 1 ELSE 0 END AS INT) = 0  -- Exclude spikes
        AND play_type IS NOT NULL
        AND play_type NOT IN ('no_play', 'timeout', 'half_end', 'quarter_end', 'game_end')
        AND season >= 2010  -- Focus on modern era with complete EPA data
),

enhanced_plays AS (
    SELECT
        *,

        -- Calculated field: EPA per play (already exists but aliased for clarity)
        epa AS epa_per_play,

        -- Calculated field: Air yards share (what % of yards came through the air vs YAC)
        CASE
            WHEN pass_attempt = 1 AND yards_gained > 0 THEN
                air_yards / NULLIF(yards_gained, 0)
            ELSE NULL
        END AS air_yards_share,

        -- Play classification: Successful play (positive EPA)
        CASE
            WHEN epa > 0 THEN 1
            ELSE 0
        END AS is_successful_play,

        -- Play classification: Explosive play (15+ yards for pass, 10+ for rush)
        CASE
            WHEN pass_attempt = 1 AND yards_gained >= 15 THEN 1
            WHEN rush_attempt = 1 AND yards_gained >= 10 THEN 1
            ELSE 0
        END AS is_explosive_play,

        -- Play classification: Scoring play
        CASE
            WHEN touchdown = 1 OR field_goal_attempt = 1 OR safety = 1 THEN 1
            ELSE 0
        END AS is_scoring_play,

        -- Play classification: Turnover
        CASE
            WHEN interception = 1 OR fumble_lost = 1 THEN 1
            ELSE 0
        END AS is_turnover,

        -- Drive context: Play number within drive (calculated via window function)
        ROW_NUMBER() OVER (
            PARTITION BY game_id, drive
            ORDER BY play_id
        ) AS drive_play_number,

        -- Drive context: Yards remaining in drive
        CASE
            WHEN drive_start_yard_line IS NOT NULL AND yardline_100 IS NOT NULL THEN
                ABS(yardline_100 - drive_start_yard_line)
            ELSE NULL
        END AS drive_yards_gained_so_far,

        -- Field position classification
        CASE
            WHEN yardline_100 <= 10 THEN 'Red Zone'
            WHEN yardline_100 <= 20 THEN 'Green Zone'
            WHEN yardline_100 <= 50 THEN 'Opponent Territory'
            WHEN yardline_100 <= 80 THEN 'Own Territory'
            ELSE 'Deep Own Territory'
        END AS field_zone,

        -- Down situation classification
        CASE
            WHEN down = 3 AND ydstogo <= 3 THEN 'Third and Short'
            WHEN down = 3 AND ydstogo BETWEEN 4 AND 7 THEN 'Third and Medium'
            WHEN down = 3 AND ydstogo >= 8 THEN 'Third and Long'
            WHEN down = 4 THEN 'Fourth Down'
            ELSE 'Early Down'
        END AS down_situation,

        -- Score situation
        CASE
            WHEN ABS(score_differential) <= 3 THEN 'One Score Game'
            WHEN ABS(score_differential) <= 8 THEN 'Two Score Game'
            WHEN ABS(score_differential) > 8 THEN 'Blowout'
            ELSE 'Tie Game'
        END AS score_situation

    FROM base_plays
)

SELECT * FROM enhanced_plays