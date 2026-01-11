-- models/2_intermediate/int_game_vegas_lines.sql
-- Extract game-level Vegas lines from play-by-play data
-- Grain: One row per game with opening lines

{{ config(
    materialized='table',
    tags=['intermediate', 'game_level', 'vegas']
) }}

WITH first_play_per_game AS (
    -- Get the first play of each game to capture opening lines
    -- Vegas lines are constant throughout the game in nflverse data,
    -- so we just need one row per game
    SELECT
        game_id,
        season,
        week,
        home_team,
        away_team,
        spread_line::number as spread_line,  -- Positive means home team favored
        total_line::number as total_line,   -- Over/under total points
        ROW_NUMBER() OVER (
            PARTITION BY game_id
            ORDER BY play_id
        ) AS play_order
    FROM {{ ref('stgnv_play_by_play') }}
    WHERE spread_line IS NOT NULL
      AND total_line IS NOT NULL
)

SELECT
    game_id,
    season,
    week,
    home_team,
    away_team,
    spread_line AS vegas_spread,
    total_line AS vegas_total,

    -- Derived: Implied probabilities
    -- Spread of -3 implies ~60% win probability for favorite
    -- Formula: 0.50 + (spread_points * 0.033) for favorites
    CASE
        WHEN spread_line < 0 THEN 0.50 + (ABS(spread_line) * 0.033)
        WHEN spread_line > 0 THEN 0.50 - (spread_line * 0.033)
        ELSE 0.50
    END AS vegas_home_win_prob,

    -- Betting context classification
    CASE
        WHEN ABS(spread_line) <= 3 THEN 'toss_up'
        WHEN ABS(spread_line) BETWEEN 3.5 AND 7 THEN 'moderate_favorite'
        WHEN ABS(spread_line) > 7 THEN 'heavy_favorite'
        ELSE 'unknown'
    END AS spread_category

FROM first_play_per_game
WHERE play_order = 1  -- Only first play to get one row per game
