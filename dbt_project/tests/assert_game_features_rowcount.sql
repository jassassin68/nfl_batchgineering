-- Row-count floor for mart_game_prediction_features.
--
-- Catches a silently-empty or near-empty mart (e.g. an upstream join that
-- quietly dropped every row). A full NFL season is ~270 games; the floor is
-- set conservatively at 100 so it tolerates partial-season loads while still
-- failing loudly if the mart collapses. Raise the floor as more history loads.

SELECT row_count
FROM (
    SELECT COUNT(*) AS row_count
    FROM {{ ref('mart_game_prediction_features') }}
)
WHERE row_count < 100
