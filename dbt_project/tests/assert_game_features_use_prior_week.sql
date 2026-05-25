-- Look-ahead-bias guard.
--
-- mart_game_prediction_features must build each week-N game from week-(N-1)
-- team features only (data available before kickoff). The mart achieves this
-- with a lag join: `gc.week = htf.week + 1`.
--
-- This test re-derives that join independently and fails if a game's home
-- rolling-4wk EPA does not equal the SAME team's value from the prior week in
-- mart_predictive_features. A non-match means the lag join was broken and the
-- feature has leaked current-week (future) information into the prediction.
--
-- IS DISTINCT FROM treats two NULLs as equal, so legitimately missing
-- features do not trip the test.

SELECT
    g.game_id,
    g.season,
    g.week,
    g.home_team,
    g.home_epa_l4w        AS game_feature_value,
    p.epa_per_play_l4w    AS prior_week_value
FROM {{ ref('mart_game_prediction_features') }} g
JOIN {{ ref('mart_predictive_features') }} p
    ON g.home_team = p.team
    AND g.season = p.season
    AND g.week = p.week + 1
WHERE g.home_epa_l4w IS DISTINCT FROM p.epa_per_play_l4w
