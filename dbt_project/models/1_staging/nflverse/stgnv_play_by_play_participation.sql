with

final as (
  select
    "nflverse_game_id" as nflverse_game_id,
  "old_game_id" as old_game_id,
  "play_id" as play_id,
  "possession_team" as possession_team,
  "offense_formation" as offense_formation,
  "offense_personnel" as offense_personnel,
  "defenders_in_box" as defenders_in_box,
  "defense_personnel" as defense_personnel,
  "number_of_pass_rushers" as number_of_pass_rushers,
  "players_on_play" as players_on_play,
  "offense_players" as offense_players,
  "defense_players" as defense_players,
  "n_offense" as n_offense,
  "n_defense" as n_defense,
  "ngs_air_yards" as ngs_air_yards,
  "time_to_throw" as time_to_throw,
  "was_pressure" as was_pressure,
  "route" as route,
  "defense_man_zone_type" as defense_man_zone_type,
  "defense_coverage_type" as defense_coverage_type,
  "source_year" as source_year,
  "load_type" as load_type,
  "loaded_at" as loaded_at
  from {{ source('nfl_verse_raw', 'play_by_play_participation') }}
)

select * from final