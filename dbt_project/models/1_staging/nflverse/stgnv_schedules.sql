-- NFL Game Schedules staging model
-- Contains all games including future unplayed games (where home_score is NULL)
-- Source: nflverse schedules dataset

with

final as (
    select
        "game_id" as game_id,
        "season"::number as season,
        "game_type" as game_type,
        "week"::number as week,
        try_to_date("gameday") as gameday,
        "weekday" as weekday,
        "gametime" as gametime,
        "away_team" as away_team,
        "away_score"::number as away_score,
        "home_team" as home_team,
        "home_score"::number as home_score,
        "location" as location,
        "result"::number as result,
        "total"::number as total,
        "overtime" as overtime,
        "old_game_id" as old_game_id,
        "gsis" as gsis,
        "nfl_detail_id" as nfl_detail_id,
        "pfr" as pfr,
        "pff" as pff,
        "espn" as espn,
        "ftn" as ftn,
        "away_rest"::number as away_rest,
        "home_rest"::number as home_rest,
        "away_moneyline"::number as away_moneyline,
        "home_moneyline"::number as home_moneyline,
        "spread_line"::number as spread_line,
        "away_spread_odds"::number as away_spread_odds,
        "home_spread_odds"::number as home_spread_odds,
        "total_line"::number as total_line,
        "under_odds"::number as under_odds,
        "over_odds"::number as over_odds,
        "div_game"::number as div_game,
        "roof" as roof,
        "surface" as surface,
        "temp"::number as temp,
        "wind"::number as wind,
        "away_qb_id" as away_qb_id,
        "home_qb_id" as home_qb_id,
        "away_qb_name" as away_qb_name,
        "home_qb_name" as home_qb_name,
        "away_coach" as away_coach,
        "home_coach" as home_coach,
        "referee" as referee,
        "stadium_id" as stadium_id,
        "stadium" as stadium,
        "source_year" as source_year,
        "load_type" as load_type,
        "loaded_at" as loaded_at
    from {{ source('nfl_verse_raw', 'schedules') }}
)

select * from final
