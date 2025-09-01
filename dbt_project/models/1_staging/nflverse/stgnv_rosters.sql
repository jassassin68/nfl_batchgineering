with

final as (
  select
    "season" as season,
    "team" as team,
    "position" as position,
    "depth_chart_position" as depth_chart_position,
    "jersey_number" as jersey_number,
    "status" as status,
    "full_name" as full_name,
    "first_name" as first_name,
    "last_name" as last_name,
    "birth_date" as birth_date,
    "height" as height,
    "weight" as weight,
    "college" as college,
    "gsis_id" as gsis_id,
    "espn_id" as espn_id,
    "sportradar_id" as sportradar_id,
    "yahoo_id" as yahoo_id,
    "rotowire_id" as rotowire_id,
    "pff_id" as pff_id,
    "pfr_id" as pfr_id,
    "fantasy_data_id" as fantasy_data_id,
    "sleeper_id" as sleeper_id,
    "years_exp" as years_exp,
    "headshot_url" as headshot_url,
    "ngs_position" as ngs_position,
    "week" as week,
    "game_type" as game_type,
    "status_description_abbr" as status_description_abbr,
    "football_name" as football_name,
    "esb_id" as esb_id,
    "gsis_it_id" as gsis_it_id,
    "smart_id" as smart_id,
    "entry_year" as entry_year,
    "rookie_year" as rookie_year,
    "draft_club" as draft_club,
    "draft_number" as draft_number,
    "source_year" as source_year,
    "load_type" as load_type,
    "loaded_at" as loaded_at
  from {{ source('nfl_verse_raw', 'rosters') }}
)

select * from final