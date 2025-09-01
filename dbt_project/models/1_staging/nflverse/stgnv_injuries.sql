with

final as (
  select
    "season" as season,
    "game_type" as game_type,
    "team" as team,
    "week" as week,
    "gsis_id" as gsis_id,
    "position" as position,
    "full_name" as full_name,
    "first_name" as first_name,
    "last_name" as last_name,
    "report_primary_injury" as report_primary_injury,
    "report_secondary_injury" as report_secondary_injury,
    "report_status" as report_status,
    "practice_primary_injury" as practice_primary_injury,
    "practice_secondary_injury" as practice_secondary_injury,
    "practice_status" as practice_status,
    "date_modified" as date_modified,
    "source_year" as source_year,
    "load_type" as load_type,
    "loaded_at" as loaded_at
  from {{ source('nfl_verse_raw', 'injuries') }}
)

select * from final