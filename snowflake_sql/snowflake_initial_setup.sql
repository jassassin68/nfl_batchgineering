-- sql/setup_external_tables.sql
-- Set up external tables to reference nflverse data directly

USE DATABASE NFL_BATCHGINEERING;
USE SCHEMA RAW;

-- Create file format for parquet files
CREATE OR REPLACE FILE FORMAT parquet_format
    TYPE = 'PARQUET'
    COMPRESSION = 'AUTO';

-- Create file format for CSV files
CREATE OR REPLACE FILE FORMAT csv_format 
    TYPE = 'CSV'
    FIELD_DELIMITER = ','
    SKIP_HEADER = 1
    NULL_IF = ('NULL', 'null', '', 'NA')
    EMPTY_FIELD_AS_NULL = TRUE
    FIELD_OPTIONALLY_ENCLOSED_BY = '"';

-- External table for play-by-play data (parquet format for performance)
CREATE OR REPLACE EXTERNAL TABLE ext_play_by_play (
    play_id NUMBER AS (value:play_id::NUMBER),
    game_id VARCHAR AS (value:game_id::VARCHAR),
    old_game_id VARCHAR AS (value:old_game_id::VARCHAR),
    season NUMBER AS (value:season::NUMBER), 
    week NUMBER AS (value:week::NUMBER),
    season_type VARCHAR AS (value:season_type::VARCHAR),
    home_team VARCHAR AS (value:home_team::VARCHAR),
    away_team VARCHAR AS (value:away_team::VARCHAR),
    posteam VARCHAR AS (value:posteam::VARCHAR),
    defteam VARCHAR AS (value:defteam::VARCHAR),
    game_date DATE AS (value:game_date::DATE),
    down NUMBER AS (value:down::NUMBER),
    ydstogo NUMBER AS (value:ydstogo::NUMBER),
    yardline_100 NUMBER AS (value:yardline_100::NUMBER),
    quarter_seconds_remaining NUMBER AS (value:quarter_seconds_remaining::NUMBER),
    half_seconds_remaining NUMBER AS (value:half_seconds_remaining::NUMBER),
    game_seconds_remaining NUMBER AS (value:game_seconds_remaining::NUMBER),
    qtr NUMBER AS (value:qtr::NUMBER),
    desc VARCHAR AS (value:desc::VARCHAR),
    play_type VARCHAR AS (value:play_type::VARCHAR),
    yards_gained NUMBER AS (value:yards_gained::NUMBER),
    -- Key player columns
    passer_player_id VARCHAR AS (value:passer_player_id::VARCHAR),
    passer_player_name VARCHAR AS (value:passer_player_name::VARCHAR),
    receiver_player_id VARCHAR AS (value:receiver_player_id::VARCHAR),
    receiver_player_name VARCHAR AS (value:receiver_player_name::VARCHAR),
    rusher_player_id VARCHAR AS (value:rusher_player_id::VARCHAR),
    rusher_player_name VARCHAR AS (value:rusher_player_name::VARCHAR),
    -- Stats
    passing_yards NUMBER AS (value:passing_yards::NUMBER),
    receiving_yards NUMBER AS (value:receiving_yards::NUMBER),
    rushing_yards NUMBER AS (value:rushing_yards::NUMBER),
    air_yards NUMBER AS (value:air_yards::NUMBER),
    yards_after_catch NUMBER AS (value:yards_after_catch::NUMBER),
    -- Advanced metrics
    epa NUMBER AS (value:epa::NUMBER),
    wpa NUMBER AS (value:wpa::NUMBER),
    wp NUMBER AS (value:wp::NUMBER),
    -- Indicators (stored as numbers in nflverse, convert to boolean-like)
    rush_attempt NUMBER AS (value:rush_attempt::NUMBER),
    pass_attempt NUMBER AS (value:pass_attempt::NUMBER),
    sack NUMBER AS (value:sack::NUMBER),
    touchdown NUMBER AS (value:touchdown::NUMBER),
    interception NUMBER AS (value:interception::NUMBER),
    fumble_lost NUMBER AS (value:fumble_lost::NUMBER),
    first_down_rush NUMBER AS (value:first_down_rush::NUMBER),
    first_down_pass NUMBER AS (value:first_down_pass::NUMBER),
    -- Scores
    total_home_score NUMBER AS (value:total_home_score::NUMBER),
    total_away_score NUMBER AS (value:total_away_score::NUMBER),
    posteam_score NUMBER AS (value:posteam_score::NUMBER),
    defteam_score NUMBER AS (value:defteam_score::NUMBER),
    score_differential NUMBER AS (value:score_differential::NUMBER),
    -- Additional situational columns
    shotgun NUMBER AS (value:shotgun::NUMBER),
    no_huddle NUMBER AS (value:no_huddle::NUMBER),
    qb_dropback NUMBER AS (value:qb_dropback::NUMBER),
    complete_pass NUMBER AS (value:complete_pass::NUMBER),
    -- File metadata
    metadata$filename AS source_filename,
    metadata$file_row_number AS file_row_number
)
LOCATION = 's3://nflverse-data/releases/pbp/' -- Note: This is conceptual, actual URL structure may differ
FILE_FORMAT = parquet_format
AUTO_REFRESH = TRUE
REFRESH_ON_CREATE = TRUE;

-- External table for player stats (parquet)
CREATE OR REPLACE EXTERNAL TABLE ext_player_stats(
    player_id VARCHAR AS (value:player_id::VARCHAR),
    player_name VARCHAR AS (value:player_name::VARCHAR),
    player_display_name VARCHAR AS (value:player_display_name::VARCHAR),
    position VARCHAR AS (value:position::VARCHAR),
    position_group VARCHAR AS (value:position_group::VARCHAR),
    recent_team VARCHAR AS (value:recent_team::VARCHAR),
    season NUMBER AS (value:season::NUMBER),
    week NUMBER AS (value:week::NUMBER),
    season_type VARCHAR AS (value:season_type::VARCHAR),
    -- Passing stats
    attempts NUMBER AS (value:attempts::NUMBER),
    completions NUMBER AS (value:completions::NUMBER),
    passing_yards NUMBER AS (value:passing_yards::NUMBER),
    passing_tds NUMBER AS (value:passing_tds::NUMBER),
    interceptions NUMBER AS (value:interceptions::NUMBER),
    sacks NUMBER AS (value:sacks::NUMBER),
    -- Rushing stats  
    carries NUMBER AS (value:carries::NUMBER),
    rushing_yards NUMBER AS (value:rushing_yards::NUMBER),
    rushing_tds NUMBER AS (value:rushing_tds::NUMBER),
    -- Receiving stats
    receptions NUMBER AS (value:receptions::NUMBER),
    targets NUMBER AS (value:targets::NUMBER),
    receiving_yards NUMBER AS (value:receiving_yards::NUMBER),
    receiving_tds NUMBER AS (value:receiving_tds::NUMBER),
    -- Advanced metrics
    target_share NUMBER AS (value:target_share::NUMBER),
    air_yards_share NUMBER AS (value:air_yards_share::NUMBER),
    wopr NUMBER AS (value:wopr::NUMBER),
    -- Fantasy points
    fantasy_points NUMBER AS (value:fantasy_points::NUMBER),
    fantasy_points_ppr NUMBER AS (value:fantasy_points_ppr::NUMBER),
    -- File metadata
    metadata$filename AS source_filename,
    metadata$file_row_number AS file_row_number
)
LOCATION = 's3://nflverse-data/releases/player_stats/'
FILE_FORMAT = parquet_format
AUTO_REFRESH = TRUE
REFRESH_ON_CREATE = TRUE;

-- External table for schedules (parquet)
CREATE OR REPLACE EXTERNAL TABLE ext_schedules(
    game_id VARCHAR AS (value:game_id::VARCHAR),
    season NUMBER AS (value:season::NUMBER),
    game_type VARCHAR AS (value:game_type::VARCHAR),
    week NUMBER AS (value:week::NUMBER),
    gameday DATE AS (value:gameday::DATE),
    weekday VARCHAR AS (value:weekday::VARCHAR),
    gametime TIME AS (value:gametime::TIME),
    away_team VARCHAR AS (value:away_team::VARCHAR),
    home_team VARCHAR AS (value:home_team::VARCHAR),
    away_score NUMBER AS (value:away_score::NUMBER),
    home_score NUMBER AS (value:home_score::NUMBER),
    location VARCHAR AS (value:location::VARCHAR),
    result NUMBER AS (value:result::NUMBER),
    total NUMBER AS (value:total::NUMBER),
    -- Betting lines
    spread_line NUMBER AS (value:spread_line::NUMBER),
    total_line NUMBER AS (value:total_line::NUMBER),
    away_moneyline NUMBER AS (value:away_moneyline::NUMBER),
    home_moneyline NUMBER AS (value:home_moneyline::NUMBER),
    -- Game context
    div_game NUMBER AS (value:div_game::NUMBER),
    away_rest NUMBER AS (value:away_rest::NUMBER),
    home_rest NUMBER AS (value:home_rest::NUMBER),
    -- Conditions
    roof VARCHAR AS (value:roof::VARCHAR),
    surface VARCHAR AS (value:surface::VARCHAR),
    temp NUMBER AS (value:temp::NUMBER),
    wind NUMBER AS (value:wind::NUMBER),
    -- Personnel
    away_coach VARCHAR AS (value:away_coach::VARCHAR),
    home_coach VARCHAR AS (value:home_coach::VARCHAR),
    referee VARCHAR AS (value:referee::VARCHAR),
    stadium VARCHAR AS (value:stadium::VARCHAR),
    -- File metadata
    metadata$filename AS source_filename,
    metadata$file_row_number AS file_row_number
)
LOCATION = 's3://nflverse-data/releases/schedules/'
FILE_FORMAT = parquet_format
AUTO_REFRESH = TRUE
REFRESH_ON_CREATE = TRUE;

-- External table for rosters (parquet)
CREATE OR REPLACE EXTERNAL TABLE ext_rosters(
    season NUMBER AS (value:season::NUMBER),
    team VARCHAR AS (value:team::VARCHAR),
    position VARCHAR AS (value:position::VARCHAR),
    depth_chart_position VARCHAR AS (value:depth_chart_position::VARCHAR),
    jersey_number NUMBER AS (value:jersey_number::NUMBER),
    status VARCHAR AS (value:status::VARCHAR),
    full_name VARCHAR AS (value:full_name::VARCHAR),
    first_name VARCHAR AS (value:first_name::VARCHAR),
    last_name VARCHAR AS (value:last_name::VARCHAR),
    birth_date DATE AS (value:birth_date::DATE),
    height VARCHAR AS (value:height::VARCHAR),
    weight NUMBER AS (value:weight::NUMBER),
    college VARCHAR AS (value:college::VARCHAR),
    gsis_id VARCHAR AS (value:gsis_id::VARCHAR),
    espn_id VARCHAR AS (value:espn_id::VARCHAR),
    yahoo_id VARCHAR AS (value:yahoo_id::VARCHAR),
    pfr_id VARCHAR AS (value:pfr_id::VARCHAR),
    fantasy_data_id VARCHAR AS (value:fantasy_data_id::VARCHAR),
    sleeper_id VARCHAR AS (value:sleeper_id::VARCHAR),
    week NUMBER AS (value:week::NUMBER),
    -- File metadata
    metadata$filename AS source_filename,
    metadata$file_row_number AS file_row_number
)
LOCATION = 's3://nflverse-data/releases/rosters/'
FILE_FORMAT = parquet_format
AUTO_REFRESH = TRUE
REFRESH_ON_CREATE = TRUE;

-- Test the external tables
SELECT COUNT(*) FROM ext_play_by_play WHERE season = 2024;
SELECT COUNT(*) FROM ext_player_stats WHERE season = 2024;
SELECT COUNT(*) FROM ext_schedules WHERE season = 2024;
SELECT COUNT(*) FROM ext_rosters WHERE season = 2024;

-- Show external table properties
SHOW EXTERNAL TABLES;
DESCRIBE EXTERNAL TABLE ext_play_by_play;

-- Note: The actual URLs need to be determined from nflverse-data repository
-- This is a template showing the structure - you'll need to find the correct
-- GitHub release URLs for the parquet files