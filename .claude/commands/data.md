# /data - Data Management Command

## Description
Manage data ingestion, feature engineering, and validation for the NFL prediction system.

## Usage
```
/data [action] [options]
```

## Actions
- `refresh` - Update data from nflverse
- `features` - Rebuild feature store
- `validate` - Run data quality checks
- `explore` - Launch exploratory analysis
- `schema` - Show current data schema
- `lines` - Fetch latest Vegas lines

## Options
- `--seasons` - Specific seasons to process
- `--incremental` - Only update new data
- `--full` - Full rebuild (slow)
- `--table` - Specific table to process
- `--output` - Output format (table, json, csv)

## Examples

### Refresh latest data incrementally
```
/data refresh --incremental
```

### Rebuild all features for specific seasons
```
/data features --seasons 2022-2024 --full
```

### Validate data quality
```
/data validate --seasons 2024
```

### Fetch current Vegas lines
```
/data lines
```

## Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                            │
├─────────────────────────────────────────────────────────────┤
│  nflverse         Football Outsiders    Vegas Lines APIs    │
│  (play-by-play)   (DVOA, advanced)      (The Odds API)     │
│  (rosters)        (snap counts)         (historical)        │
│  (schedules)                                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   SNOWFLAKE RAW LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  raw_pbp          raw_dvoa              raw_vegas_lines    │
│  raw_rosters      raw_snap_counts       raw_weather        │
│  raw_schedules    raw_injuries                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   dbt STAGING LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  stg_plays        stg_games             stg_teams          │
│  stg_players      stg_injuries          stg_lines          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   dbt INTERMEDIATE LAYER                    │
├─────────────────────────────────────────────────────────────┤
│  int_play_epa           int_game_results                   │
│  int_team_rolling_stats int_opponent_adjustments           │
│  int_player_metrics     int_situational_stats              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   FEATURE STORE (MART)                      │
├─────────────────────────────────────────────────────────────┤
│  fct_game_features     (point-in-time correct)             │
│  fct_team_features     (versioned by date)                 │
│  fct_matchup_features  (ready for model input)             │
└─────────────────────────────────────────────────────────────┘
```

## Feature Store Schema

### fct_game_features
Primary feature table for model training and prediction.

| Column | Type | Description |
|--------|------|-------------|
| game_id | VARCHAR | Unique game identifier |
| game_date | DATE | Game date |
| season | INT | NFL season year |
| week | INT | Week number |
| home_team | VARCHAR | Home team abbreviation |
| away_team | VARCHAR | Away team abbreviation |
| home_epa_off_4g | FLOAT | Home team offensive EPA (4-game rolling) |
| home_epa_def_4g | FLOAT | Home team defensive EPA (4-game rolling) |
| away_epa_off_4g | FLOAT | Away team offensive EPA (4-game rolling) |
| away_epa_def_4g | FLOAT | Away team defensive EPA (4-game rolling) |
| home_success_rate | FLOAT | Home team success rate |
| away_success_rate | FLOAT | Away team success rate |
| home_rest_days | INT | Days since last game (home) |
| away_rest_days | INT | Days since last game (away) |
| home_travel_miles | INT | Travel distance for home team |
| away_travel_miles | INT | Travel distance for away team |
| divisional_game | BOOLEAN | Is divisional matchup |
| spread_open | FLOAT | Opening Vegas spread |
| total_open | FLOAT | Opening Vegas total |
| actual_margin | FLOAT | Actual point differential (target) |
| actual_total | FLOAT | Actual total points (target) |
| feature_date | TIMESTAMP | When features were calculated |

### Point-in-Time Correctness (CRITICAL)

Every feature must be calculated using ONLY data available before `game_date`.

```sql
-- CORRECT: Uses only games before current game
SELECT 
    g.game_id,
    AVG(past.epa) OVER (
        PARTITION BY g.home_team 
        ORDER BY past.game_date 
        ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING  -- Excludes current game!
    ) as home_epa_4g
FROM games g
LEFT JOIN game_stats past 
    ON past.team = g.home_team 
    AND past.game_date < g.game_date  -- Strict inequality!

-- WRONG: Includes current game data (look-ahead bias!)
SELECT 
    g.game_id,
    AVG(all_games.epa) as team_epa  -- Uses full season including future!
FROM games g
JOIN game_stats all_games ON all_games.team = g.home_team
```

## Data Quality Checks

### /data validate runs:

1. **Schema Validation**
   - All required columns present
   - Correct data types
   - No unexpected nulls

2. **Temporal Validation**
   - No future dates in features
   - Feature date < game date
   - Monotonic timestamps

3. **Statistical Validation**
   - EPA values in expected range (-1.0 to 1.0)
   - Success rates between 0 and 1
   - No impossible scores

4. **Consistency Validation**
   - Home/away team different
   - Game results match actual scores
   - Line movements plausible

## Code Template

```python
import nfl_data_py as nfl
import polars as pl
from snowflake.connector import connect

def refresh_data(
    seasons: list[int],
    incremental: bool = True
):
    """Refresh NFL data from nflverse."""
    
    # Load play-by-play with EPA
    pbp = nfl.import_pbp_data(seasons)
    
    # Load schedules
    schedules = nfl.import_schedules(seasons)
    
    # Load rosters
    rosters = nfl.import_rosters(seasons)
    
    if incremental:
        # Only load new games
        existing = get_existing_games()
        pbp = pbp[~pbp['game_id'].isin(existing)]
    
    # Upload to Snowflake
    upload_to_snowflake(pbp, 'raw_pbp')
    upload_to_snowflake(schedules, 'raw_schedules')
    upload_to_snowflake(rosters, 'raw_rosters')
    
    # Trigger dbt run
    run_dbt_models()
    
    return {'games_added': len(pbp['game_id'].unique())}

def build_features(
    seasons: list[int],
    full_rebuild: bool = False
):
    """Build feature store from raw data."""
    
    if full_rebuild:
        truncate_feature_store()
    
    # Run dbt models
    run_dbt(
        models=['staging', 'intermediate', 'marts'],
        full_refresh=full_rebuild
    )
    
    # Validate features
    validation_results = validate_features(seasons)
    
    if not validation_results['passed']:
        raise DataValidationError(validation_results['errors'])
    
    return validation_results

def validate_data(seasons: list[int]) -> dict:
    """Run comprehensive data quality checks."""
    
    checks = []
    
    # Schema checks
    checks.append(validate_schema())
    
    # Temporal checks (CRITICAL)
    checks.append(validate_no_lookahead())
    
    # Statistical checks
    checks.append(validate_value_ranges())
    
    # Consistency checks
    checks.append(validate_game_consistency())
    
    return {
        'passed': all(c['passed'] for c in checks),
        'checks': checks,
        'errors': [c['error'] for c in checks if not c['passed']]
    }
```

## dbt Model Examples

### stg_plays.sql
```sql
SELECT
    game_id,
    play_id,
    posteam as possession_team,
    defteam as defense_team,
    epa,
    success,
    pass,
    rush,
    qb_epa,
    cpoe,
    week,
    season
FROM {{ source('raw', 'pbp') }}
WHERE play_type IN ('pass', 'run')
```

### int_team_rolling_epa.sql
```sql
WITH game_epa AS (
    SELECT
        game_id,
        game_date,
        possession_team as team,
        AVG(epa) as game_epa,
        AVG(CASE WHEN pass = 1 THEN epa END) as pass_epa,
        AVG(CASE WHEN rush = 1 THEN epa END) as rush_epa,
        AVG(success) as success_rate
    FROM {{ ref('stg_plays') }}
    GROUP BY 1, 2, 3
)

SELECT
    game_id,
    game_date,
    team,
    AVG(game_epa) OVER (
        PARTITION BY team 
        ORDER BY game_date 
        ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
    ) as epa_4g_rolling,
    -- CRITICAL: Uses PRECEDING to exclude current game
    ...
FROM game_epa
```
