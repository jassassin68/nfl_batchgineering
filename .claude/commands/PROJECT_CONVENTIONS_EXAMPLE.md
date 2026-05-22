# Project Conventions — datastack overrides
#
# This file is read by datastack skills when they check for project-specific overrides.
# Place this at PROJECT_ROOT/.claude/PROJECT_CONVENTIONS.md
# or PROJECT_ROOT/PROJECT_CONVENTIONS.md
#
# Skills will apply these conventions over their global defaults.
# Delete any section you don't need — only configured sections are applied.

## Snowflake environment

- Database (dev): NFL_DEV
- Database (prod): NFL_PROD
- Warehouse: NFL_ANALYTICS_WH
- Raw schema: RAW
- Staging schema: STAGING
- Marts schema: ANALYTICS
- Role: SYSADMIN

## dbt project conventions

### Naming
- Staging: `stg_<source>__<entity>` (double underscore separator)
- Intermediate: `int_<entity>_<verb>` (e.g., int_team_rolling_epa)
- Facts: `fct_<entity>` (e.g., fct_game_predictions)
- Dimensions: `dim_<entity>` (e.g., dim_teams)
- One Big Table: `obt_<entity>` (e.g., obt_game_features)

### Materialization rules
- Staging models: always `view`
- Intermediate models: `view` by default, `table` if aggregating or referenced by 3+ downstream models
- Mart models: always `table`
- Incremental: use for any model processing >1M rows where source has a reliable timestamp

### SQL style
- Lowercase keywords (select, from, where)
- Trailing commas
- CTEs only, never nested subqueries
- Final CTE always named `final`
- Column order: pk → fk → dimensions → metrics → timestamps

### Testing requirements
- Every model: unique + not_null on primary key
- Every foreign key: relationships test
- Every status/type column: accepted_values
- Marts: at minimum one custom test or row count assertion

## Python conventions

- Package manager: uv (not pip)
- Linter/formatter: ruff
- DataFrame library: polars (convert to pandas only at Snowflake write boundary)
- Credentials: .env + python-dotenv
- Logging: Python logging module, never print()

## Source systems
# List your source systems so /plan-discover and /new-source know what exists

- nflverse: NFL play-by-play, schedules, rosters (loaded via Python/nfl_data_py)
- vegas_lines: Historical betting lines (loaded via Python scraping)

## Domain terminology
# Project-specific terms that skills should understand

- EPA: Expected Points Added (per-play efficiency metric)
- DVOA: Defense-adjusted Value Over Average
- ATS: Against The Spread
- PBP: Play-by-play data
- Grain for game-level models: one row per game_id
- Grain for play-level models: one row per game_id + play_id
