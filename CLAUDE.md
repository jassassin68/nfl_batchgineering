# CLAUDE.md

## Project Overview
NFL prediction system: data pipeline + ML models for player stats and game outcomes (spreads, totals). Sources nflverse data, transforms via dbt in Snowflake, trains XGBoost models.

## Tech Stack
- **Database**: Snowflake (NFL_ANALYTICS database)
- **Transformations**: dbt-snowflake (SQL-first)
- **Python**: 3.11+, polars, XGBoost, scikit-learn
- **Package Management**: uv (not pip directly)
- **Code Quality**: ruff (not black/flake8)
- **Orchestration**: Dagster (complex), cron (simple)

## Key Directories
```
src/ingestion/       # Data loaders (nflverse → Snowflake)
src/ml/              # ML models (XGBoost training/prediction)
src/utils/           # Shared utilities, Snowflake connection
dbt_project/models/
  ├── staging/       # Views: raw data cleaning (stg_*)
  ├── intermediate/  # Mixed: feature engineering (int_*)
  └── marts/         # Tables: final ML datasets (fct_*, dim_*)
dagster_project/     # Orchestration assets/jobs/schedules
sql/                 # Ad-hoc queries, monitoring
notebooks/           # Exploration only, not production
```

## Snowflake Schema Structure
```
NFL_ANALYTICS.RAW          # Native tables from bulk Python loads
NFL_ANALYTICS.STAGING      # dbt views (cleaned data)
NFL_ANALYTICS.INTERMEDIATE # dbt tables/views (features)
NFL_ANALYTICS.MARTS        # dbt tables (ML-ready datasets)
NFL_ANALYTICS.ML           # Model artifacts, predictions
```

## Commands
```bash
# Environment
source .venv/bin/activate
uv sync                              # Install dependencies (NOT pip install)
uv add <package>                     # Add new dependency

# Code quality
ruff check .                         # Lint
ruff format .                        # Format
ruff check --fix .                   # Auto-fix

# dbt
cd dbt_project
dbt run --select staging             # Run staging models
dbt run --select marts               # Run mart models
dbt test --select marts              # Test marts
dbt docs generate && dbt docs serve  # Documentation

# ML
python src/ml/train_models.py --retrain           # Retrain all
python src/ml/train_models.py --position WR       # Train specific
python src/ml/predict.py --week next              # Generate predictions

# Full refresh
./scripts/weekly_refresh.sh          # Tuesday post-MNF refresh
```

## Coding Standards

### Python
- Type hints required on all functions
- Use polars, not pandas (performance)
- Pydantic for data validation
- Logging over print statements
- Classes inherit from `BaseModel` in `src/ml/base_model.py`

### SQL / dbt
- Lowercase SQL keywords
- Always alias tables in JOINs
- CTEs over nested subqueries
- Model naming: `stg_`, `int_`, `fct_`, `dim_`
- Staging models: `materialized='view'`
- Intermediate with heavy computation: `materialized='table'`
- Marts: always `materialized='table'`

### dbt Model Config Pattern
```sql
-- Simple transforms → view
{{ config(materialized='view') }}

-- Heavy window functions, aggregations → table
{{ config(materialized='table') }}
```

## File Boundaries
- **Safe to edit**: src/, dbt_project/models/, sql/, scripts/
- **Edit with caution**: dbt_project/dbt_project.yml, pyproject.toml
- **Never touch**: .env, models/*.pkl (trained artifacts), .venv/

## Testing Requirements
- pytest for Python (`tests/` directory)
- dbt tests for data quality (schema.yml files)
- Run `ruff check` before committing
- Run `dbt test --select marts` after model changes

## Domain Context

### nflverse Data Sources
- `player_stats`: Weekly player statistics
- `play_by_play`: Every play with EPA metrics
- `schedules`: Game schedule with betting lines
- `rosters`: Team rosters with player positions

### Key Metrics
- **EPA** (Expected Points Added): Primary efficiency metric
- **Fantasy Points**: PPR scoring for player predictions
- **Rolling averages**: 5-game and 10-game windows standard

### Position Abbreviations
- QB (Quarterback), RB (Running Back), WR (Wide Receiver), TE (Tight End)

### Feature Engineering Patterns
```sql
-- Standard rolling average pattern
AVG(metric) OVER (
  PARTITION BY player_id 
  ORDER BY game_date 
  ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING  -- Exclude current game
) AS metric_5game_avg
```

## Cost Awareness
- Snowflake budget: <$50/month
- Use X-Small warehouse with auto-suspend
- Prefer views over tables when possible
- Monitor with `sql/monitoring_queries.sql`

## Common Pitfalls
- Don't use pandas (use polars instead)
- Don't pip install (use uv add)
- Don't materialize staging models as tables
- Don't include current game in rolling averages (data leakage)
- Don't commit .env or model artifacts

## Verification Commands
```bash
# Before committing
ruff check . && ruff format --check .
cd dbt_project && dbt test --select marts
pytest tests/ -v
```