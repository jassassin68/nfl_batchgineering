# nfl_batchgineering
Manage code related to data work using NFL datasets

# NFL Prediction System - Project Plan

## Project Overview
Build a data pipeline and ML system to predict NFL player stats and team outcomes (spreads, totals) using nflverse data, targeting next week's games with ~20 hours of development time.

## Tech Stack
- **Database**: Snowflake (primary compute/storage)
- **ETL/Transformations**: dbt + SQL-first approach
- **Data Ingestion**: Python with `nfl_data_py`, `polars`, `uv`
- **ML**: Python with `XGBoost`, `polars`, `scikit-learn`
- **Orchestration**: Dagster (for complex workflows), cron (for simple scheduling)
- **Code Quality**: `ruff` for linting/formatting

## Repository Structure
```
nfl-prediction-system/
├── README.md
├── pyproject.toml              # uv/ruff config
├── .env.example               # Environment variables template
├── requirements.txt           # Python dependencies
├── dagster_project/           # Dagster orchestration
│   ├── __init__.py
│   ├── assets/               # Data assets
│   ├── jobs/                 # Job definitions
│   └── schedules/            # Scheduling
├── dbt_project/              # dbt transformations
│   ├── dbt_project.yml
│   ├── profiles.yml
│   ├── models/
│   │   ├── staging/          # Raw data cleaning
│   │   ├── intermediate/     # Feature engineering
│   │   └── marts/           # Final modeling datasets
│   ├── macros/              # Reusable SQL functions
│   └── tests/               # Data quality tests
├── src/
│   ├── ingestion/           # Data loading scripts
│   ├── ml/                  # ML model code
│   └── utils/               # Shared utilities
├── sql/                     # Ad-hoc SQL scripts
├── models/                  # Trained model artifacts
└── notebooks/               # Analysis/exploration
```

## Phase 1: Infrastructure Setup (4 hours)

### 1.1 Repository & Environment Setup (1 hour)
- [x] Initialize GitHub repository with proper `.gitignore`
- [x] Set up `uv` for dependency management
- [x] Configure `ruff` for code formatting/linting
- [x] Create basic `pyproject.toml`

### 1.2 Snowflake Setup (1 hour)
- [ ] Create Snowflake trial account
- [ ] Set up database/schema structure:
  ```sql
  CREATE DATABASE NFL_ANALYTICS;
  CREATE SCHEMA NFL_ANALYTICS.RAW;      -- Raw nflverse data
  CREATE SCHEMA NFL_ANALYTICS.STAGING;  -- Cleaned data
  CREATE SCHEMA NFL_ANALYTICS.MARTS;    -- Feature-engineered data
  CREATE SCHEMA NFL_ANALYTICS.ML;       -- ML artifacts/predictions
  ```
- [ ] Configure connection parameters

### 1.3 dbt Setup (1 hour)
- [ ] Install dbt-snowflake
- [ ] Initialize dbt project with Snowflake profile
- [ ] Create basic model structure and testing framework
- [ ] Set up dbt documentation

### 1.4 Dagster Setup (1 hour)
- [ ] Initialize Dagster project
- [ ] Configure Snowflake resource
- [ ] Create basic asset structure
- [ ] Set up local development environment

## Phase 2: Data Ingestion Pipeline (4 hours)

### 2.1 Raw Data Ingestion (2 hours)
Create Python scripts to load nflverse data into Snowflake:

**Priority Datasets:**
- Play-by-play data (2019-2024) - Core for all features
- Weekly rosters - Player position/team mapping
- Schedules - Game info, spreads, totals
- Player stats (seasonal/weekly) - Target variables for player predictions
- Team-level stats - Target variables for team predictions

**Python Script Structure:**
```python
# src/ingestion/nflverse_loader.py
import polars as pl
import nfl_data_py as nfl
from snowflake.connector.pandas_tools import write_pandas

def load_play_by_play_data(years: list[int]) -> None:
    """Load play-by-play data to Snowflake"""
    # Use nfl_data_py to get data
    # Convert to polars for efficient processing
    # Write to Snowflake RAW schema

def load_player_stats(years: list[int]) -> None:
    """Load player statistics"""
    
def load_schedules(years: list[int]) -> None:
    """Load schedule/betting data"""
```

### 2.2 Dagster Assets (2 hours)
- [ ] Create Dagster assets for each data source
- [ ] Implement incremental loading logic
- [ ] Add data quality checks
- [ ] Set up monitoring/alerting

## Phase 3: dbt Data Transformations (6 hours)

### 3.1 Staging Models (2 hours)
Clean and standardize raw data:

```sql
-- models/staging/stg_play_by_play.sql
-- Standardize column names, handle nulls, basic filtering

-- models/staging/stg_player_stats.sql  
-- Clean player statistics, handle position changes

-- models/staging/stg_schedules.sql
-- Standardize team names, parse betting lines
```

### 3.2 Intermediate Models - Feature Engineering (3 hours)

**Player Features:**
```sql
-- models/intermediate/int_player_rolling_stats.sql
-- Rolling averages (3, 5, 10 game windows)
-- Target share trends, snap count trends
-- Matchup-specific performance

-- models/intermediate/int_player_situational_stats.sql  
-- Red zone performance, down/distance splits
-- Weather/dome performance, prime time performance
-- Rest days impact
```

**Team Features:**
```sql
-- models/intermediate/int_team_offensive_metrics.sql
-- EPA per play, success rate, explosive play rate
-- Red zone efficiency, 3rd down conversion rates

-- models/intermediate/int_team_defensive_metrics.sql
-- Defensive EPA allowed, pressure rates
-- Coverage metrics, run defense efficiency

-- models/intermediate/int_team_situational_factors.sql
-- Home/away splits, division games, revenge games
-- Rest advantage, travel distance, weather
```

### 3.3 Marts - Final Modeling Datasets (1 hour)
```sql
-- models/marts/fct_player_game_predictions.sql
-- Final dataset for player stat predictions
-- One row per player per upcoming game

-- models/marts/fct_team_game_predictions.sql  
-- Final dataset for spread/total predictions
-- One row per game with both teams' features
```

## Phase 4: ML Model Development (4 hours)

### 4.1 Model Infrastructure (1 hour)
```python
# src/ml/base_model.py
from abc import ABC, abstractmethod
import polars as pl
import xgboost as xgb

class BaseModel(ABC):
    @abstractmethod
    def prepare_features(self, df: pl.DataFrame) -> pl.DataFrame:
        pass
    
    @abstractmethod  
    def train(self, df: pl.DataFrame) -> None:
        pass
        
    @abstractmethod
    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        pass
```

### 4.2 Player Stat Models (1.5 hours)
```python
# src/ml/player_models.py
class PlayerStatModel(BaseModel):
    """Predict player statistics (yards, TDs, receptions, etc.)"""
    
    def __init__(self, stat_type: str):
        self.stat_type = stat_type  # 'passing_yards', 'rushing_tds', etc.
        self.model = xgb.XGBRegressor()
        
    def prepare_features(self, df: pl.DataFrame) -> pl.DataFrame:
        # Rolling averages, matchup factors, team context
        # Snap count trends, target share, etc.
        pass
```

### 4.3 Team Outcome Models (1.5 hours)
```python
# src/ml/team_models.py  
class SpreadModel(BaseModel):
    """Predict point spreads"""
    
class TotalModel(BaseModel):
    """Predict game totals (over/under)"""
```

## Phase 5: Orchestration & Automation (2 hours)

### 5.1 Dagster Jobs (1 hour)
```python
# dagster_project/jobs/weekly_refresh.py
@job
def weekly_data_refresh():
    """Refresh data every Tuesday after MNF"""
    # Load new game data
    # Run dbt transformations  
    # Retrain models if needed
    # Generate predictions for upcoming week

@job  
def daily_prediction_update():
    """Update predictions as injury/line data changes"""
    # Light refresh of key inputs
    # Regenerate predictions
```

### 5.2 Scheduling (1 hour)
- [ ] Set up cron jobs for data refresh
- [ ] Configure Dagster schedules
- [ ] Add monitoring/alerting for failed runs

## Key SQL Transformations (Examples)

### Rolling Player Statistics
```sql
-- Calculate 5-game rolling averages for player performance
WITH player_game_stats AS (
  SELECT 
    player_id,
    game_id,
    game_date,
    passing_yards,
    rushing_yards,
    receiving_yards,
    fantasy_points
  FROM {{ ref('stg_player_stats') }}
),

rolling_stats AS (
  SELECT 
    player_id,
    game_id,
    AVG(passing_yards) OVER (
      PARTITION BY player_id 
      ORDER BY game_date 
      ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
    ) AS passing_yards_5game_avg,
    
    AVG(fantasy_points) OVER (
      PARTITION BY player_id 
      ORDER BY game_date 
      ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING  
    ) AS fantasy_points_5game_avg
    
  FROM player_game_stats
)

SELECT * FROM rolling_stats
```

### Matchup-Based Features
```sql
-- Team defensive rankings against specific positions
WITH defensive_rankings AS (
  SELECT 
    team,
    season,
    RANK() OVER (PARTITION BY season ORDER BY avg_passing_yards_allowed) AS pass_def_rank,
    RANK() OVER (PARTITION BY season ORDER BY avg_rushing_yards_allowed) AS rush_def_rank,
    avg_fantasy_points_allowed_to_qb,
    avg_fantasy_points_allowed_to_rb,
    avg_fantasy_points_allowed_to_wr
  FROM {{ ref('int_team_defensive_metrics') }}
)

SELECT * FROM defensive_rankings
```

## Timeline & Milestones

### Week 1 (8 hours)
- Complete Phases 1-2 (Infrastructure + Data Ingestion)
- Have raw nflverse data flowing into Snowflake
- Basic dbt project structure

### Week 2 (8 hours)  
- Complete Phase 3 (dbt transformations)
- Feature engineering pipeline fully built
- Data quality tests passing

### Week 3 (4 hours)
- Complete Phases 4-5 (ML + Orchestration)
- First working models generating predictions
- Automated pipeline running

## Success Metrics
- [ ] Data pipeline refreshes automatically within $50/month budget
- [ ] Models generate predictions for next week's games
- [ ] Pipeline processes 5+ years of historical data efficiently
- [ ] Code is well-documented and AI assistant-friendly
- [ ] Predictions include confidence intervals/uncertainty estimates

## Budget Considerations
- **Snowflake**: Should stay under $30-40/month with proper warehouse sizing
- **Python dependencies**: Free (using open source stack)
- **Development time**: 20 hours target

## Getting Started - First Steps
1. Clone repository template
2. Set up Snowflake trial account  
3. Configure local development environment with uv
4. Run initial data ingestion for 2023-2024 seasons
5. Build first dbt staging models
6. Create simple player stat prediction model

This plan balances your requirements for SQL-first processing, modern Python tooling, and cost-effectiveness while building toward production-ready ML predictions.