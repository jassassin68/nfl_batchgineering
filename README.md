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
  CREATE SCHEMA NFL_ANALYTICS.RAW;        -- External tables pointing to nflverse
  CREATE SCHEMA NFL_ANALYTICS.STAGING;    -- Cleaned data (views)
  CREATE SCHEMA NFL_ANALYTICS.INTERMEDIATE; -- Feature engineering (selective tables)
  CREATE SCHEMA NFL_ANALYTICS.MARTS;      -- Final ML datasets (tables)
  CREATE SCHEMA NFL_ANALYTICS.ML;         -- ML artifacts/predictions
  ```
- [ ] Configure connection parameters

### 1.3 External Tables Setup (1 hour)
- [ ] Research nflverse GitHub release URLs for parquet files
- [ ] Create external tables pointing to nflverse data:
  ```sql
  -- External tables for direct GitHub access
  CREATE EXTERNAL TABLE ext_play_by_play LOCATION='s3://nflverse-data/...'
  CREATE EXTERNAL TABLE ext_player_stats LOCATION='s3://nflverse-data/...'
  CREATE EXTERNAL TABLE ext_schedules LOCATION='s3://nflverse-data/...'
  CREATE EXTERNAL TABLE ext_rosters LOCATION='s3://nflverse-data/...'
  ```
- [ ] Test external table connectivity and performance
- [ ] Validate data structure matches expectations

### 1.4 dbt Setup (1 hour)
- [ ] Install dbt-snowflake
- [ ] Initialize dbt project with Snowflake profile
- [ ] Configure sources to point to external tables
- [ ] Create basic model structure and testing framework
- [ ] Set up dbt documentation

## Phase 2: dbt Data Transformations (6 hours)

### 2.1 Staging Models - External Table Integration (2 hours)
Clean and standardize data from external tables (materialized as VIEWS):

```sql
-- models/staging/sources.yml
sources:
  - name: external
    description: External tables pointing to nflverse GitHub data
    tables:
      - name: ext_play_by_play
      - name: ext_player_stats
      - name: ext_schedules
      - name: ext_rosters

-- models/staging/stg_play_by_play.sql
-- Read from external table, standardize columns, basic filtering
SELECT 
  game_id,
  season, 
  week,
  UPPER(TRIM(posteam)) AS posteam,
  CASE WHEN rush_attempt = 1 THEN TRUE ELSE FALSE END AS is_rush,
  epa,
  -- Include external table metadata
  source_filename,
  CURRENT_TIMESTAMP() AS processed_at
FROM {{ source('external', 'ext_play_by_play') }}
WHERE season >= {{ var('current_season') - var('lookback_seasons') }}

-- models/staging/stg_player_stats.sql  
-- Clean player statistics, handle position changes
-- models/staging/stg_schedules.sql
-- Standardize team names, parse betting lines
-- models/staging/stg_rosters.sql
-- Clean roster data with player mappings
```

### 2.2 Intermediate Models - Feature Engineering (3 hours)
**Selective materialization - only materialize complex calculations as TABLES**

**Player Features (TABLE - complex window functions):**
```sql
-- models/intermediate/int_player_rolling_stats.sql
{{ config(materialized='table') }}  -- Materialize for performance

WITH rolling_calculations AS (
  SELECT 
    player_id,
    game_date,
    -- Heavy computation - worth materializing
    AVG(fantasy_points) OVER (
      PARTITION BY player_id 
      ORDER BY game_date 
      ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
    ) AS fantasy_points_5game_avg,
    AVG(targets) OVER (
      PARTITION BY player_id 
      ORDER BY game_date 
      ROWS BETWEEN 9 PRECEDING AND 1 PRECEDING
    ) AS targets_10game_avg
  FROM {{ ref('stg_player_stats') }}
)

-- models/intermediate/int_player_situational_stats.sql
{{ config(materialized='view') }}  -- Simple aggregations, keep as view
-- Red zone performance, down/distance splits
-- Weather/dome performance, prime time performance
```

**Team Features (TABLE - aggregations across large datasets):**
```sql
-- models/intermediate/int_team_offensive_metrics.sql
{{ config(materialized='table') }}  -- Materialize team aggregations

WITH team_offense AS (
  SELECT 
    posteam AS team,
    season,
    week,
    AVG(epa) AS avg_epa_per_play,
    SUM(CASE WHEN yards_gained >= 10 THEN 1 ELSE 0 END) / COUNT(*) AS explosive_play_rate,
    SUM(CASE WHEN is_first_down THEN 1 ELSE 0 END) / COUNT(*) AS success_rate
  FROM {{ ref('stg_play_by_play') }}
  WHERE is_rush OR is_pass
  GROUP BY posteam, season, week
)

-- models/intermediate/int_team_defensive_metrics.sql
{{ config(materialized='table') }}
-- Defensive EPA allowed, pressure rates, coverage metrics

-- models/intermediate/int_team_situational_factors.sql
{{ config(materialized='view') }}  -- Simple joins, keep as view
-- Home/away splits, division games, rest advantage
```

### 2.3 Marts - Final Modeling Datasets (1 hour)
**Always materialized as TABLES for ML performance**

```sql
-- models/marts/fct_player_game_predictions.sql
{{ config(
  materialized='table',
  schema='marts'
) }}

-- Final dataset for player stat predictions
-- One row per player per upcoming game
WITH player_base AS (
  SELECT DISTINCT
    p.player_id,
    p.player_name,
    p.position,
    p.team,
    s.game_id,
    s.game_date,
    s.home_team,
    s.away_team
  FROM {{ ref('stg_rosters') }} p
  JOIN {{ ref('stg_schedules') }} s ON p.team IN (s.home_team, s.away_team)
  WHERE s.game_date > CURRENT_DATE()  -- Only future games
),

enriched_features AS (
  SELECT 
    pb.*,
    -- Rolling stats
    pr.fantasy_points_5game_avg,
    pr.targets_10game_avg,
    -- Team context
    CASE WHEN pb.team = pb.home_team THEN tom.avg_epa_per_play ELSE toa.avg_epa_per_play END AS team_offensive_epa,
    -- Opponent defense
    CASE WHEN pb.team = pb.home_team THEN tda.avg_epa_allowed ELSE tdm.avg_epa_allowed END AS opp_defensive_epa
  FROM player_base pb
  LEFT JOIN {{ ref('int_player_rolling_stats') }} pr 
    ON pb.player_id = pr.player_id
  LEFT JOIN {{ ref('int_team_offensive_metrics') }} tom 
    ON pb.home_team = tom.team
  LEFT JOIN {{ ref('int_team_offensive_metrics') }} toa 
    ON pb.away_team = toa.team
  LEFT JOIN {{ ref('int_team_defensive_metrics') }} tdm 
    ON pb.home_team = tdm.team  
  LEFT JOIN {{ ref('int_team_defensive_metrics') }} tda 
    ON pb.away_team = tda.team
)

SELECT * FROM enriched_features

-- models/marts/fct_team_game_predictions.sql
{{ config(
  materialized='table',
  schema='marts' 
) }}

-- Final dataset for spread/total predictions  
-- One row per game with both teams' features
WITH upcoming_games AS (
  SELECT 
    game_id,
    game_date,
    home_team,
    away_team,
    spread_line,
    total_line
  FROM {{ ref('stg_schedules') }}
  WHERE game_date > CURRENT_DATE()
),

team_features AS (
  SELECT 
    ug.*,
    -- Home team metrics
    tom_h.avg_epa_per_play AS home_offensive_epa,
    tdm_h.avg_epa_allowed AS home_defensive_epa,
    -- Away team metrics  
    tom_a.avg_epa_per_play AS away_offensive_epa,
    tdm_a.avg_epa_allowed AS away_defensive_epa,
    -- Situational factors
    tsf.rest_advantage,
    tsf.is_division_game
  FROM upcoming_games ug
  LEFT JOIN {{ ref('int_team_offensive_metrics') }} tom_h ON ug.home_team = tom_h.team
  LEFT JOIN {{ ref('int_team_defensive_metrics') }} tdm_h ON ug.home_team = tdm_h.team  
  LEFT JOIN {{ ref('int_team_offensive_metrics') }} tom_a ON ug.away_team = tom_a.team
  LEFT JOIN {{ ref('int_team_defensive_metrics') }} tdm_a ON ug.away_team = tdm_a.team
  LEFT JOIN {{ ref('int_team_situational_factors') }} tsf ON ug.game_id = tsf.game_id
)

SELECT * FROM team_features
```

## Phase 3: ML Model Development (4 hours)

### 3.1 Model Infrastructure (1 hour)
```python
# src/ml/base_model.py
from abc import ABC, abstractmethod
import polars as pl
import xgboost as xgb
import snowflake.connector

class BaseModel(ABC):
    def __init__(self, connection_params: dict):
        self.conn = snowflake.connector.connect(**connection_params)
    
    def load_data_from_snowflake(self, query: str) -> pl.DataFrame:
        """Load data directly from Snowflake marts"""
        cursor = self.conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pl.DataFrame(results, schema=columns)
    
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

### 3.2 Player Stat Models (1.5 hours)
```python
# src/ml/player_models.py
class PlayerStatModel(BaseModel):
    """Predict player statistics (yards, TDs, receptions, etc.)"""
    
    def __init__(self, connection_params: dict, stat_type: str, position: str):
        super().__init__(connection_params)
        self.stat_type = stat_type  # 'fantasy_points', 'receiving_yards', etc.
        self.position = position    # 'QB', 'RB', 'WR', 'TE'
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8
        )
        
    def load_training_data(self) -> pl.DataFrame:
        """Load historical player data for training"""
        query = f"""
        SELECT *
        FROM NFL_ANALYTICS.MARTS.FCT_PLAYER_GAME_PREDICTIONS
        WHERE position = '{self.position}'
          AND game_date < CURRENT_DATE()
          AND {self.stat_type} IS NOT NULL
        ORDER BY game_date DESC
        LIMIT 10000
        """
        return self.load_data_from_snowflake(query)
        
    def load_prediction_data(self) -> pl.DataFrame:
        """Load upcoming games for predictions"""
        query = f"""
        SELECT *
        FROM NFL_ANALYTICS.MARTS.FCT_PLAYER_GAME_PREDICTIONS  
        WHERE position = '{self.position}'
          AND game_date >= CURRENT_DATE()
        """
        return self.load_data_from_snowflake(query)
        
    def prepare_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prepare features for ML model"""
        feature_cols = [
            'fantasy_points_5game_avg',
            'targets_10game_avg',
            'team_offensive_epa',
            'opp_defensive_epa',
            # Add more position-specific features
        ]
        
        if self.position in ['WR', 'TE']:
            feature_cols.extend([
                'target_share_5game_avg',
                'air_yards_share_5game_avg'
            ])
        elif self.position == 'RB':
            feature_cols.extend([
                'carries_5game_avg',
                'goal_line_carries_avg'
            ])
            
        return df.select(feature_cols + [self.stat_type])
```

### 3.3 Team Outcome Models (1.5 hours)
```python
# src/ml/team_models.py  
class SpreadModel(BaseModel):
    """Predict point spreads"""
    
    def __init__(self, connection_params: dict):
        super().__init__(connection_params)
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05
        )
    
    def load_training_data(self) -> pl.DataFrame:
        """Load historical game data with outcomes"""
        query = """
        WITH historical_games AS (
          SELECT 
            tgp.*,
            s.away_score - s.home_score AS actual_spread,
            s.away_score + s.home_score AS actual_total
          FROM NFL_ANALYTICS.MARTS.FCT_TEAM_GAME_PREDICTIONS tgp
          JOIN NFL_ANALYTICS.STAGING.STG_SCHEDULES s 
            ON tgp.game_id = s.game_id
          WHERE s.away_score IS NOT NULL  -- Game has been played
            AND s.game_date >= '2019-01-01'
        )
        SELECT * FROM historical_games
        ORDER BY game_date DESC
        """
        return self.load_data_from_snowflake(query)
    
    def prepare_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prepare team-level features"""
        feature_cols = [
            'home_offensive_epa',
            'home_defensive_epa', 
            'away_offensive_epa',
            'away_defensive_epa',
            'rest_advantage',
            'is_division_game',
            'spread_line'  # Market spread as feature
        ]
        return df.select(feature_cols + ['actual_spread'])
        
class TotalModel(BaseModel):
    """Predict game totals (over/under)"""
    
    def __init__(self, connection_params: dict):
        super().__init__(connection_params)
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=150,
            max_depth=6
        )
        
    def prepare_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prepare features for total points prediction"""
        feature_cols = [
            'home_offensive_epa',
            'away_offensive_epa',
            'home_defensive_epa',
            'away_defensive_epa', 
            'total_line',  # Market total as feature
            'weather_wind_speed',
            'is_dome_game'
        ]
        return df.select(feature_cols + ['actual_total'])
```

## Phase 4: Orchestration & Automation (2 hours)

### 4.1 Simple Automation Setup (1 hour)
**Start with simple cron-based scheduling, upgrade to Dagster later if needed**

```bash
# scripts/weekly_refresh.sh
#!/bin/bash
# Weekly data refresh - run Tuesdays after MNF

echo "Starting weekly refresh..."

# Refresh external tables (automatic with nflverse updates)
echo "External tables auto-refresh with nflverse updates"

# Run dbt transformations
cd dbt_project
dbt run --select marts
dbt test --select marts

# Retrain models
cd ../
python src/ml/train_models.py --retrain

# Generate predictions for upcoming week  
python src/ml/predict.py --week next

echo "Weekly refresh complete!"
```

```python
# src/ml/train_models.py
"""Simple model training script"""
import click
from ml.player_models import PlayerStatModel
from ml.team_models import SpreadModel, TotalModel
from utils.snowflake_connection import get_connection_params

@click.command()
@click.option('--retrain', is_flag=True, help='Retrain all models')
@click.option('--position', help='Train models for specific position only')
def main(retrain: bool, position: str):
    """Train ML models"""
    conn_params = get_connection_params()
    
    if retrain or not position:
        # Train team models
        spread_model = SpreadModel(conn_params)
        total_model = TotalModel(conn_params)
        
        print("Training spread model...")
        spread_data = spread_model.load_training_data()
        spread_features = spread_model.prepare_features(spread_data)
        spread_model.train(spread_features)
        
        print("Training total model...")
        total_data = total_model.load_training_data()
        total_features = total_model.prepare_features(total_data)
        total_model.train(total_features)
    
    # Train player models by position
    positions = [position] if position else ['QB', 'RB', 'WR', 'TE']
    
    for pos in positions:
        print(f"Training {pos} fantasy points model...")
        player_model = PlayerStatModel(conn_params, 'fantasy_points', pos)
        player_data = player_model.load_training_data()
        player_features = player_model.prepare_features(player_data)
        player_model.train(player_features)

if __name__ == "__main__":
    main()
```

### 4.2 Monitoring and Cost Optimization (1 hour)
```sql
-- sql/monitoring_queries.sql

-- Monitor external table query costs
SELECT 
  query_text,
  execution_time,
  bytes_scanned,
  credits_used,
  start_time
FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY 
WHERE query_text LIKE '%ext_%'
  AND start_time >= DATEADD(day, -7, CURRENT_TIMESTAMP())
ORDER BY credits_used DESC;

-- Monitor dbt model performance
SELECT 
  query_text,
  execution_time,
  credits_used
FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY 
WHERE query_text LIKE '%dbt%'
  AND start_time >= DATEADD(day, -1, CURRENT_TIMESTAMP())
ORDER BY execution_time DESC;

-- Check warehouse utilization
SELECT 
  warehouse_name,
  credits_used,
  SUM(credits_used) OVER (PARTITION BY warehouse_name) AS total_credits
FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY 
WHERE start_time >= DATEADD(month, -1, CURRENT_TIMESTAMP());
```

```python
# src/utils/cost_monitor.py
"""Monitor Snowflake costs and performance"""
import snowflake.connector
from datetime import datetime, timedelta

class CostMonitor:
    def __init__(self, connection_params: dict):
        self.conn = snowflake.connector.connect(**connection_params)
    
    def check_daily_costs(self) -> dict:
        """Check costs for the last 24 hours"""
        query = """
        SELECT 
          SUM(credits_used) AS total_credits,
          AVG(execution_time) / 1000 AS avg_execution_time_seconds,
          COUNT(*) AS total_queries
        FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY 
        WHERE start_time >= DATEADD(day, -1, CURRENT_TIMESTAMP())
        """
        
        cursor = self.conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        
        return {
            'credits_used': result[0] or 0,
            'avg_execution_time': result[1] or 0,
            'total_queries': result[2] or 0,
            'estimated_daily_cost': (result[0] or 0) * 2  # Rough $2 per credit
        }
    
    def optimize_warehouses(self):
        """Auto-suspend idle warehouses"""
        cursor = self.conn.cursor()
        cursor.execute("ALTER WAREHOUSE NFL_ANALYTICS_WH SUSPEND")
        print("Warehouse suspended to save costs")
```

## Key Architectural Differences from Original Plan

### **🔄 Data Flow Changes**
```
OLD: nfl_data_py → Python Processing → Snowflake Raw → dbt
NEW: GitHub nflverse → External Tables → dbt Staging → Selective Materialization
```

### **💰 Cost Optimization Strategy**
- **External Tables**: ~$5-10/month (query costs only)
- **Selective Materialization**: Only complex calculations stored as tables
- **Views for Simple Logic**: Column renames, filters stay as views
- **Smart Warehouse Management**: Auto-suspend, right-sizing

### **⚡ Performance Trade-offs**
- **Slower**: Initial external table queries (2-3x slower than native)
- **Faster**: Complex feature engineering (materialized tables)
- **Optimal**: ML model training (marts are native tables)

## Updated Timeline & Milestones

### Week 1 (8 hours)
- ✅ Complete Phase 1 (Infrastructure + External Tables)
- ✅ Have external tables connected to nflverse data
- ✅ Basic dbt project with staging models working

### Week 2 (6 hours)  
- ✅ Complete Phase 2 (dbt transformations)
- ✅ Feature engineering pipeline with selective materialization
- ✅ Data quality tests passing on marts

### Week 3 (4 hours)
- ✅ Complete Phases 3-4 (ML + Simple Automation)
- ✅ First working models generating predictions
- ✅ Basic refresh scripts running

### Week 4 (2 hours - Optional Enhancement)
- ⭐ Upgrade to Dagster orchestration if needed
- ⭐ Advanced monitoring and alerting
- ⭐ Model performance tuning

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


## Success Metrics
- [ ] External tables successfully read nflverse data with <$50/month total costs
- [ ] Models generate predictions for next week's games
- [ ] Pipeline processes 5+ years of historical data efficiently via selective materialization
- [ ] Code is well-documented and AI assistant-friendly
- [ ] Predictions include confidence intervals/uncertainty estimates
- [ ] dbt models run in <5 minutes for weekly refresh
- [ ] External table queries complete in <30 seconds for staging models
- [ ] ML models achieve >60% directional accuracy on player stats
- [ ] Team models beat market spread/total predictions by >52% accuracy

## Budget Considerations (Updated for External Tables)
- **Snowflake External Table Queries**: ~$10-15/month (reading from GitHub)
- **Materialized Table Storage**: ~$5-10/month (only complex features stored)
- **Compute for dbt/ML**: ~$15-20/month (X-Small warehouse, auto-suspend)
- **Python dependencies**: Free (open source stack)
- **Total Expected**: ~$30-45/month (well under $50 budget)
- **Development time**: 20 hours target

## Cost Monitoring Commands
```sql
-- Weekly cost check (run this every Tuesday)
SELECT 
  DATE_TRUNC('week', start_time) AS week,
  SUM(credits_used) AS total_credits,
  SUM(credits_used) * 2 AS estimated_cost_usd
FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY 
WHERE start_time >= DATEADD(week, -4, CURRENT_TIMESTAMP())
GROUP BY week
ORDER BY week DESC;

-- External table usage specifically
SELECT 
  DATE(start_time) AS query_date,
  COUNT(*) AS external_table_queries,
  SUM(credits_used) AS external_table_credits
FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY 
WHERE query_text ILIKE '%ext_%'
  AND start_time >= DATEADD(week, -1, CURRENT_TIMESTAMP())
GROUP BY query_date
ORDER BY query_date DESC;
```

## Getting Started - Updated First Steps
1. **✅ Repository Setup**: uv, ruff, .gitignore configured
2. **🎯 Snowflake Trial**: Create account and run external table setup
3. **🔗 Find nflverse URLs**: Research GitHub release URLs for parquet files
4. **📊 Test External Tables**: Verify connectivity and basic queries
5. **🏗️ dbt Foundation**: Initialize project and create first staging model
6. **🤖 First ML Model**: Simple player stat prediction from marts

## External Table URL Research Template
```sql
-- Template for external table creation (URLs need to be researched)
-- Check https://github.com/nflverse/nflverse-data/releases for actual URLs

CREATE EXTERNAL TABLE ext_play_by_play(
    -- Column definitions from earlier artifact
) 
LOCATION = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_YYYY.parquet'
FILE_FORMAT = parquet_format
AUTO_REFRESH = TRUE;

-- Pattern likely to be:
-- pbp: 'https://github.com/nflverse/nflverse-data/releases/download/pbp/'
-- player_stats: 'https://github.com/nflverse/nflverse-data/releases/download/player_stats/'  
-- schedules: 'https://github.com/nflverse/nflverse-data/releases/download/schedules/'
-- rosters: 'https://github.com/nflverse/nflverse-data/releases/download/rosters/'
```

## Fallback Strategy
If external tables become too expensive or slow:

1. **Hybrid Approach**: External for historical, native for current season
2. **Selective Ingestion**: Use Python loader for just recent data (current season only)
3. **Materialization Strategy**: Increase what's stored as native tables vs. views
4. **Warehouse Scaling**: Move to larger warehouse if query performance is too slow

This plan balances your requirements for SQL-first processing, modern Python tooling, external table efficiency, and cost-effectiveness while building toward production-ready ML predictions with next week's games as the target output.# NFL Prediction System - Project Plan
