-- Complete Snowflake Database Setup for NFL Prediction System
-- Implements medallion architecture with proper schemas and configurations

-- Set context
USE ROLE ACCOUNTADMIN;

-- Create or use existing database
CREATE DATABASE IF NOT EXISTS NFL_BATCHGINEERING;
USE DATABASE NFL_BATCHGINEERING;

-- Create all required schemas for medallion architecture
-- RAW schema (Bronze layer) - already exists but ensure it's properly configured
CREATE SCHEMA IF NOT EXISTS RAW
    COMMENT = 'Bronze layer: Raw data from nflverse sources, all columns as VARCHAR for flexibility';

-- STAGING schema (Silver layer) - cleaned and standardized data
CREATE SCHEMA IF NOT EXISTS STAGING
    COMMENT = 'Silver layer: Cleaned and standardized data with proper typing from RAW layer';

-- INTERMEDIATE schema (Gold layer) - feature engineered data
CREATE SCHEMA IF NOT EXISTS INTERMEDIATE
    COMMENT = 'Gold layer: Feature-engineered datasets with rolling statistics and team metrics';

-- MARTS schema (Platinum layer) - ML-ready datasets
CREATE SCHEMA IF NOT EXISTS MARTS
    COMMENT = 'Platinum layer: ML-ready datasets optimized for model consumption';

-- ML schema - predictions and model artifacts
CREATE SCHEMA IF NOT EXISTS ML
    COMMENT = 'ML layer: Model predictions, artifacts, and performance tracking';

-- Configure warehouse for optimal performance and cost efficiency
CREATE WAREHOUSE IF NOT EXISTS COMPUTE_WH
    WITH 
    WAREHOUSE_SIZE = 'X-SMALL'  -- Start small, can scale up as needed
    AUTO_SUSPEND = 60           -- Auto-suspend after 1 minute of inactivity
    AUTO_RESUME = TRUE          -- Auto-resume when queries are submitted
    COMMENT = 'Primary warehouse for NFL prediction system with auto-suspend for cost optimization';

-- Create a separate warehouse for ML training (can be larger when needed)
CREATE WAREHOUSE IF NOT EXISTS ML_TRAINING_WH
    WITH 
    WAREHOUSE_SIZE = 'X-SMALL'    -- Larger for ML training workloads
    AUTO_SUSPEND = 60           -- Quick suspend for cost control
    AUTO_RESUME = TRUE
    COMMENT = 'Dedicated warehouse for ML model training and batch predictions';

-- Set default warehouse
USE WAREHOUSE COMPUTE_WH;

-- Create file formats for different data types
USE SCHEMA RAW;

-- Parquet format for nflverse data
CREATE OR REPLACE FILE FORMAT parquet_format
    TYPE = 'PARQUET'
    COMPRESSION = 'AUTO'
    COMMENT = 'File format for nflverse parquet files from GitHub releases';

-- CSV format for additional data sources
CREATE OR REPLACE FILE FORMAT csv_format 
    TYPE = 'CSV'
    FIELD_DELIMITER = ','
    SKIP_HEADER = 1
    NULL_IF = ('NULL', 'null', '', 'NA', 'n/a')
    EMPTY_FIELD_AS_NULL = TRUE
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    COMMENT = 'File format for CSV data sources';

-- JSON format for configuration and metadata
CREATE OR REPLACE FILE FORMAT json_format
    TYPE = 'JSON'
    COMPRESSION = 'AUTO'
    COMMENT = 'File format for JSON configuration and metadata files';

-- Create native tables in RAW.NFLVERSE schema for bulk-loaded data
-- This follows the design where Python TrainingDataLoader loads data as native tables
CREATE SCHEMA IF NOT EXISTS RAW.NFLVERSE
    COMMENT = 'Native Snowflake tables loaded via Python TrainingDataLoader from nflverse GitHub';

USE SCHEMA RAW.NFLVERSE;

-- NOTE: RAW.NFLVERSE tables are created dynamically by the TrainingDataLoader
-- 
-- The TrainingDataLoader will create the following tables based on actual nflverse data schemas:
-- 1. play_by_play - from https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet
-- 2. player_summary_stats - from https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_regpost_{year}.parquet  
-- 3. team_summary_stats - from https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_regpost_{year}.parquet
-- 4. rosters - from https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{year}.parquet
-- 5. play_by_play_participation - from https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{year}.parquet
-- 6. injuries - from https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{year}.parquet
--
-- This approach ensures:
-- - Exact column names from source data (no assumptions)
-- - Automatic handling of schema changes in nflverse data
-- - All columns stored as VARCHAR for maximum flexibility
-- - Metadata columns added: source_year, load_type, loaded_at
--
-- To create these tables, run: python src/ingestion/training_data_loader.py

-- Create prediction tables in ML schema
USE SCHEMA ML;

CREATE OR REPLACE TABLE predictions (
    prediction_id VARCHAR(50) PRIMARY KEY,
    prediction_type VARCHAR(20) NOT NULL, -- 'player_stat', 'spread', 'total'
    game_id VARCHAR(50),
    player_id VARCHAR(50),
    prediction_value FLOAT,
    confidence_interval_lower FLOAT,
    confidence_interval_upper FLOAT,
    model_version VARCHAR(20),
    features_used VARIANT, -- JSON object with feature values
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    game_date DATE,
    actual_value FLOAT, -- Populated after game completion
    prediction_error FLOAT -- Calculated after game completion
)
COMMENT = 'ML model predictions with confidence intervals and performance tracking';

CREATE OR REPLACE TABLE model_artifacts (
    model_id VARCHAR(50) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'player_stat', 'spread', 'total'
    model_version VARCHAR(20) NOT NULL,
    training_data_hash VARCHAR(64), -- Hash of training data for reproducibility
    hyperparameters VARIANT, -- JSON object with model hyperparameters
    feature_importance VARIANT, -- JSON object with feature importance scores
    performance_metrics VARIANT, -- JSON object with validation metrics
    model_artifact_path VARCHAR(500), -- Path to serialized model file
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    is_active BOOLEAN DEFAULT TRUE
)
COMMENT = 'ML model artifacts and metadata for version control and reproducibility';

CREATE OR REPLACE TABLE model_performance (
    performance_id VARCHAR(50) PRIMARY KEY,
    model_id VARCHAR(50) NOT NULL,
    evaluation_date DATE NOT NULL,
    evaluation_period_start DATE,
    evaluation_period_end DATE,
    accuracy_metrics VARIANT, -- JSON with accuracy, precision, recall, etc.
    prediction_count INTEGER,
    avg_prediction_error FLOAT,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    FOREIGN KEY (model_id) REFERENCES model_artifacts(model_id)
)
COMMENT = 'Model performance tracking over time for drift detection';

-- Create cost monitoring view
CREATE OR REPLACE VIEW cost_monitoring AS
SELECT 
    DATE(start_time) as query_date,
    warehouse_name,
    COUNT(*) as query_count,
    SUM(credits_used) as daily_credits,
    SUM(credits_used) * 2 as estimated_daily_cost_usd, -- Approximate cost per credit
    AVG(execution_time) / 1000 as avg_execution_seconds
FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY 
WHERE start_time >= CURRENT_DATE() - 30
  AND warehouse_name IN ('COMPUTE_WH', 'ML_TRAINING_WH')
GROUP BY query_date, warehouse_name
ORDER BY query_date DESC, warehouse_name
COMMENT = 'Daily cost and performance monitoring for NFL prediction system warehouses';

-- Create data quality monitoring view
USE SCHEMA RAW.NFLVERSE;

CREATE OR REPLACE VIEW data_freshness_check AS
SELECT 
    'play_by_play' as table_name,
    COUNT(*) as total_rows,
    MAX(TRY_CAST(loaded_at AS TIMESTAMP_NTZ)) as last_loaded,
    COUNT(DISTINCT TRY_CAST(season AS INTEGER)) as seasons_available,
    MAX(TRY_CAST(season AS INTEGER)) as latest_season,
    MAX(TRY_CAST(week AS INTEGER)) as latest_week
FROM play_by_play
UNION ALL
SELECT 
    'player_summary_stats' as table_name,
    COUNT(*) as total_rows,
    MAX(TRY_CAST(loaded_at AS TIMESTAMP_NTZ)) as last_loaded,
    COUNT(DISTINCT TRY_CAST(season AS INTEGER)) as seasons_available,
    MAX(TRY_CAST(season AS INTEGER)) as latest_season,
    MAX(TRY_CAST(week AS INTEGER)) as latest_week
FROM player_summary_stats
UNION ALL
SELECT 
    'team_summary_stats' as table_name,
    COUNT(*) as total_rows,
    MAX(TRY_CAST(loaded_at AS TIMESTAMP_NTZ)) as last_loaded,
    COUNT(DISTINCT TRY_CAST(season AS INTEGER)) as seasons_available,
    MAX(TRY_CAST(season AS INTEGER)) as latest_season,
    MAX(TRY_CAST(week AS INTEGER)) as latest_week
FROM team_summary_stats
UNION ALL
SELECT 
    'rosters' as table_name,
    COUNT(*) as total_rows,
    MAX(TRY_CAST(loaded_at AS TIMESTAMP_NTZ)) as last_loaded,
    COUNT(DISTINCT TRY_CAST(season AS INTEGER)) as seasons_available,
    MAX(TRY_CAST(season AS INTEGER)) as latest_season,
    MAX(TRY_CAST(week AS INTEGER)) as latest_week
FROM rosters
COMMENT = 'Data freshness and completeness monitoring for all nflverse tables';

-- Grant appropriate permissions
-- Note: Using ACCOUNTADMIN role which has all permissions
-- In production, you would create specific roles with limited permissions

-- Create a read-only role for analytics
CREATE ROLE IF NOT EXISTS NFL_ANALYTICS_READER;
GRANT USAGE ON DATABASE NFL_BATCHGINEERING TO ROLE NFL_ANALYTICS_READER;
GRANT USAGE ON ALL SCHEMAS IN DATABASE NFL_BATCHGINEERING TO ROLE NFL_ANALYTICS_READER;
GRANT SELECT ON ALL TABLES IN DATABASE NFL_BATCHGINEERING TO ROLE NFL_ANALYTICS_READER;
GRANT SELECT ON ALL VIEWS IN DATABASE NFL_BATCHGINEERING TO ROLE NFL_ANALYTICS_READER;
GRANT USAGE ON WAREHOUSE COMPUTE_WH TO ROLE NFL_ANALYTICS_READER;

-- Create a data engineer role for ETL operations
CREATE ROLE IF NOT EXISTS NFL_DATA_ENGINEER;
GRANT USAGE ON DATABASE NFL_BATCHGINEERING TO ROLE NFL_DATA_ENGINEER;
GRANT USAGE ON ALL SCHEMAS IN DATABASE NFL_BATCHGINEERING TO ROLE NFL_DATA_ENGINEER;
GRANT ALL PRIVILEGES ON ALL TABLES IN DATABASE NFL_BATCHGINEERING TO ROLE NFL_DATA_ENGINEER;
GRANT ALL PRIVILEGES ON ALL VIEWS IN DATABASE NFL_BATCHGINEERING TO ROLE NFL_DATA_ENGINEER;
GRANT USAGE ON WAREHOUSE COMPUTE_WH TO ROLE NFL_DATA_ENGINEER;
GRANT USAGE ON WAREHOUSE ML_TRAINING_WH TO ROLE NFL_DATA_ENGINEER;

-- Set up resource monitors for cost control
CREATE RESOURCE MONITOR IF NOT EXISTS NFL_DAILY_MONITOR
    WITH 
    CREDIT_QUOTA = 10 -- $20 daily limit at $2 per credit
    FREQUENCY = DAILY
    START_TIMESTAMP = IMMEDIATELY
    TRIGGERS 
        ON 75 PERCENT DO NOTIFY
        ON 90 PERCENT DO SUSPEND_IMMEDIATE
        ON 100 PERCENT DO SUSPEND_IMMEDIATE;

-- Apply resource monitor to warehouses
ALTER WAREHOUSE COMPUTE_WH SET RESOURCE_MONITOR = NFL_DAILY_MONITOR;
ALTER WAREHOUSE ML_TRAINING_WH SET RESOURCE_MONITOR = NFL_DAILY_MONITOR;

-- Display setup summary
SELECT 'Database Setup Complete' as status;

-- Show created schemas
SHOW SCHEMAS IN DATABASE NFL_BATCHGINEERING;

-- Show created warehouses
SHOW WAREHOUSES LIKE '%WH';

-- Show created tables in RAW.NFLVERSE
USE SCHEMA RAW.NFLVERSE;
SHOW TABLES;

-- Show resource monitors
SHOW RESOURCE MONITORS;

-- Display cost monitoring information
USE SCHEMA ML;
SELECT * FROM cost_monitoring WHERE query_date >= CURRENT_DATE() - 7;

-- Display data freshness check
USE SCHEMA RAW.NFLVERSE;
SELECT * FROM data_freshness_check;

COMMIT;