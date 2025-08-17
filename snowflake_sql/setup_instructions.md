# Snowflake Database Setup Instructions

## Overview
This document provides step-by-step instructions to complete the Snowflake database setup for the NFL Prediction System, implementing the medallion architecture with proper schemas and configurations.

## Prerequisites
1. Snowflake account with ACCOUNTADMIN privileges
2. Environment variables configured in `.env` file:
   - `SNOWFLAKE_ACCOUNT`
   - `SNOWFLAKE_USER` 
   - `SNOWFLAKE_PASSWORD`
   - `SNOWFLAKE_WAREHOUSE`
   - `SNOWFLAKE_ROLE`

## Setup Steps

### Step 1: Execute Database Setup SQL
Run the complete database setup script in Snowflake:

```bash
# Option 1: Using Python script (recommended)
python src/setup/database_setup.py

# Option 2: Manual execution in Snowflake Web UI
# Copy and paste the contents of snowflake_sql/complete_database_setup.sql
# into the Snowflake worksheet and execute
```

### Step 2: Verify Setup
The setup script will automatically validate the configuration. You should see:

#### ✅ Required Schemas Created:
- `RAW` - Bronze layer for raw nflverse data
- `STAGING` - Silver layer for cleaned data  
- `INTERMEDIATE` - Gold layer for feature engineering
- `MARTS` - Platinum layer for ML-ready datasets
- `ML` - Predictions and model artifacts

#### ✅ Required Warehouses Created:
- `COMPUTE_WH` - Primary warehouse (X-SMALL, auto-suspend 60s)
- `ML_TRAINING_WH` - ML training warehouse (SMALL, auto-suspend 60s)

#### ✅ RAW.NFLVERSE Schema Created:
The schema is ready for the TrainingDataLoader to create tables dynamically:
- `play_by_play` - Created when TrainingDataLoader runs
- `player_summary_stats` - Created when TrainingDataLoader runs  
- `team_summary_stats` - Created when TrainingDataLoader runs
- `rosters` - Created when TrainingDataLoader runs
- `play_by_play_participation` - Created when TrainingDataLoader runs
- `injuries` - Created when TrainingDataLoader runs

**Note**: Tables are created dynamically by the TrainingDataLoader based on actual nflverse parquet file schemas. This ensures exact column names and compatibility with schema changes.

#### ✅ Required Tables in ML Schema:
- `predictions` - Model predictions with confidence intervals
- `model_artifacts` - Model metadata and versioning
- `model_performance` - Model performance tracking

#### ✅ File Formats Created:
- `parquet_format` - For nflverse parquet files
- `csv_format` - For CSV data sources
- `json_format` - For JSON configuration files

#### ✅ Cost Control Features:
- Resource monitor `NFL_DAILY_MONITOR` with $20 daily limit
- Auto-suspend warehouses after 60 seconds of inactivity
- Cost monitoring view for tracking daily expenses

## Architecture Details

### Medallion Architecture Implementation
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────┐
│   Python ETL    │    │   RAW.NFLVERSE   │    │    STAGING      │    │ INTERMEDIATE │
│ TrainingData    │───▶│  (Bronze Layer)  │───▶│ (Silver Layer)  │───▶│ (Gold Layer) │
│    Loader       │    │ All VARCHAR cols │    │ Typed & Cleaned │    │   Features   │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────┘
                                                                              │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐           │
│   ML Models     │◀───│       ML         │◀───│     MARTS       │◀──────────┘
│  Predictions    │    │   Predictions    │    │ (Platinum Layer)│
│   & Artifacts   │    │   & Artifacts    │    │  ML-Ready Data  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Data Flow
1. **Python TrainingDataLoader** downloads nflverse data and loads to `RAW.NFLVERSE` tables
   - Downloads actual parquet files from nflverse GitHub releases
   - Dynamically creates table schemas based on actual column names and types
   - Converts all columns to VARCHAR for maximum flexibility
   - Adds metadata columns: `source_year`, `load_type`, `loaded_at`
2. **dbt staging models** clean and type-convert data from RAW to STAGING (views)
3. **dbt intermediate models** create features and rolling statistics (selective materialization)
4. **dbt marts models** create ML-ready datasets (tables)
5. **ML models** train on marts data and store predictions in ML schema

### Dynamic Schema Approach
The database setup creates placeholder tables, but the **actual schemas are determined by the real nflverse data**:

- **Flexibility**: Handles schema changes in nflverse data automatically
- **Accuracy**: Uses exact column names from source parquet files
- **Simplicity**: All columns stored as VARCHAR initially, typed in staging layer
- **Reliability**: No hardcoded assumptions about column names or types

The TrainingDataLoader will:
1. Download parquet files from URLs like: `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2024.parquet`
2. Inspect the actual column structure
3. Create/update Snowflake tables with the real schema
4. Load data using `MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE` for robustness

### Cost Optimization Features
- **Auto-suspend warehouses**: Suspend after 60 seconds of inactivity
- **Resource monitors**: Daily $20 limit with alerts at 75% and suspension at 90%
- **Selective materialization**: Views for simple logic, tables for complex calculations
- **Right-sized warehouses**: X-SMALL for regular queries, SMALL for ML training

## Validation Queries

After setup, run these queries to verify everything is working:

```sql
-- Check all schemas exist
SHOW SCHEMAS IN DATABASE NFL_BATCHGINEERING;

-- Check warehouses are configured correctly
SHOW WAREHOUSES LIKE '%WH';

-- Check RAW schema is ready (tables created by TrainingDataLoader)
USE SCHEMA RAW.NFLVERSE;
SHOW TABLES; -- Will be empty until TrainingDataLoader runs

-- Check ML tables are created  
USE SCHEMA ML;
SHOW TABLES;

-- Check file formats
USE SCHEMA RAW;
SHOW FILE FORMATS;

-- Check resource monitors
SHOW RESOURCE MONITORS;

-- Check cost monitoring (will be empty until queries are run)
USE SCHEMA ML;
SELECT * FROM cost_monitoring WHERE query_date >= CURRENT_DATE() - 7;

-- Check data freshness (will show 0 rows until data is loaded)
USE SCHEMA RAW.NFLVERSE;
SELECT * FROM data_freshness_check;
```

## Next Steps

After completing the database setup:

1. **Run TrainingDataLoader**: `python src/ingestion/training_data_loader.py` to create and populate RAW tables
2. **Task 2**: Enhance data ingestion pipeline with TrainingDataLoader
3. **Task 3**: Initialize dbt project with proper configuration  
4. **Task 4**: Build staging models for data standardization

## Troubleshooting

### Common Issues

#### Issue: "Database does not exist"
**Solution**: Ensure you have ACCOUNTADMIN privileges and the database name is correct

#### Issue: "Insufficient privileges"
**Solution**: Check that your user has ACCOUNTADMIN role or appropriate permissions

#### Issue: "Warehouse not found"
**Solution**: Verify warehouse names in the setup script match your Snowflake configuration

#### Issue: Resource monitor errors
**Solution**: Resource monitors require ACCOUNTADMIN privileges. If you don't have these, comment out the resource monitor sections.

### Performance Optimization

If queries are slow:
1. Check warehouse size - scale up if needed
2. Verify auto-suspend is working to control costs
3. Monitor query performance in `cost_monitoring` view
4. Consider clustering keys for large tables (implement in later tasks)

### Cost Management

Monitor costs daily:
```sql
-- Daily cost check
SELECT * FROM ML.cost_monitoring 
WHERE query_date >= CURRENT_DATE() - 1
ORDER BY estimated_daily_cost_usd DESC;
```

Set up alerts if costs exceed budget:
- 75% of daily budget: Email notification
- 90% of daily budget: Suspend warehouse
- 100% of daily budget: Suspend all warehouses

## Requirements Satisfied

This setup satisfies the following requirements from the specification:

- **Requirement 1.1**: Scalable and cost-effective data infrastructure with Snowflake database and proper schema structure
- **Requirement 1.4**: Selective materialization strategy (views for simple logic, tables for complex calculations)

The medallion architecture provides a solid foundation for the remaining tasks in the implementation plan.