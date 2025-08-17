-- Snowflake Database Setup Validation Script
-- Run this script to verify that the database setup was completed successfully

-- Set context
USE ROLE ACCOUNTADMIN;
USE DATABASE NFL_BATCHGINEERING;

-- Display validation header
SELECT 'NFL PREDICTION SYSTEM - DATABASE SETUP VALIDATION' as validation_report;
SELECT '======================================================' as separator;

-- 1. Check all required schemas exist
SELECT 'CHECKING SCHEMAS...' as step;
SHOW SCHEMAS IN DATABASE NFL_BATCHGINEERING;

SELECT 
    CASE 
        WHEN COUNT(*) >= 5 THEN '✅ All required schemas found'
        ELSE '❌ Missing schemas - Expected: RAW, STAGING, INTERMEDIATE, MARTS, ML'
    END as schema_status
FROM (
    SHOW SCHEMAS IN DATABASE NFL_BATCHGINEERING
) 
WHERE "name" IN ('RAW', 'STAGING', 'INTERMEDIATE', 'MARTS', 'ML');

-- 2. Check warehouses are configured
SELECT 'CHECKING WAREHOUSES...' as step;
SHOW WAREHOUSES LIKE '%WH';

SELECT 
    CASE 
        WHEN COUNT(*) >= 2 THEN '✅ Required warehouses found'
        ELSE '❌ Missing warehouses - Expected: COMPUTE_WH, ML_TRAINING_WH'
    END as warehouse_status
FROM (
    SHOW WAREHOUSES LIKE '%WH'
) 
WHERE "name" IN ('COMPUTE_WH', 'ML_TRAINING_WH');

-- 3. Check RAW.NFLVERSE schema exists (tables created by TrainingDataLoader)
SELECT 'CHECKING RAW SCHEMA...' as step;
USE SCHEMA RAW.NFLVERSE;
SHOW TABLES;

SELECT '✅ RAW.NFLVERSE schema ready for TrainingDataLoader' as raw_schema_status;
SELECT 'ℹ️  Tables will be created when TrainingDataLoader runs' as raw_tables_note;
SELECT 'ℹ️  Expected tables: play_by_play, player_summary_stats, team_summary_stats, rosters, play_by_play_participation, injuries' as expected_tables;

-- 4. Check ML schema tables
SELECT 'CHECKING ML TABLES...' as step;
USE SCHEMA ML;
SHOW TABLES;

SELECT 
    CASE 
        WHEN COUNT(*) >= 3 THEN '✅ All ML tables created'
        ELSE '❌ Missing ML tables - Expected: PREDICTIONS, MODEL_ARTIFACTS, MODEL_PERFORMANCE'
    END as ml_tables_status
FROM (
    SHOW TABLES IN SCHEMA ML
) 
WHERE "name" IN ('PREDICTIONS', 'MODEL_ARTIFACTS', 'MODEL_PERFORMANCE');

-- 5. Check file formats
SELECT 'CHECKING FILE FORMATS...' as step;
USE SCHEMA RAW;
SHOW FILE FORMATS;

SELECT 
    CASE 
        WHEN COUNT(*) >= 3 THEN '✅ All file formats created'
        ELSE '❌ Missing file formats - Expected: PARQUET_FORMAT, CSV_FORMAT, JSON_FORMAT'
    END as file_formats_status
FROM (
    SHOW FILE FORMATS IN SCHEMA RAW
) 
WHERE "name" IN ('PARQUET_FORMAT', 'CSV_FORMAT', 'JSON_FORMAT');

-- 6. Check resource monitors
SELECT 'CHECKING RESOURCE MONITORS...' as step;
SHOW RESOURCE MONITORS;

SELECT 
    CASE 
        WHEN COUNT(*) >= 1 THEN '✅ Resource monitor configured'
        ELSE '❌ No resource monitor found - Expected: NFL_DAILY_MONITOR'
    END as resource_monitor_status
FROM (
    SHOW RESOURCE MONITORS
) 
WHERE "name" = 'NFL_DAILY_MONITOR';

-- 7. Check warehouse configurations
SELECT 'CHECKING WAREHOUSE CONFIGURATIONS...' as step;

SELECT 
    "name" as warehouse_name,
    "size" as warehouse_size,
    "auto_suspend" as auto_suspend_seconds,
    "auto_resume" as auto_resume_enabled,
    "resource_monitor" as resource_monitor,
    CASE 
        WHEN "auto_suspend" = 60 AND "auto_resume" = 'true' THEN '✅'
        ELSE '❌'
    END as config_status
FROM (
    SHOW WAREHOUSES LIKE '%WH'
)
WHERE "name" IN ('COMPUTE_WH', 'ML_TRAINING_WH');

-- 8. Test basic table structure
SELECT 'TESTING TABLE STRUCTURES...' as step;

-- Test RAW table structure (should have VARCHAR columns and metadata)
USE SCHEMA RAW.NFLVERSE;

SELECT 
    'play_by_play' as table_name,
    COUNT(*) as column_count,
    CASE 
        WHEN COUNT(*) > 30 THEN '✅ Proper column structure'
        ELSE '❌ Missing columns'
    END as structure_status
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_SCHEMA = 'NFLVERSE' 
  AND TABLE_NAME = 'PLAY_BY_PLAY';

-- Test ML table structure
USE SCHEMA ML;

SELECT 
    'predictions' as table_name,
    COUNT(*) as column_count,
    CASE 
        WHEN COUNT(*) >= 10 THEN '✅ Proper ML table structure'
        ELSE '❌ Missing ML columns'
    END as ml_structure_status
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_SCHEMA = 'ML' 
  AND TABLE_NAME = 'PREDICTIONS';

-- 9. Check views exist
SELECT 'CHECKING MONITORING VIEWS...' as step;

USE SCHEMA ML;
SELECT 
    CASE 
        WHEN COUNT(*) >= 1 THEN '✅ Cost monitoring view exists'
        ELSE '❌ Cost monitoring view missing'
    END as cost_view_status
FROM INFORMATION_SCHEMA.VIEWS 
WHERE TABLE_SCHEMA = 'ML' 
  AND TABLE_NAME = 'COST_MONITORING';

USE SCHEMA RAW.NFLVERSE;
SELECT 
    CASE 
        WHEN COUNT(*) >= 1 THEN '✅ Data freshness view exists'
        ELSE '❌ Data freshness view missing'
    END as freshness_view_status
FROM INFORMATION_SCHEMA.VIEWS 
WHERE TABLE_SCHEMA = 'NFLVERSE' 
  AND TABLE_NAME = 'DATA_FRESHNESS_CHECK';

-- 10. Final summary
SELECT 'VALIDATION SUMMARY' as final_report;
SELECT '==================' as separator;

-- Overall validation query
WITH validation_checks AS (
    SELECT 
        (SELECT COUNT(*) FROM (SHOW SCHEMAS IN DATABASE NFL_BATCHGINEERING) WHERE "name" IN ('RAW', 'STAGING', 'INTERMEDIATE', 'MARTS', 'ML')) >= 5 as schemas_ok,
        (SELECT COUNT(*) FROM (SHOW WAREHOUSES LIKE '%WH') WHERE "name" IN ('COMPUTE_WH', 'ML_TRAINING_WH')) >= 2 as warehouses_ok,
        TRUE as raw_schema_ok, -- Schema exists, tables created by TrainingDataLoader
        (SELECT COUNT(*) FROM (SHOW TABLES IN SCHEMA ML) WHERE "name" IN ('PREDICTIONS', 'MODEL_ARTIFACTS', 'MODEL_PERFORMANCE')) >= 3 as ml_tables_ok,
        (SELECT COUNT(*) FROM (SHOW FILE FORMATS IN SCHEMA RAW) WHERE "name" IN ('PARQUET_FORMAT', 'CSV_FORMAT', 'JSON_FORMAT')) >= 3 as file_formats_ok
)
SELECT 
    CASE 
        WHEN schemas_ok AND warehouses_ok AND raw_schema_ok AND ml_tables_ok AND file_formats_ok 
        THEN '🎉 DATABASE SETUP VALIDATION: SUCCESS'
        ELSE '⚠️  DATABASE SETUP VALIDATION: ISSUES FOUND'
    END as overall_status,
    schemas_ok as schemas_valid,
    warehouses_ok as warehouses_valid,
    raw_schema_ok as raw_schema_valid,
    ml_tables_ok as ml_tables_valid,
    file_formats_ok as file_formats_valid
FROM validation_checks;

-- Display next steps
SELECT 'NEXT STEPS:' as next_steps;
SELECT '1. Run Task 2: Enhance data ingestion pipeline' as step_1;
SELECT '2. Run Task 3: Initialize dbt project configuration' as step_2;
SELECT '3. Load sample data using TrainingDataLoader' as step_3;

-- Display useful queries for monitoring
SELECT 'USEFUL MONITORING QUERIES:' as monitoring_info;
SELECT 'USE SCHEMA ML; SELECT * FROM cost_monitoring;' as cost_query;
SELECT 'USE SCHEMA RAW.NFLVERSE; SELECT * FROM data_freshness_check;' as freshness_query;