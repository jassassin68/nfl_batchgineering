use role ACCOUNTADMIN;

-- Set up Raw database to land raw data in.
CREATE DATABASE RAW;
CREATE SCHEMA RAW.NFLVERSE;

-- Set up Development database to materialize experimental analytical dbt model still in development in.
CREATE DATABASE DEVELOPMENT;

-- Set up Production database to materialize prod analytical dbt model in.
CREATE DATABASE PRODUCTION_ANALYTICS;
CREATE SCHEMA PRODUCTION_ANALYTICS.ANALYTICS;
