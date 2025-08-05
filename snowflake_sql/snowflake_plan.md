# Hybrid Approach: External Tables + Selective Materialization

## **Strategy Overview**

Use **External Tables for raw data access** + **Selective materialization** for performance-critical datasets.

## **Architecture**

```
GitHub nflverse-data
├── External Tables (RAW schema)
│   ├── ext_play_by_play → Direct read from GitHub
│   ├── ext_player_stats → Direct read from GitHub  
│   ├── ext_schedules → Direct read from GitHub
│   └── ext_rosters → Direct read from GitHub
│
├── Staging (VIEW - computed on-demand)
│   ├── stg_play_by_play → Clean external data
│   ├── stg_player_stats → Clean external data
│   └── stg_schedules → Clean external data
│
├── Intermediate (TABLE - materialized for performance)
│   ├── int_player_rolling_stats → 5-game averages, etc.
│   ├── int_team_metrics → Aggregated team performance
│   └── int_matchup_history → Head-to-head analysis
│
└── Marts (TABLE - final ML datasets)
    ├── fct_player_predictions → ML-ready player data
    └── fct_team_predictions → ML-ready team data
```

## **Implementation Plan**

### Phase 1: Set up External Tables (1 hour)

1. **Find nflverse URLs**: Research actual GitHub release URLs
2. **Create external tables**: Point to parquet files on GitHub
3. **Test connectivity**: Verify data accessibility

### Phase 2: dbt Staging Models (1 hour)

Update staging models to use external tables as sources:

```sql
-- models/staging/sources.yml
sources:
  - name: external
    description: External tables pointing to nflverse data
    tables:
      - name: ext_play_by_play
        external: yes
      - name: ext_player_stats  
        external: yes
```

### Phase 3: Performance Optimization (1 hour)

Materialize only performance-critical transformations:

```sql
-- models/intermediate/int_player_rolling_stats.sql
{{ config(materialized='table') }}  -- Materialize for performance

-- Heavy computation - worth materializing
WITH rolling_calculations AS (
  SELECT 
    player_id,
    game_date,
    AVG(fantasy_points) OVER (
      PARTITION BY player_id 
      ORDER BY game_date 
      ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
    ) AS fantasy_points_5game_avg
  FROM {{ ref('stg_player_stats') }}
)
```

## **Cost-Benefit Analysis**

### **External Tables Approach Costs:**
- **Query costs**: ~$2-5/month (reading external data)
- **Materialization costs**: ~$10-15/month (only for complex models)
- **Total**: ~$15-20/month (vs $30-40 with full ingestion)

### **Performance Trade-offs:**
- **Staging queries**: Slower (2-3x) due to network reads
- **Complex transformations**: Fast (materialized tables)
- **ML model training**: Fast (marts are materialized)

## **When to Use Each Approach**

### **✅ Use External Tables For:**
- **Simple transformations**: Column renaming, basic filtering
- **One-time analyses**: Ad-hoc queries, exploration
- **Current season data**: Frequently updated, small volume

### **✅ Materialize Tables For:**
- **Complex calculations**: Rolling averages, window functions
- **Frequently accessed data**: ML training datasets
- **Historical analysis**: 5+ years of data, large volumes

## **URL Discovery Process**

Since I can't find the exact URLs in my search, here's how to find them:

1. **Visit nflverse-data releases**: https://github.com/nflverse/nflverse-data/releases
2. **Find parquet files**: Look for `.parquet` download links
3. **Example URL pattern**: `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2024.parquet`

## **Updated dbt Sources**

```yml
# models/staging/sources.yml
version: 2

sources:
  - name: external
    description: External tables reading from nflverse GitHub
    tables:
      - name: ext_play_by_play
        description: Play-by-play data from GitHub
        external: yes
        columns:
          - name: game_id
            tests:
              - not_null
      - name: ext_player_stats
        description: Player stats from GitHub  
        external: yes
      - name: ext_schedules
        description: Schedules from GitHub
        external: yes
      - name: ext_rosters
        description: Rosters from GitHub
        external: yes
```

## **Modified Staging Models**

```sql
-- models/staging/stg_play_by_play.sql
{{ config(materialized='view') }}  -- Keep as view, read from external

SELECT 
  game_id,
  season,
  week,
  -- Clean and standardize columns
  UPPER(TRIM(posteam)) AS posteam,
  UPPER(TRIM(defteam)) AS defteam,
  -- Convert numeric indicators to booleans
  CASE WHEN rush_attempt = 1 THEN TRUE ELSE FALSE END AS is_rush,
  CASE WHEN pass_attempt = 1 THEN TRUE ELSE FALSE END AS is_pass,
  -- Include source metadata
  source_filename,
  CURRENT_TIMESTAMP() AS processed_at
FROM {{ source('external', 'ext_play_by_play') }}
WHERE season >= {{ var('current_season') - var('lookback_seasons') }}
```

## **Monitoring and Optimization**

### **Query Performance Monitoring**
```sql
-- Check external table query performance
SELECT 
  query_text,
  execution_time,
  bytes_scanned,
  credits_used
FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY 
WHERE query_text LIKE '%ext_%'
  AND start_time >= DATEADD(day, -7, CURRENT_TIMESTAMP())
ORDER BY execution_time DESC;
```

### **Cost Optimization Rules**
1. **Minimize external table scans**: Filter early, select specific columns
2. **Cache frequently used data**: Materialize complex calculations
3. **Use LIMIT for testing**: Avoid full table scans during development

## **Fallback Strategy**

If external tables become too expensive or slow:

1. **Selective ingestion**: Load only recent data (current season)
2. **Hybrid materialization**: External for historical, native for current
3. **Scheduled snapshots**: Daily/weekly copies of key datasets

## **Decision Matrix**

| Factor | External Tables | Python Ingestion |
|--------|----------------|------------------|
| **Setup Complexity** | ⭐⭐⭐⭐⭐ Simple | ⭐⭐⭐ Moderate |
| **Ongoing Maintenance** | ⭐⭐⭐⭐⭐ Minimal | ⭐⭐ High |
| **Query Performance** | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐⭐ Fast |
| **Storage Costs** | ⭐⭐⭐⭐⭐ None | ⭐⭐ Moderate |
| **Compute Costs** | ⭐⭐⭐⭐ Low | ⭐⭐⭐ Higher |
| **Data Freshness** | ⭐⭐⭐⭐⭐ Real-time | ⭐⭐⭐ Batch |
| **Customization** | ⭐⭐ Limited | ⭐⭐⭐⭐⭐ Full |

## **Recommendation**

**Start with External Tables** for your 20-hour timeline:

1. **Week 1**: Set up external tables, basic staging models
2. **Week 2**: Build intermediate models, selectively materialize
3. **Week 3**: Create marts and ML models
4. **Week 4**: Optimize based on performance/cost monitoring

If performance becomes an issue later, you can always migrate specific datasets to native tables without changing your dbt models - just the sources configuration.

This approach gets you running **immediately** with **minimal setup** and **lowest cost**, while preserving the option to optimize later.