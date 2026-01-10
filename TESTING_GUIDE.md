# Testing and Validation Guide for NFL Batchgineering

This guide walks you through testing all intermediate and marts models in Snowflake.

## Prerequisites

Before testing, ensure:
1. ✅ Snowflake database is set up and accessible
2. ✅ dbt profile is configured (`~/.dbt/profiles.yml`)
3. ✅ Raw data has been loaded via `src/ingestion/training_data_loader.py`
4. ✅ All staging models (`1_staging/nflverse/`) are built and passing tests

## Testing Strategy

We'll test in order of dependencies:
1. **Intermediate Layer** (depends on staging)
2. **Marts Layer** (depends on intermediate)

---

## Phase 1: Test Intermediate Models

### Step 1.1: Build Intermediate Models

Navigate to your dbt project directory:
```bash
cd dbt_project
```

Build only the intermediate models:
```bash
dbt run --select 2_intermediate
```

**Expected Output:**
```
Completed successfully

Done. PASS=4 WARN=0 ERROR=0 SKIP=0 TOTAL=4
```

**Models that should build:**
- `int_plays_cleaned` (view)
- `int_team_offensive_metrics` (table)
- `int_team_defensive_strength` (table)
- `int_situational_efficiency` (table)

### Step 1.2: Run Data Quality Tests

Run all tests on intermediate models:
```bash
dbt test --select 2_intermediate
```

**Expected Output:**
```
Completed successfully

Done. PASS=X WARN=0 ERROR=0 SKIP=0 TOTAL=X
```

### Step 1.3: Manual Validation Queries

Run these queries in Snowflake to validate data quality:

#### Query 1: Check row counts and coverage
```sql
-- Check int_plays_cleaned coverage
SELECT
    season,
    COUNT(*) as play_count,
    COUNT(DISTINCT game_id) as game_count,
    COUNT(DISTINCT week) as week_count,
    MIN(week) as first_week,
    MAX(week) as last_week
FROM intermediate.int_plays_cleaned
GROUP BY season
ORDER BY season DESC
LIMIT 5;
```

**What to look for:**
- Seasons 2010-2024 present
- ~40,000-50,000 plays per season
- ~250-280 games per season
- Weeks 1-18 covered

#### Query 2: Validate EPA metrics are reasonable
```sql
-- Check offensive EPA metrics are in reasonable ranges
SELECT
    team,
    season,
    week,
    rolling_4wk_epa_per_play,
    rolling_4wk_success_rate,
    rolling_4wk_explosive_rate,
    total_plays
FROM intermediate.int_team_offensive_metrics
WHERE season = 2023
    AND week = 10
ORDER BY rolling_4wk_epa_per_play DESC
LIMIT 10;
```

**What to look for:**
- EPA per play typically between -0.3 and +0.3
- Success rates between 0.35 and 0.55
- Explosive rates between 0.10 and 0.25
- Total plays > 30 per week

#### Query 3: Check defensive metrics
```sql
-- Top defenses by EPA allowed (lower is better)
SELECT
    team,
    season,
    week,
    rolling_4wk_epa_allowed,
    defensive_epa_rank,
    rolling_4wk_sack_rate,
    rolling_4wk_forced_turnover_rate
FROM intermediate.int_team_defensive_strength
WHERE season = 2023
    AND week = 10
ORDER BY defensive_epa_rank ASC
LIMIT 10;
```

**What to look for:**
- EPA allowed typically between -0.3 and +0.3 (negative is better for defense)
- Defensive ranks 1-32
- Sack rates between 0.05 and 0.15
- Forced turnover rates between 0.01 and 0.05

#### Query 4: Check for NULL values in critical columns
```sql
-- Should return 0 nulls for key metrics
SELECT
    'int_plays_cleaned' as model,
    COUNT(*) as total_rows,
    SUM(CASE WHEN epa IS NULL THEN 1 ELSE 0 END) as null_epa,
    SUM(CASE WHEN posteam IS NULL THEN 1 ELSE 0 END) as null_posteam
FROM intermediate.int_plays_cleaned
WHERE season >= 2020

UNION ALL

SELECT
    'int_team_offensive_metrics',
    COUNT(*),
    SUM(CASE WHEN rolling_4wk_epa_per_play IS NULL THEN 1 ELSE 0 END),
    SUM(CASE WHEN team IS NULL THEN 1 ELSE 0 END)
FROM intermediate.int_team_offensive_metrics
WHERE season >= 2020

UNION ALL

SELECT
    'int_team_defensive_strength',
    COUNT(*),
    SUM(CASE WHEN rolling_4wk_epa_allowed IS NULL THEN 1 ELSE 0 END),
    SUM(CASE WHEN team IS NULL THEN 1 ELSE 0 END)
FROM intermediate.int_team_defensive_strength
WHERE season >= 2020;
```

**What to look for:**
- null_posteam / null_team should be 0
- null_epa can have some nulls in early weeks (insufficient rolling window)

### Step 1.4: Troubleshooting Intermediate Layer

**Common Issues:**

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `Object does not exist: stgnv_play_by_play` | Staging models not built | Run `dbt run --select 1_staging` |
| `Invalid identifier: first_down_rush` | Column name mismatch | Check staging model has this column |
| EPA values all NULL | Data type casting issue | Check CAST() statements in int_plays_cleaned |
| No rows after week 4 filter | Insufficient data | Check raw data loaded for multiple seasons |

---

## Phase 2: Test Marts Models

### Step 2.1: Build Marts Models

Build all marts models:
```bash
dbt run --select 3_marts
```

**Expected Output:**
```
Completed successfully

Done. PASS=4 WARN=0 ERROR=0 SKIP=0 TOTAL=4
```

**Models that should build:**
- `mart_predictive_features` (table)
- `mart_game_prediction_features` (table)
- `mart_weather_impact_metrics` (table)
- `mart_model_validation` (table)

### Step 2.2: Run Data Quality Tests

```bash
dbt test --select 3_marts
```

### Step 2.3: Manual Validation Queries

#### Query 1: Check mart_predictive_features
```sql
-- Validate predictive features are populated
SELECT
    team,
    season,
    week,
    epa_per_play_l4w,
    success_rate_l4w,
    efficiency_tier,
    season_def_epa_allowed,
    def_strength_tier
FROM marts.mart_predictive_features
WHERE season = 2023
    AND week = 10
ORDER BY epa_per_play_l4w DESC
LIMIT 10;
```

**What to look for:**
- All columns populated (no NULLs)
- Efficiency tiers: 'Elite', 'Good', 'Average', 'Poor'
- Def strength tiers: same values

#### Query 2: Check game prediction features
```sql
-- Validate game-level feature construction
SELECT
    game_id,
    season,
    week,
    home_team,
    away_team,
    home_score,
    away_score,
    home_epa_l4w,
    away_epa_l4w,
    temp,
    wind,
    div_game,
    playoff
FROM marts.mart_game_prediction_features
WHERE season = 2023
ORDER BY week DESC, game_id
LIMIT 20;
```

**What to look for:**
- One row per game
- Both home and away features populated
- Scores present (these are actual outcomes)
- Weather data reasonable (temp 0-100, wind 0-40)
- div_game is 0 or 1

#### Query 3: Check weather impact metrics
```sql
-- Validate weather impact calculations
SELECT
    team,
    season,
    epa_good_weather,
    epa_adverse_weather,
    epa_weather_impact,
    games_good_weather,
    games_adverse_weather,
    weather_resistance_tier,
    weather_adjustment_confidence
FROM marts.mart_weather_impact_metrics
WHERE season = 2023
    AND games_adverse_weather >= 2  -- Sufficient sample
ORDER BY epa_weather_impact DESC
LIMIT 10;
```

**What to look for:**
- epa_good_weather typically higher than epa_adverse_weather
- Positive epa_weather_impact means team performs worse in bad weather
- Weather resistance tier should reflect impact magnitude
- Confidence should match sample size (High = 4+ games)

#### Query 4: Count games by season
```sql
-- Ensure we have full game coverage
SELECT
    season,
    COUNT(DISTINCT game_id) as game_count,
    COUNT(DISTINCT home_team) as teams,
    MIN(week) as first_week,
    MAX(week) as last_week
FROM marts.mart_game_prediction_features
GROUP BY season
ORDER BY season DESC;
```

**What to look for:**
- ~256 games per regular season
- 32 teams
- Weeks 1-18 (regular season)

### Step 2.4: Cross-Model Validation

Validate that features join correctly across models:

```sql
-- Join predictive features to game features
SELECT
    gpf.game_id,
    gpf.home_team,
    gpf.away_team,
    gpf.home_epa_l4w as game_home_epa,
    pf_home.epa_per_play_l4w as predictive_home_epa,
    gpf.away_epa_l4w as game_away_epa,
    pf_away.epa_per_play_l4w as predictive_away_epa
FROM marts.mart_game_prediction_features gpf
LEFT JOIN marts.mart_predictive_features pf_home
    ON gpf.home_team = pf_home.team
    AND gpf.season = pf_home.season
    AND gpf.week = pf_home.week
LEFT JOIN marts.mart_predictive_features pf_away
    ON gpf.away_team = pf_away.team
    AND gpf.season = pf_away.season
    AND gpf.week = pf_away.week
WHERE gpf.season = 2023
    AND gpf.week = 10
LIMIT 10;
```

**What to look for:**
- game_home_epa should roughly match predictive_home_epa
- No NULLs in predictive columns (join should succeed)

### Step 2.5: Troubleshooting Marts Layer

**Common Issues:**

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `Object does not exist: int_team_offensive_metrics` | Intermediate not built | Run `dbt run --select 2_intermediate` |
| Many NULL values in home/away features | Join failure | Check team names match exactly between models |
| `group by` error in game_schedule CTE | GROUP BY columns incorrect | Verify game_date is included in GROUP BY |
| Weather metrics all NULL | No weather data in source | Check stgnv_play_by_play has temp/wind columns |

---

## Phase 3: Full Pipeline Test

### Step 3.1: Run Everything in Order

Test the full dependency chain:
```bash
# Build and test everything
dbt build --select 1_staging+ 2_intermediate+ 3_marts+
```

This will:
1. Build staging models
2. Run tests on staging
3. Build intermediate models
4. Run tests on intermediate
5. Build marts models
6. Run tests on marts

### Step 3.2: Generate Documentation

Generate and serve dbt docs:
```bash
dbt docs generate
dbt docs serve
```

Open http://localhost:8080 in your browser to view:
- Lineage graphs showing dependencies
- Column-level documentation
- Test results
- Model metadata

---

## Success Criteria Checklist

Before proceeding to Task 3 (ML pipeline), verify:

### Intermediate Layer ✓
- [ ] All 4 intermediate models build successfully
- [ ] All dbt tests pass
- [ ] EPA metrics are in reasonable ranges (-0.3 to 0.3)
- [ ] Success rates between 0.35 and 0.55
- [ ] Rolling windows calculate correctly (NULL in early weeks is OK)
- [ ] All 32 teams present in each week

### Marts Layer ✓
- [ ] All 4 marts models build successfully
- [ ] All dbt tests pass
- [ ] `mart_predictive_features`: All tiers populated correctly
- [ ] `mart_game_prediction_features`: One row per game, both teams have features
- [ ] `mart_weather_impact_metrics`: Weather impact calculations reasonable
- [ ] Joins between models work correctly (no orphaned records)

### Data Quality ✓
- [ ] No unexpected NULL values in key columns
- [ ] Row counts match expectations (~250 games/season)
- [ ] Features align across models (game features match predictive features)
- [ ] All seasons from 2010-2024 represented

---

## Next Steps

Once all tests pass, you're ready to proceed to **Task 3: ML Training Pipeline**.

The validated marts data will be the input for training XGBoost models to predict:
- Game spreads (point differential)
- Game totals (combined score)

If you encounter any errors during testing, refer to the troubleshooting sections above or review the dbt logs:
```bash
# View detailed logs
cat logs/dbt.log

# Run with debug output
dbt run --select model_name --debug
```
