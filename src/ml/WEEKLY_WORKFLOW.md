# Weekly NFL Prediction Workflow

Step-by-step guide for generating predictions each week.

## Prerequisites

1. Trained models in `src/ml/models/ensemble/`
2. Access to Snowflake (configured in `.env`)
3. Python environment activated with required packages

## Weekly Steps

### Step 1: Update Schedule Data (Tuesday)

After Monday Night Football, update the schedule data in Snowflake:

```bash
# Load latest schedule from nflverse
python src/ingestion/training_data_loader.py --datasets schedules

# Verify data loaded
# In Snowflake: SELECT COUNT(*) FROM raw.nflverse.schedules
```

### Step 2: Update dbt Models (Tuesday)

Refresh the dbt models to process the new schedule data:

```bash
cd dbt_project

# Refresh staging and marts for upcoming games
dbt run --select stgnv_schedules int_upcoming_games mart_upcoming_game_predictions

# Verify upcoming games exist
# In Snowflake:
# SELECT * FROM analytics.mart_upcoming_game_predictions
# WHERE season = 2025 AND week = 5
```

### Step 3: Create Vegas Lines File (Wednesday)

When Vegas lines are published (usually Tuesday/Wednesday), create or update the Vegas lines CSV:

**File: `data/vegas_lines.csv`**

```csv
game_id,vegas_spread,vegas_total
2025_05_KC_BUF,-3.5,48.5
2025_05_SF_SEA,-2.0,45.0
2025_05_PHI_DAL,1.5,47.0
```

**Notes:**
- `game_id` format: `YYYY_WW_AWAY_HOME` (e.g., `2025_05_KC_BUF`)
- `vegas_spread`: Negative = home team favored
- `vegas_total`: Over/under line
- This file overrides any outdated lines from the nflverse schedule

### Step 4: Generate Predictions (Wednesday/Thursday)

Run the prediction script:

```bash
# Basic usage
python src/ml/predict.py --week 5 --season 2025 --output predictions_week5.csv

# With Vegas lines file
python src/ml/predict.py --week 5 --season 2025 \
    --vegas-file data/vegas_lines.csv \
    --output predictions_week5.csv

# Also write to Snowflake for web UI access
python src/ml/predict.py --week 5 --season 2025 \
    --vegas-file data/vegas_lines.csv \
    --output predictions_week5.csv \
    --snowflake
```

### Step 5: Review Predictions

**Option A: Open in Excel**
- Open `predictions_week5.csv` in Excel
- Sort by `abs(edge)` descending to see best opportunities
- Filter for `bet_recommendation != "NO BET"`

**Option B: Query in Snowflake**
```sql
SELECT *
FROM ml.predictions
WHERE week = 5 AND season = 2025
ORDER BY ABS(edge) DESC;
```

**Option C: Upload to Google Sheets**
- Upload CSV to Google Sheets
- Share with team for collaborative review

## Understanding the Output

| Column | Description |
|--------|-------------|
| `gameday` | Date of the game |
| `home_team` | Home team abbreviation |
| `away_team` | Away team abbreviation |
| `vegas_spread` | Current Vegas line (negative = home favored) |
| `predicted_spread` | Model's predicted spread |
| `edge` | Model spread - Vegas spread (positive = model likes home more) |
| `home_win_prob` | Model's probability home team wins |
| `bet_recommendation` | BET HOME / BET AWAY / NO BET |
| `confidence` | High / Medium / Low |

## Betting Guidelines

1. **Only bet when `bet_recommendation` is BET HOME or BET AWAY**
   - These games have edge >= 3 points

2. **Consider confidence level**
   - High: edge >= 5 points
   - Medium: edge 3-5 points
   - Low: edge < 3 points

3. **Recommended bet sizing (Kelly fraction)**
   - High confidence: 2-3% of bankroll
   - Medium confidence: 1-2% of bankroll
   - Never bet more than 3% on any single game

## Troubleshooting

### "No upcoming games found"
- Make sure you ran `dbt run` to refresh the `mart_upcoming_game_predictions` view
- Check that the schedule data was loaded: `SELECT * FROM raw.nflverse.schedules WHERE season = 2025`

### "No models loaded"
- Verify models exist in `src/ml/models/ensemble/`
- Run training if needed: `python src/ml/train_ensemble.py`

### "Missing features"
- Some features may not be available early in the season
- The model handles this by using 0 for missing features
- Predictions are less reliable in weeks 1-3

### Snowflake connection error
- Check `.env` file has correct credentials
- Verify key pair authentication is configured

## Quick Reference

```bash
# Full weekly workflow
python src/ingestion/training_data_loader.py --datasets schedules
cd dbt_project && dbt run --select stgnv_schedules int_upcoming_games mart_upcoming_game_predictions
cd ..
python src/ml/predict.py --week 5 --season 2025 --vegas-file data/vegas_lines.csv --output predictions.csv --snowflake
```
