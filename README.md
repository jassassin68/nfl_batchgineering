# nfl_batchgineering

An end-to-end NFL game prediction system targeting point spread and totals betting. The pipeline ingests play-by-play and schedule data from nflverse, transforms it through a layered dbt project in Snowflake, trains a hybrid ensemble ML model, and generates weekly predictions — all orchestrated by Dagster.

**Primary Goal**: Achieve >52.4% ATS (against-the-spread) accuracy to overcome standard -110 juice.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Warehouse | Snowflake |
| Data Transformations | dbt (dbt-snowflake) |
| Data Ingestion | Python · Polars · nflverse parquet releases |
| ML Models | XGBoost · PyMC (Bayesian) · PyTorch (Neural Net) · scikit-learn |
| Orchestration | Dagster (assets, jobs, schedules) |
| Code Quality | ruff (lint + format) |
| Logging | loguru |
| Auth | Snowflake key-pair authentication |
| Runtime | Python 3.11+ |

---

## Repository Structure

```
nfl_batchgineering/
├── dagster_project/               # Dagster orchestration definitions
│   ├── __init__.py                # Definitions entry point (assets, jobs, schedules)
│   ├── constants.py               # Shared path constants
│   ├── assets/
│   │   ├── ingestion.py           # nflverse → Snowflake raw assets
│   │   ├── dbt_assets.py          # dbt model assets (auto-generated from manifest)
│   │   ├── ml_training.py         # XGBoost training asset
│   │   └── predictions.py         # Weekly predictions asset
│   ├── jobs/
│   │   └── weekly_pipeline.py     # Full pipeline job definition
│   ├── resources/
│   │   └── dbt_resource.py        # dbt CLI resource configuration
│   └── schedules/
│       └── weekly_schedule.py     # Tuesday 8AM NFL-season schedule
│
├── dbt_project/                   # dbt transformations
│   ├── dbt_project.yml            # Project config (lookback: 10 seasons)
│   ├── models/
│   │   ├── 1_staging/nflverse/    # Cleaned views over raw Snowflake tables
│   │   │   ├── stgnv_play_by_play.sql
│   │   │   ├── stgnv_player_summary_stats.sql
│   │   │   ├── stgnv_team_summary_stats.sql
│   │   │   ├── stgnv_rosters.sql
│   │   │   ├── stgnv_schedules.sql
│   │   │   ├── stgnv_injuries.sql
│   │   │   └── stgnv_play_by_play_participation.sql
│   │   ├── 2_intermediate/        # Feature engineering (views + materialized tables)
│   │   │   ├── int_plays_cleaned.sql
│   │   │   ├── int_team_offensive_metrics.sql     # Rolling 4-week EPA/play
│   │   │   ├── int_team_defensive_strength.sql    # Rolling defensive EPA allowed
│   │   │   ├── int_situational_efficiency.sql     # Red zone, 3rd down, etc.
│   │   │   ├── int_game_vegas_lines.sql           # Opening lines, line movement
│   │   │   └── int_upcoming_games.sql             # Future schedule with context
│   │   └── 3_marts/               # Final ML-ready tables
│   │       ├── mart_predictive_features.sql       # Per-team, per-week feature set
│   │       ├── mart_game_prediction_features.sql  # Historical game rows for training
│   │       ├── mart_upcoming_game_predictions.sql # Upcoming games for inference
│   │       ├── mart_weather_impact_metrics.sql    # Weather-adjusted efficiency
│   │       └── mart_model_validation.sql          # Walk-forward validation dataset
│   ├── macros/
│   └── tests/
│
├── src/
│   ├── ingestion/
│   │   └── training_data_loader.py   # Bulk parquet → Snowflake loader (Polars)
│   └── ml/
│       ├── base.py                    # Abstract BasePredictor interface
│       ├── models/
│       │   ├── spread_predictor.py    # XGBoost spread model
│       │   ├── elo_model.py           # Elo rating baseline
│       │   ├── bayesian.py            # Bayesian state-space model (PyMC)
│       │   ├── neural.py              # Shallow neural network (PyTorch)
│       │   └── ensemble.py            # Ridge stacking meta-learner
│       ├── utils/
│       │   ├── feature_engineering.py # Feature selection and preparation
│       │   ├── evaluation.py          # ATS accuracy, Brier score, RMSE
│       │   └── validation.py          # Walk-forward CV helpers
│       ├── train_spread_model.py      # XGBoost training entry point
│       ├── train_elo_model.py         # Elo model training
│       ├── train_ensemble.py          # Full ensemble training
│       └── predict.py                 # Inference: load models → generate predictions
│
├── models/                        # Serialized model artifacts (.pkl, .json)
├── data/                          # Prediction CSV outputs
├── notebooks/                     # Exploratory analysis
├── snowflake_sql/                 # Ad-hoc SQL scripts
├── logs/                          # Rotating application logs
├── pyproject.toml                 # Project metadata, uv/ruff config
├── requirements.txt               # Python dependencies
├── TESTING_GUIDE.md               # dbt layer validation walkthrough
└── CLAUDE.md                      # AI assistant project instructions
```

---

## Data Pipeline Overview

```
nflverse GitHub Releases (parquet)
         │
         ▼
 TrainingDataLoader (Polars + Snowflake connector)
         │  bulk COPY INTO
         ▼
 Snowflake RAW.NFLVERSE  ──────────────────────────────────────┐
         │                                                       │
         ▼  dbt run                                             │
 1_staging (views)                                              │
  · stgnv_play_by_play                                          │
  · stgnv_schedules, stgnv_rosters                              │
  · stgnv_player/team_summary_stats                             │
  · stgnv_injuries, stgnv_pbp_participation                     │
         │                                                       │
         ▼  dbt run                                             │
 2_intermediate (views + tables)                                │
  · int_plays_cleaned                                           │
  · int_team_offensive_metrics  (rolling EPA/play)              │
  · int_team_defensive_strength (rolling EPA allowed)           │
  · int_situational_efficiency                                  │
  · int_game_vegas_lines                                        │
  · int_upcoming_games                                          │
         │                                                       │
         ▼  dbt run                                             │
 3_marts (tables)                                               │
  · mart_game_prediction_features  ← training data             │
  · mart_predictive_features       ← per-team feature store    │
  · mart_upcoming_game_predictions ← inference input           │
  · mart_model_validation                                       │
  · mart_weather_impact_metrics                                 │
         │                                                       │
         ▼  Python                                              │
 ML Training (Snowflake → Polars → sklearn/XGBoost/PyMC)       │
  · SpreadPredictor (XGBoost)                                   │
  · EloModel (baseline)                                         │
  · BayesianStateSpace (PyMC AR(1))                             │
  · NeuralNetPredictor (PyTorch)                                │
  · StackingEnsemble (Ridge meta-learner)                       │
         │                                                       │
         ▼  Python                                              │
 Weekly Predictions → Snowflake ML schema + CSV output ────────┘
```

---

## ML Model Architecture

The system uses a stacking ensemble of four base models, combined by a Ridge regression meta-learner:

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   XGBoost   │  │  Bayesian   │  │   Neural    │  │    Elo      │
│  Regressor  │  │ State-Space │  │   Network   │  │  Baseline   │
│ (max_depth4)│  │  (PyMC AR1) │  │ (PyTorch,   │  │  (Glickman) │
│ (lr=0.05)   │  │             │  │  ≤3 layers) │  │             │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       └────────────────┴────────────────┴────────────────┘
                              │
                   ┌──────────┴──────────┐
                   │   Ridge Regression  │
                   │   Meta-Learner      │
                   │   (+ uncertainty)   │
                   └──────────┬──────────┘
                              │
                   ┌──────────┴──────────┐
                   │  Betting Decision   │
                   │  (edge ≥ 3 pts →   │
                   │   0.25x Kelly bet) │
                   └─────────────────────┘
```

All models implement the `BasePredictor` abstract interface (`src/ml/base.py`) providing `fit()`, `predict()`, `predict_proba()`, `save_model()`, and `load_model()`.

### Validation Strategy

Walk-forward cross-validation is **required** — random splits are never used:

- Train on seasons `[t-n ... t-1]`, test on season `t`
- Repeat sliding the window forward for each available test season
- Target metrics: ATS Accuracy (>52.4%), Brier Score, RMSE vs Vegas line

---

## Dagster Orchestration

The Dagster project (`dagster_project/`) defines the full pipeline as software-defined assets:

| Asset | Description |
|---|---|
| `raw_nflverse_data` | Loads current-season parquet datasets to Snowflake |
| `raw_schedules` | Loads full schedules file (all seasons, full replace) |
| `nfl_dbt_assets` | Runs dbt models (staging → intermediate → marts) |
| `trained_xgboost_model` | Trains XGBoost spread predictor, saves to `models/` |
| `weekly_predictions` | Loads upcoming games mart, runs ensemble inference, writes to Snowflake + CSV |

**Schedule**: Every Tuesday at 8:00 AM during the NFL season (September–February). The schedule auto-derives the current NFL week and season from the execution date.

To launch the Dagster UI locally:
```bash
dagster dev -m dagster_project
```

---

## Snowflake Schema Structure

```sql
NFL_ANALYTICS.RAW.NFLVERSE       -- Bulk-loaded parquet tables
NFL_ANALYTICS.STAGING            -- dbt views (stgnv_*)
NFL_ANALYTICS.INTERMEDIATE       -- dbt views/tables (int_*)
NFL_ANALYTICS.MARTS              -- dbt materialized tables (mart_*)
NFL_ANALYTICS.ML                 -- Model artifacts and predictions
```

---

## Setup

### Prerequisites
- Python 3.11+
- Snowflake account with key-pair authentication configured
- dbt CLI (installed via `dbt-snowflake`)

### Install

```bash
# Clone and create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file at the project root:

```dotenv
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_WAREHOUSE=NFL_ANALYTICS_WH
SNOWFLAKE_DATABASE=NFL_ANALYTICS
SNOWFLAKE_SCHEMA=RAW
SNOWFLAKE_ROLE=your_role

# Key-pair authentication
SNOWFLAKE_KEYPAIR_PRIVATE_KEY="-----BEGIN ENCRYPTED PRIVATE KEY-----\n...\n-----END ENCRYPTED PRIVATE KEY-----"
SNOWFLAKE_KEYPAIR_PASSPHRASE=your_key_passphrase
```

### dbt Profile

Add the following to `~/.dbt/profiles.yml`:

```yaml
nfl_batchgineering:
  target: dev
  outputs:
    dev:
      type: snowflake
      account: "{{ env_var('SNOWFLAKE_ACCOUNT') }}"
      user: "{{ env_var('SNOWFLAKE_USER') }}"
      private_key: "{{ env_var('SNOWFLAKE_KEYPAIR_PRIVATE_KEY') }}"
      private_key_passphrase: "{{ env_var('SNOWFLAKE_KEYPAIR_PASSPHRASE') }}"
      role: "{{ env_var('SNOWFLAKE_ROLE') }}"
      database: NFL_ANALYTICS
      warehouse: NFL_ANALYTICS_WH
      schema: STAGING
      threads: 4
```

---

## Running the Pipeline

### 1. Load raw data to Snowflake

```bash
python src/ingestion/training_data_loader.py
```

Loads the following nflverse datasets for the current season (historical years are a one-time load):
- `play_by_play` — full play-by-play with EPA
- `rosters` — active and historical rosters
- `player_summary_stats` — per-player season/weekly stats
- `team_summary_stats` — per-team summary stats
- `play_by_play_participation` — player participation per play
- `injuries` — weekly injury reports
- `schedules` — full schedule with Vegas lines (all seasons, single file)

### 2. Run dbt transformations

```bash
cd dbt_project
dbt run        # Build all models
dbt test       # Run data quality tests
```

Or layer by layer:
```bash
dbt run --select 1_staging
dbt run --select 2_intermediate
dbt run --select 3_marts
```

See `TESTING_GUIDE.md` for validation queries and expected row counts.

### 3. Train models

```bash
# XGBoost spread model (primary)
python src/ml/train_spread_model.py

# Elo baseline model
python src/ml/train_elo_model.py

# Full ensemble (requires base models trained first)
python src/ml/train_ensemble.py
```

Trained artifacts are serialized to the `models/` directory.

### 4. Generate predictions

```bash
python src/ml/predict.py --week 15 --season 2025 --output predictions_week15.csv
```

Or run the full pipeline via Dagster:
```bash
dagster dev -m dagster_project
```

---

## Key Design Decisions

### No Look-Ahead Bias
All rolling features use point-in-time joins. Window functions use `ROWS BETWEEN N PRECEDING AND 1 PRECEDING` — never including the current game. This is enforced in dbt intermediate models and validated via `mart_model_validation`.

### Sample Size Constraints
NFL seasons have ~270 games. With ~20 years of data that is ~5,000 samples. The architecture deliberately avoids deep learning (LSTM, transformers) in favor of shallow models with strong regularization.

### Calibration Over Accuracy
Models are optimized for Brier Score. Well-calibrated probabilities produce higher ROI than raw directional accuracy (Walsh & Joshi, 2024).

### Fractional Kelly Sizing
All bet sizing uses 0.25x Kelly to reduce variance. A minimum edge threshold of 3–4 points against the Vegas line is required before any bet is recommended.

---

## Code Quality

```bash
# Lint and format with ruff
ruff check .
ruff format .
```

Configuration in `pyproject.toml`: line length 88, Python 3.11 target, isort integrated.

---

## License

MIT
