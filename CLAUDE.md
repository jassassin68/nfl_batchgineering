# CLAUDE.md - NFL Prediction System Project Rules

## Project Overview

This is an NFL game prediction system for point spread and totals betting. The system uses a hybrid ensemble approach combining gradient boosting, Bayesian state-space models, and neural networks.

**Primary Goal**: Achieve >52.4% ATS accuracy to overcome -110 juice.

---

## Core Principles

### 1. Prevent Look-Ahead Bias (CRITICAL)
- NEVER use future data to predict past games
- All features must be calculated using only data available BEFORE the game
- Use point-in-time joins, not latest values
- Validate temporal correctness in every feature pipeline

### 2. Respect Sample Size Constraints
- NFL has ~270 games/season, ~5,000 total over 20 years
- This fundamentally limits model complexity
- Prefer simpler models with strong regularization
- NO deep learning architectures (LSTM, transformers) - insufficient data

### 3. Optimize for Calibration, Not Accuracy
- Target: Brier Score optimization
- Well-calibrated probabilities > raw accuracy
- Source: Walsh & Joshi (2024) showed 69.86% higher returns

### 4. Conservative Edge Thresholds
- Minimum edge: 3-4 points vs Vegas line
- If edge < threshold, NO BET (even if model is confident)
- Kelly fraction: Use 0.25-0.5x Kelly, never full Kelly

---

## Code Standards

### Python Style
- Use Python 3.11+
- Type hints on all functions
- Docstrings with Args, Returns, Raises
- Follow PEP 8, enforced by ruff
- Use Polars over Pandas where possible (performance)

### Testing Requirements
- Unit tests for all feature engineering functions
- Integration tests for data pipelines
- Backtesting framework must use walk-forward validation
- Minimum 80% coverage on core modules

### Data Validation
- Schema validation on all ingested data
- Check for nulls, outliers, impossible values
- Log data quality metrics
- Fail fast on data integrity issues

---

## Feature Engineering Rules

### EPA-Based Features (Primary)
```python
# Correct: Rolling window with point-in-time
def calculate_rolling_epa(team_id: str, game_date: date, window: int = 4) -> float:
    """Calculate rolling EPA using only games BEFORE game_date."""
    ...

# WRONG: This introduces look-ahead bias
def calculate_season_epa(team_id: str, season: int) -> float:
    """Uses full season data - includes future games!"""
    ...
```

### Weighting
- Offense EPA: weight 1.6x
- Defense EPA: weight 1.0x
- Source: nfeloapp research on optimal weighting

### Required Feature Categories
1. Efficiency: EPA/play (pass/rush, off/def), Success Rate
2. Player-level: QB EPA/dropback, CPOE, RYOE
3. Situational: Red zone, 3rd down, scoring position
4. Context: Rest days, travel distance, divisional game
5. Market: Opening line, line movement, sharp money indicators

---

## Model Architecture

### Ensemble Structure
```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   XGBoost   │  │  Bayesian   │  │   Neural    │  │    Elo      │
│  Regressor  │  │ State-Space │  │   Network   │  │  Baseline   │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │                │
       └────────────────┴────────────────┴────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │  Stacking Meta-     │
                    │  Learner (Ridge)    │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │  Betting Decision   │
                    │  Layer              │
                    └─────────────────────┘
```

### XGBoost Configuration
```python
xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 4,           # Shallow trees
    'learning_rate': 0.05,    # Low LR
    'n_estimators': 200,      # Early stopping
    'reg_alpha': 0.1,         # L1 regularization
    'reg_lambda': 1.0,        # L2 regularization
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
```

### Bayesian State-Space (Glickman-Stern)
- Team strength: Latent AR(1) process
- Week-to-week variance: τ²
- Season-to-season variance: σ²
- Home field advantage: Fixed effect (~2.5 points)

### Neural Network Constraints
- Maximum 3 hidden layers
- Dropout: 0.3-0.5
- Early stopping with patience=10
- L2 regularization
- NO recurrent architectures

---

## Validation Protocol

### Walk-Forward Cross-Validation (REQUIRED)
```python
def walk_forward_cv(data: DataFrame, n_seasons_train: int = 5):
    """
    Train on seasons [t-n, t-1], test on season t.
    Repeat for each available test season.
    
    NEVER use random splits - this introduces temporal leakage.
    """
    seasons = data['season'].unique().sort()
    for test_season in seasons[n_seasons_train:]:
        train = data[data['season'] < test_season]
        test = data[data['season'] == test_season]
        yield train, test
```

### Metrics to Report
- ATS Accuracy (target: >52.4%)
- RMSE vs actual margin
- RMSE vs Vegas line (should be lower than actual)
- Brier Score
- Calibration curve
- ROI simulation with Kelly sizing

---

## Betting Logic

### Edge Calculation
```python
def calculate_edge(model_pred: float, vegas_line: float) -> float:
    """
    Positive edge = model favors home more than Vegas
    Negative edge = model favors away more than Vegas
    """
    return model_pred - vegas_line

def should_bet(edge: float, threshold: float = 3.0) -> bool:
    """Only bet when edge exceeds threshold."""
    return abs(edge) >= threshold
```

### Kelly Criterion
```python
def kelly_fraction(
    win_prob: float,
    odds: float = -110,
    kelly_mult: float = 0.25  # Use fractional Kelly
) -> float:
    """
    Calculate Kelly bet size.
    
    Args:
        win_prob: Model's estimated win probability
        odds: American odds (negative for favorites)
        kelly_mult: Fraction of full Kelly (0.25 recommended)
    """
    decimal_odds = american_to_decimal(odds)
    b = decimal_odds - 1  # Net profit per unit
    p = win_prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    return max(0, kelly * kelly_mult)
```

---

## Data Sources

### Primary: nflverse
- GitHub: https://github.com/nflverse/nflverse-data

### Secondary
- Football Outsiders: DVOA (may require scraping)
- Weather: Visual Crossing or Open-Meteo API
- Vegas Lines: The Odds API or historical archives

### Storage
- Snowflake: Primary data warehouse
- dbt: Transformation layer
- Feature store: Point-in-time correct features

---

## Common Pitfalls to Avoid

### 1. Survivorship Bias
- Include all teams, even those that relocated/rebranded
- Don't exclude "bad" games or outliers

### 2. Overfitting
- More features ≠ better model
- Start simple, add complexity only if validated
- Watch for train/test performance gap

### 3. Line Movement Confusion
- Use OPENING lines for backtesting consistency
- Closing lines incorporate market information

### 4. Ignoring Uncertainty
- Always report prediction intervals
- Edge without confidence = noise

### 5. Data Leakage
- Player stats from current game
- Season-end metrics applied retroactively
- Future injury information

---

## File Organization

```
src/
├── ingestion/
│   ├── training_data_loader.py          # nflverse data loading to Snowflake
├── ml/
│   ├── __init__.py
│   ├── base.py            # Abstract base model
│   ├── elo.py             # Elo rating system
│   ├── xgboost_model.py   # XGBoost implementation
│   ├── bayesian.py        # State-space model
│   ├── neural.py          # Shallow NN
│   └── ensemble.py        # Stacking meta-learner
├── betting/
│   ├── __init__.py
│   ├── edge.py            # Edge calculation
│   ├── kelly.py           # Bet sizing
│   └── backtest.py        # Historical simulation
└── api/
    ├── __init__.py
    └── main.py            # FastAPI endpoints
```

---

## Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Required packages
# polars, xgboost, pymc, torch, fastapi, uvicorn
# nfl-data-py, scikit-learn, mlflow, prefect
```

---

## Git Workflow

- Main branch: Production-ready code only
- Feature branches: `feature/description`
- Commit messages: Conventional commits
- PR required for main, with passing tests

---

## Logging & Monitoring

- Use `structlog` for structured logging
- Log all predictions with timestamps
- Track model performance over time
- Alert on significant accuracy degradation

---

## Questions to Ask Before Starting

1. What Snowflake database/schema should I use?
2. Is there existing dbt project structure?
3. What historical Vegas line data is available?
4. What's the target deployment environment?
5. Are there existing ETL schedules to integrate with?
