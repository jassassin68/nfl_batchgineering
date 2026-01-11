# NFL PREDICTION SYSTEM - CLAUDE CODE PLAN MODE PROMPT

**Copy everything below this line into Claude Code Plan Mode:**

---

## MISSION

Build a production-grade NFL game prediction system for point spread and totals betting. The system must use a hybrid ensemble approach validated by academic research, achieving >52.4% ATS accuracy to overcome -110 juice.

---

## RESEARCH FOUNDATION

This project is based on systematic review of ML in sports betting (Galekwa et al., 2024, arXiv:2410.21484v1) and NFL-specific research.

### Recommended Architecture

**Hybrid Ensemble combining:**
1. **XGBoost** (primary) - Handles mixed features, built-in regularization for small datasets
2. **Bayesian State-Space** - Dynamic team strength per Glickman & Stern (1998, JASA)
3. **Shallow Neural Network** - 2-3 layers max, ensemble diversity
4. **Elo Baseline** - Interpretable features

### Critical Feature Engineering

| Category | Features | Source |
|----------|----------|--------|
| Efficiency | EPA/play (pass/rush, off/def), Success Rate | nflverse |
| Weighting | Offense EPA 1.6x, Defense 1.0x | nfeloapp research |
| Player | QB EPA/dropback, CPOE | nflfastR |
| Temporal | Rolling 4-game weighted averages | Academic consensus |

### Key Thresholds

- **Break-even**: 52.4% accuracy at -110 juice
- **Minimum edge**: 3-4 points vs Vegas line
- **Kelly fraction**: 0.25x maximum (conservative)
- **Optimization**: Brier Score (calibration) > raw accuracy

---

## TECHNICAL REQUIREMENTS

### Stack
- **Language**: Python 3.11+
- **Data**: Polars (preferred), nfl-data-py
- **Storage**: Snowflake + dbt
- **ML**: XGBoost, PyMC (Bayesian), PyTorch (lightweight NN)
- **API**: FastAPI
- **Tracking**: MLflow
- **Orchestration**: Dagster

### Code Standards
- Type hints on all functions
- Google-style docstrings
- ruff for linting/formatting
- pytest with 80% coverage
- Walk-forward CV only (NEVER random splits)

---

## CRITICAL CONSTRAINTS

### 1. PREVENT LOOK-AHEAD BIAS
```python
# CORRECT: Uses only games BEFORE current game
def rolling_epa(team, game_date, window=4):
    return games.filter(
        (pl.col("team") == team) & 
        (pl.col("game_date") < game_date)  # Strict inequality!
    ).tail(window).mean()

# WRONG: Uses future data
def season_epa(team, season):
    return games.filter(...).mean()  # Includes future games!
```

### 2. SAMPLE SIZE LIMITS
- ~270 games/season, ~5,000 total over 20 years
- NO deep learning (LSTM, transformers) - insufficient data
- Keep max_depth ≤ 4 for tree models
- Resist complexity urges

### 3. VALIDATION PROTOCOL
```python
# Walk-forward only
for test_season in seasons[5:]:
    train = data[data['season'] < test_season]  # Past only
    test = data[data['season'] == test_season]
    # Train and evaluate
```

---

## PROJECT STRUCTURE

```
nfl-prediction-system/
├── .claude/
│   ├── settings.json
│   └── commands/
│       ├── train.md
│       ├── predict.md
│       ├── backtest.md
│       └── data.md
├── CLAUDE.md              # Project rules (provided)
├── src/
│   ├── ingestion/
│   │   ├── training_data_loader.py      # nflverse loading to Snowflake.
│   │   ├── features.py    # Feature engineering
│   │   └── validation.py  # Quality checks
│   ├── ml/
│   │   ├── base.py        # Abstract interface
│   │   ├── elo.py
│   │   ├── xgboost_model.py
│   │   ├── bayesian.py
│   │   ├── neural.py
│   │   └── ensemble.py
│   ├── betting/
│   │   ├── edge.py
│   │   ├── kelly.py
│   │   └── backtest.py
│   └── api/
│       └── main.py
├── dbt_project/
│   └── models/
│       ├── staging/
│       ├── intermediate/
│       └── marts/
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_betting.py
│   └── test_lookahead.py  # Critical!
├── config/
    └── model_config.yaml
├── snowflake_sql/
    └── Files for setting up Snowflake database(s).
├── notebooks/
    └── Research notebooks.
├── dagster_project/
    └── Files for scheduling and Orchestrating data processing files.
```

---

## IMPLEMENTATION PHASES

### Phase 1: Data Infrastructure ✅ COMPLETE
1. ✅ Project structure created
2. ✅ nflverse data ingestion → Snowflake operational
3. ✅ dbt models built:
   - ✅ `int_plays_cleaned` (play-by-play with EPA)
   - ✅ `mart_game_prediction_features` (model-ready)
4. ✅ Data validation implemented (no look-ahead bias)

**Status: Data infrastructure is operational. Feature store is populated with point-in-time correct features.**

---

### Phase 2: Baseline Models ✅ COMPLETE
1. ✅ Elo rating system with:
   - ✅ K-factor optimization
   - ✅ Home field adjustment
   - ✅ Margin of victory
2. ✅ Simple linear regression baseline
3. ✅ Establish benchmark metrics

### Phase 3: Core ML Models ← CURRENT PHASE
1. XGBoost regressor:
   - max_depth=4, learning_rate=0.05
   - L1/L2 regularization
   - Early stopping
2. Bayesian state-space (Glickman-Stern):
   - Team strength as AR(1) latent variable
   - Week-to-week and season variance
3. Shallow NN (2-3 layers, dropout 0.3-0.5)

### Phase 4: Ensemble & Betting
1. Stacking meta-learner:
   ```
   Final = w1*XGB + w2*Bayesian + w3*NN + w4*Elo
   ```
2. Optimize weights via walk-forward CV
3. Calibrate for Brier Score
4. Build edge calculation and Kelly sizing
5. Implement bankroll management

### Phase 5: API & Deployment
1. FastAPI endpoints for predictions
2. Weekly retraining pipeline (Dagster)
3. MLflow experiment tracking
4. Performance monitoring dashboard

---

## SUCCESS CRITERIA

| Metric | Target | Rationale |
|--------|--------|-----------|
| ATS Accuracy | >52.4% | Break-even at -110 |
| Brier Score | <0.25 | Well-calibrated |
| ROI | >3% | After juice |
| Bet Rate | 10-15% | Selective betting |
| Edge Threshold | 3+ pts | High-confidence only |

---

## WARNINGS (FROM ACADEMIC LITERATURE)

1. **Market Efficiency**: NFL is among most efficient. Expect small, fragile edges.
2. **Publication Bias**: 60%+ accuracy claims rarely replicate.
3. **Temporal Decay**: Edges get arbitraged in 2-5 years.
4. **Backtesting ≠ Live**: Transaction costs and execution matter.
5. **The 52.4% Paradox**: Even 55% sustained is extraordinarily difficult.

---

## GETTING STARTED

**Phase 1 and Phase 2 are complete.** Begin Phase 3: Core ML Models.

### Data Available:
- Feature store: `mart_game_prediction_features` in Snowflake
- Features include: rolling EPA (4-game), success rates, rest days, travel, Vegas lines
- Historical data: 2020-2024 seasons

### Before Proceeding, Confirm:
- Can you connect to Snowflake and query `mart_game_prediction_features`?
- What seasons have complete feature data?
- Are Vegas opening lines included in the feature store?

---

## ADDITIONAL FILES PROVIDED

I've created these files for you in the project directory:
- `CLAUDE.md` - Detailed project rules
- `.claude/settings.json` - Claude Code configuration
- `.claude/commands/train.md` - /train slash command
- `.claude/commands/predict.md` - /predict slash command
- `.claude/commands/backtest.md` - /backtest slash command
- `.claude/commands/data.md` - /data slash command
- `.claude/references/ACADEMIC_RESEARCH.md` - Full academic reference
- `requirements.txt` - Python dependencies

Review these files before starting implementation.

---

**END OF PROMPT**
