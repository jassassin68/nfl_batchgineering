# NFL Prediction System: Claude Code Plan Mode Prompt

## Mission Statement

Build a production-grade NFL game prediction system optimized for point spread and game totals betting. The system must achieve >52.4% accuracy ATS to overcome standard -110 juice, with disciplined edge calculation and Kelly-based bet sizing.

---

## Research Foundation (Academic Consensus)

### Recommended Model Architecture: Hybrid Ensemble

Based on systematic review of ML in sports betting (Galekwa et al., 2024) and NFL-specific research:

1. **Primary Model: Gradient Boosted Trees (XGBoost/LightGBM)**
   - Handles mixed feature types (categorical teams, continuous metrics, ordinal rankings)
   - Built-in regularization for small NFL dataset (~270 games/season, ~5,000 total over 20 years)
   - Feature importance for edge identification
   - Source: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" (KDD 2016)

2. **Dynamic Team Strength: Bayesian State-Space Models**
   - Models team strength as latent variable with first-order autoregressive evolution
   - Accounts for week-to-week variation (injuries, random factors) and season-to-season changes
   - Source: Glickman & Stern, "A State-Space Model for National Football League Scores" (JASA 1998)
   - Implementation: https://glicko.net/research/nfl-chapter.pdf

3. **Neural Network Component (Shallow)**
   - 2-3 hidden layers maximum (sample size constraint)
   - Dropout regularization
   - NOT LSTM/RNN (insufficient data for sequence models)
   - Source: Vu research noting ~2,000 game limitation

4. **Baseline: Elo/Glicko Rating System**
   - Provides interpretable baseline features
   - Fast recalculation for real-time updates

### Critical Feature Engineering (EPA-Based)

| Category | Features | Rationale |
|----------|----------|-----------|
| Efficiency | EPA/play (pass/rush, off/def), Success Rate, DVOA | Context-adjusted performance |
| Stability | Weighted EPA (offense 1.6x, defense 1.0x) | Optimal weighting per nfeloapp research |
| Player-level | QB EPA/dropback, CPOE, RYOE | Isolate skill from scheme |
| Situational | Red zone efficiency, 3rd down conversion | High-leverage context |
| Opponent-adjusted | DVOA, SOS-weighted metrics | Schedule effects |
| Temporal | Rolling 4-game weighted averages | Recency bias |

### Betting Decision Thresholds

- **Break-even requirement**: 52.4% accuracy at -110 juice
- **Minimum edge**: Only bet when |Model_prediction - Vegas_line| > 3-4 points
- **Optimization target**: Brier Score (calibration), NOT raw accuracy
- **Source**: Walsh & Joshi (2024) showed calibration-optimized models generate 69.86% higher returns

---

## Technical Stack Requirements

### Data Layer
- **Source**: nflverse (nflfastR for play-by-play with EPA)
- **Storage**: Snowflake (Jeff's existing infrastructure)
- **Transformation**: dbt for feature engineering pipelines
- **Additional**: Football Outsiders (DVOA), weather APIs, Vegas lines APIs

### Model Layer
- **Framework**: Python with scikit-learn ecosystem
- **Gradient Boosting**: XGBoost or LightGBM
- **Bayesian**: PyMC or Stan for state-space models
- **Neural Network**: PyTorch or TensorFlow (lightweight)
- **Ensemble**: Custom stacking meta-learner

### Application Layer
- **API**: FastAPI for prediction endpoints
- **Scheduling**: Prefect or Airflow for weekly model updates
- **Monitoring**: MLflow for experiment tracking

---

## Implementation Phases

### Phase 1: Data Infrastructure
1. Set up nflverse data ingestion pipeline to Snowflake
2. Create dbt models for:
   - Raw play-by-play staging
   - EPA aggregation (game, team, rolling windows)
   - Opponent adjustments
   - Historical Vegas lines integration
3. Build feature store with point-in-time correctness (prevent look-ahead bias)

### Phase 2: Baseline Models
1. Implement Elo rating system with:
   - K-factor optimization
   - Home field adjustment
   - Margin of victory integration
2. Build simple linear regression baseline
3. Establish benchmark metrics on historical data

### Phase 3: Core ML Models
1. XGBoost regressor for:
   - Margin of victory prediction
   - Total points prediction
2. Bayesian state-space model (Glickman-Stern style):
   - Team strength as latent variable
   - Week-to-week variance parameter
   - Season-to-season variance parameter
3. Shallow neural network (2-3 layers, dropout)

### Phase 4: Ensemble & Calibration
1. Implement stacking meta-learner:
   ```
   Final = w1*XGB + w2*StateSpace + w3*NN + w4*Elo
   ```
2. Optimize weights via walk-forward cross-validation
3. Calibrate for Brier Score
4. Build prediction intervals (not just point estimates)

### Phase 5: Betting Decision Layer
1. Edge calculation: Model_pred - Vegas_line
2. Confidence thresholds (3-4 point minimum edge)
3. Kelly criterion bet sizing with fractional Kelly (0.25-0.5x)
4. Bankroll management system

### Phase 6: Deployment & Monitoring
1. FastAPI endpoints for predictions
2. Weekly automated retraining pipeline
3. Performance tracking dashboard
4. Model drift detection

---

## Validation Requirements (CRITICAL)

### Walk-Forward Cross-Validation
- Train on seasons 1-N, test on season N+1
- NO random train/test splits (temporal leakage)
- Simulate realistic betting scenario

### Metrics to Track
- ATS accuracy (target: >52.4%)
- RMSE vs Vegas lines
- Brier Score (calibration)
- ROI with actual bet sizing
- Sharpe ratio of returns

### Guard Against
1. **Look-ahead bias**: Features must use only data available before game
2. **Overfitting**: Limited to ~5,000 games total—keep models simple
3. **Publication bias**: Published 65%+ accuracy claims rarely replicate
4. **Temporal decay**: Edges found in old data may not persist

---

## File Structure

```
nfl-prediction-system/
├── .claude/
│   ├── settings.json          # Claude Code configuration
│   └── commands/              # Slash commands
│       ├── train.md
│       ├── predict.md
│       └── backtest.md
├── CLAUDE.md                  # Project rules for Claude Code
├── src/
│   ├── data/
│   │   ├── ingest/           # nflverse data loading
│   │   ├── features/         # Feature engineering
│   │   └── validation/       # Data quality checks
│   ├── models/
│   │   ├── elo/              # Baseline Elo system
│   │   ├── xgboost/          # Gradient boosting
│   │   ├── bayesian/         # State-space models
│   │   ├── neural/           # Shallow NN
│   │   └── ensemble/         # Meta-learner
│   ├── betting/
│   │   ├── edge.py           # Edge calculation
│   │   ├── kelly.py          # Bet sizing
│   │   └── bankroll.py       # Management
│   └── api/
│       └── main.py           # FastAPI app
├── dbt/
│   └── models/               # dbt transformations
├── tests/
│   ├── unit/
│   └── integration/
├── notebooks/
│   └── exploration/          # EDA notebooks
└── config/
    └── model_config.yaml     # Hyperparameters
```

---

## Key Academic References

1. Glickman & Stern (1998) - State-space NFL model - https://glicko.net/research/nfl-chapter.pdf
2. Galekwa et al. (2024) - ML in sports betting systematic review - https://arxiv.org/html/2410.21484v1
3. Chen & Guestrin (2016) - XGBoost - KDD proceedings
4. Gray & Gray (1997) - NFL market efficiency - Journal of Finance 52(4)
5. Walsh & Joshi (2024) - Calibration vs accuracy optimization

---

## Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| ATS Accuracy | >52.4% | Break-even at -110 |
| Brier Score | <0.25 | Well-calibrated |
| ROI | >3% | After juice |
| Sharpe Ratio | >1.0 | Risk-adjusted |
| Edge Threshold | 3+ pts | High-confidence bets only |

---

## Warnings & Skepticism

1. **Market efficiency**: NFL betting markets are among the most efficient. Expect small, fragile edges.
2. **Sample size**: ~270 games/year fundamentally limits complexity. Resist deep learning urges.
3. **Temporal decay**: Any edge discovered will likely be arbitraged away within 2-5 years.
4. **Backtesting ≠ Live**: Transaction costs, line movement, and execution matter.
5. **The 52.4% paradox**: Even 55% sustained accuracy is extraordinarily difficult.

---

## Getting Started Command

```
Claude, read this plan and begin Phase 1. Start by:
1. Creating the project structure
2. Setting up nflverse data ingestion
3. Building initial dbt models for EPA aggregation
4. Implementing point-in-time feature store

Ask clarifying questions about Snowflake connection details and existing infrastructure before proceeding.
```
