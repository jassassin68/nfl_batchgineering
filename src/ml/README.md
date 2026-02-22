# NFL Prediction ML Module

Machine learning models for NFL point spread and totals prediction. Implements a 4-model ensemble architecture designed to achieve >52.4% ATS accuracy to overcome -110 juice.

## Architecture Overview

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   XGBoost   │  │  Bayesian   │  │   Neural    │  │    Elo      │
│  Regressor  │  │ State-Space │  │   Network   │  │  Baseline   │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │                │
       └────────────────┴────────────────┴────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │  Ridge Regression   │
                    │  Meta-Learner       │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │  Calibrated Output  │
                    │  (Spread + Prob)    │
                    └─────────────────────┘
```

## Directory Structure

```
src/ml/
├── README.md                 # This file
├── base.py                   # Abstract base class for all models
├── train_elo_model.py        # Elo baseline training script
├── train_spread_model.py     # XGBoost training script
├── train_ensemble.py         # Full ensemble training script
├── models/
│   ├── __init__.py
│   ├── elo_model.py          # Elo rating system
│   ├── spread_predictor.py   # XGBoost spread predictor
│   ├── bayesian.py           # Bayesian state-space model (PyMC)
│   ├── neural.py             # Shallow neural network (PyTorch)
│   └── ensemble.py           # Stacking meta-learner
└── utils/
    ├── __init__.py
    ├── feature_engineering.py # Feature selection and creation
    ├── evaluation.py          # Model evaluation metrics
    └── validation.py          # Walk-forward cross-validation
```

## Installation

```bash
# Core dependencies (already in requirements.txt)
pip install polars numpy xgboost scikit-learn

# Phase 3 dependencies for Bayesian and Neural models
pip install pymc arviz torch
```

## Quick Start

### Train Full Ensemble

```bash
# Full ensemble (all 4 models) - takes 15-30 minutes
python src/ml/train_ensemble.py --min-season 2015 --output-dir models/ensemble

# Skip Bayesian model (faster, ~5 minutes)
python src/ml/train_ensemble.py --skip-bayesian --output-dir models/ensemble

# Skip both Bayesian and Neural (fastest, ~1 minute)
python src/ml/train_ensemble.py --skip-bayesian --skip-neural --output-dir models/ensemble
```

### Train Individual Models

```bash
# Elo baseline only
python src/ml/train_elo_model.py --min-season 2020 --output-dir models/elo_baseline

# XGBoost only
python src/ml/train_spread_model.py --min-season 2015 --output-dir models/xgboost
```

---

## Models

### 1. EloModel (`models/elo_model.py`)

Time-decayed Elo rating system based on FiveThirtyEight methodology.

**Key Features:**
- K-factor: 20 (NFL standard)
- Home advantage: 48 Elo points (~2.5 point spread)
- Margin of victory multiplier with log transform
- Season-to-season regression (1/3 toward mean)

**Usage:**
```python
from src.ml.models import EloModel

elo = EloModel(k_factor=20.0, home_advantage=48.0)
elo.fit(train_df)  # DataFrame with game_id, season, week, home_team, away_team, home_score, away_score

# Predict
spread = elo.predict_spread("KC", "SF")  # Returns predicted spread
win_prob = elo.predict_win_probability("KC", "SF")  # Returns home win probability

# Get rankings
rankings = elo.get_current_ratings()  # DataFrame with team, elo_rating
```

### 2. SpreadPredictor (`models/spread_predictor.py`)

XGBoost gradient boosting regressor for spread prediction.

**Key Features:**
- Regularization: L1 (alpha=0.1), L2 (lambda=1.0)
- Early stopping with validation set
- Feature importance tracking
- Confidence scoring

**Hyperparameters (CLAUDE.md compliant):**
```python
{
    'max_depth': 4-5,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}
```

**Usage:**
```python
from src.ml.models import SpreadPredictor

model = SpreadPredictor()
model.train(X_train, y_train, X_val, y_val, feature_names=feature_cols)

predictions = model.predict(X_test)
predictions, confidence = model.predict(X_test, return_confidence=True)

# Feature importance
importance = model.get_feature_importance(importance_type='gain')
```

### 3. BayesianStateSpace (`models/bayesian.py`)

Bayesian state-space model based on Glickman & Stern (1998) JASA methodology.

**Mathematical Model:**
```
Y_ij = θ_i - θ_j + h + ε_ij

where:
  Y_ij = point differential (home - away)
  θ_i  = home team strength (latent, sum-to-zero constraint)
  θ_j  = away team strength (latent)
  h    = home field advantage (~2.5 points)
  ε_ij ~ N(0, σ²_game)
```

**Key Features:**
- Team strength as latent variables with sum-to-zero constraint
- MCMC inference via PyMC NUTS sampler
- Posterior uncertainty quantification
- Convergence diagnostics (R-hat, divergences)

**Usage:**
```python
from src.ml.models import BayesianStateSpace

model = BayesianStateSpace(n_samples=1000, n_chains=2)
diagnostics = model.fit(
    X=None,  # Not used - uses team names
    y=spreads,
    home_teams=home_team_array,
    away_teams=away_team_array
)

# Check convergence
print(f"R-hat: {diagnostics['max_rhat']}")
print(f"Divergences: {diagnostics['divergences']}")

# Predict
spread = model.predict_matchup("KC", "SF")
spreads = model.predict_batch(["KC", "BUF"], ["SF", "MIA"])

# Team rankings
rankings = model.get_team_rankings()
```

### 4. NeuralNetPredictor (`models/neural.py`)

Shallow neural network with dropout and L2 regularization.

**Architecture (CLAUDE.md compliant):**
```
Input (n_features)
  → Linear(64) → BatchNorm → ReLU → Dropout(0.4)
  → Linear(32) → BatchNorm → ReLU → Dropout(0.4)
  → Linear(16) → BatchNorm → ReLU → Dropout(0.4)
  → Linear(1)
```

**Key Features:**
- Maximum 3 hidden layers (per CLAUDE.md constraint)
- Dropout: 0.4 (within 0.3-0.5 range)
- L2 regularization via weight_decay
- Early stopping with patience=10
- MC Dropout for uncertainty estimation

**Usage:**
```python
from src.ml.models import NeuralNetPredictor

model = NeuralNetPredictor(
    hidden_dims=(64, 32, 16),
    dropout_rate=0.4,
    learning_rate=0.001,
    weight_decay=0.01,
    patience=10
)

model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
predictions = model.predict(X_test)

# Uncertainty estimation (MC Dropout)
mean_preds, std_preds = model.predict_with_uncertainty(X_test, n_samples=100)
```

### 5. StackingEnsemble (`models/ensemble.py`)

Stacking meta-learner combining all base models.

**Key Features:**
- Ridge regression meta-learner
- Combines predictions + uncertainty from base models
- Model weight interpretability
- Brier score optimization for calibration

**Usage:**
```python
from src.ml.models import StackingEnsemble

ensemble = StackingEnsemble(meta_alpha=1.0, use_uncertainty=True)

# Add pre-trained base models
ensemble.add_base_model('elo', elo_model)
ensemble.add_base_model('xgboost', xgb_model)
ensemble.add_base_model('bayesian', bayesian_model)
ensemble.add_base_model('neural', neural_model)

# Train meta-learner
results = ensemble.fit(
    X=X_train,
    y=y_train,
    home_teams=home_teams_train,
    away_teams=away_teams_train
)

print(f"Model weights: {results['model_weights']}")
print(f"Brier score: {results['brier_score']}")

# Predict
spreads = ensemble.predict(X_test, home_teams_test, away_teams_test)
probs = ensemble.predict_proba(X_test, home_teams_test, away_teams_test)

# Individual model breakdown
individual_preds = ensemble.get_individual_predictions(X_test, home_teams_test, away_teams_test)
```

---

## Utilities

### Walk-Forward Cross-Validation (`utils/validation.py`)

Time-series aware cross-validation to prevent look-ahead bias.

```python
from src.ml.utils import walk_forward_cv, walk_forward_cv_arrays

# DataFrame version
for train_df, test_df in walk_forward_cv(data, n_seasons_train=5):
    model.fit(train_df)
    preds = model.predict(test_df)

# NumPy array version
for X_train, y_train, X_test, y_test, test_season in walk_forward_cv_arrays(
    X, y, seasons, n_seasons_train=5
):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
```

**Metrics:**
```python
from src.ml.utils import (
    calculate_ats_accuracy,
    calculate_roi,
    calculate_brier_score,
    calculate_cv_metrics
)

# Against-the-spread accuracy
ats = calculate_ats_accuracy(predictions, actuals, vegas_spreads)

# ROI with edge threshold
roi = calculate_roi(predictions, actuals, vegas_spreads, edge_threshold=3.0)

# Brier score for probability calibration
brier = calculate_brier_score(predicted_probs, actual_outcomes)
```

### Feature Engineering (`utils/feature_engineering.py`)

Feature selection and derived feature creation.

```python
from src.ml.utils import (
    select_spread_features,
    create_derived_features,
    prepare_training_data
)

# Get feature columns
feature_cols = select_spread_features(df)

# Create derived features (matchup advantages, differentials)
df = create_derived_features(df)

# Full pipeline
df_prepared, features, target = prepare_training_data(df, target_type='spread')
```

**Available Features (43 base + 11 derived):**

| Category | Features |
|----------|----------|
| Home Offensive | epa_adj, epa_l4w, success_rate, success_l4w, explosive_rate, pass_epa, run_epa |
| Away Offensive | (same as home) |
| Home Defensive | def_epa, def_rank, def_pass_epa, def_run_epa, def_epa_l4w |
| Away Defensive | (same as home) |
| Situational | rz_td_rate, third_conv, two_min_epa (home + away) |
| Context | temp, wind, div_game, playoff |
| Vegas | vegas_spread, vegas_total, vegas_home_win_prob |
| Derived | epa_differential, matchup_advantages, wind_run_advantage, etc. |

### Evaluation (`utils/evaluation.py`)

Model evaluation and reporting.

```python
from src.ml.utils import evaluate_spread_model, create_performance_report

# Quick evaluation
metrics = evaluate_spread_model(y_true, y_pred, verbose=True)
# Returns: mae, rmse, r2, directional_accuracy, ats_accuracy, roi

# Full report with visualizations
report_path = create_performance_report(
    y_true, y_pred,
    model_name='ensemble',
    output_dir='reports/'
)
```

---

## Training Scripts

### train_ensemble.py

Full ensemble training pipeline.

```bash
python src/ml/train_ensemble.py [OPTIONS]

Options:
  --min-season INT        Minimum season to include (default: 2015)
  --limit INT             Limit games for testing
  --skip-bayesian         Skip Bayesian model (slow MCMC)
  --skip-neural           Skip Neural Network model
  --k-factor FLOAT        Elo K-factor (default: 20.0)
  --home-advantage FLOAT  Elo home advantage (default: 48.0)
  --max-depth INT         XGBoost max depth (default: 4)
  --learning-rate FLOAT   XGBoost learning rate (default: 0.05)
  --n-bayesian-samples INT MCMC samples (default: 500)
  --output-dir PATH       Output directory (default: models/ensemble)
```

**Output Structure:**
```
models/ensemble/
├── stacking_ensemble.pkl     # Meta-learner
├── training_summary.json     # Results and metrics
├── elo/
│   └── elo_model.json
├── xgboost/
│   ├── spread_predictor.json
│   └── spread_predictor_metadata.json
├── bayesian/
│   ├── bayesian_state_space.pkl
│   └── bayesian_state_space_trace.nc
├── neural/
│   ├── neural_net.pt
│   └── neural_net_metadata.json
└── base_models/
    └── (copies for ensemble loading)
```

### train_elo_model.py

Elo baseline training script.

```bash
python src/ml/train_elo_model.py [OPTIONS]

Options:
  --min-season INT        Minimum season (default: 2020)
  --k-factor FLOAT        Elo K-factor (default: 20.0)
  --home-advantage FLOAT  Home advantage in Elo points (default: 48.0)
  --output-dir PATH       Output directory (default: models/elo_baseline)
  --create-report         Generate visualizations
```

### train_spread_model.py

XGBoost standalone training script.

```bash
python src/ml/train_spread_model.py [OPTIONS]

Options:
  --min-season INT        Minimum season (default: 2015)
  --max-depth INT         XGBoost max depth (default: 5)
  --learning-rate FLOAT   Learning rate (default: 0.05)
  --n-estimators INT      Boosting rounds (default: 300)
  --early-stopping INT    Early stopping patience (default: 50)
  --output-dir PATH       Output directory (default: ml_models)
```

---

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| ATS Accuracy | >52.4% | Break-even at -110 juice |
| Brier Score | <0.25 | Well-calibrated probabilities |
| MAE | <10 pts | Competitive with Vegas |
| ROI (3pt edge) | >3% | After transaction costs |
| Bet Rate | 10-15% | Selective high-confidence bets |

---

## Constraints (from CLAUDE.md)

1. **No Look-Ahead Bias**: Walk-forward CV only, never random splits
2. **Sample Size Limits**: ~270 games/season, ~5,000 total over 20 years
3. **Neural Network**: Max 3 hidden layers, dropout 0.3-0.5
4. **XGBoost**: max_depth 4-5, heavy regularization
5. **No Deep Learning**: LSTM/transformers forbidden (insufficient data)
6. **Edge Threshold**: Minimum 3-4 points vs Vegas line to bet
7. **Kelly Fraction**: 0.25x maximum (conservative)

---

## API Reference

### BasePredictor (Abstract Base Class)

All models inherit from `BasePredictor` and implement:

```python
class BasePredictor(ABC):
    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None, feature_names=None, **kwargs) -> Dict

    @abstractmethod
    def predict(self, X) -> np.ndarray

    def predict_proba(self, X) -> np.ndarray  # Default: logistic(spread/5.5)

    @abstractmethod
    def save_model(self, output_dir) -> Path

    @abstractmethod
    def load_model(self, model_path) -> None

    def get_feature_importance(self) -> Optional[Dict]

    def get_prediction_uncertainty(self, X) -> np.ndarray
```

---

## Troubleshooting

### PyMC/Bayesian Issues

```bash
# If MCMC is slow or diverging
python src/ml/train_ensemble.py --n-bayesian-samples 200  # Reduce samples

# Check convergence in output
# R-hat should be < 1.1
# Divergences should be 0
```

### PyTorch/Neural Issues

```bash
# If CUDA out of memory
python src/ml/train_ensemble.py  # Uses CPU by default

# Force CPU
export CUDA_VISIBLE_DEVICES=""
```

### Import Errors

```bash
# Missing PyMC
pip install pymc arviz

# Missing PyTorch
pip install torch

# Models gracefully handle missing dependencies
# Ensemble works with 2+ models
```

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'src'`
- **Solution**: Run from project root or add to PYTHONPATH

**Issue**: `Snowflake connection failed`
- **Solution**: Check `.env` file has correct credentials

**Issue**: `No data returned from Snowflake`
- **Solution**: Verify marts tables are populated: `dbt run --select 3_marts`

**Issue**: `MAE > 15 points (poor performance)`
- **Solution**: Check for insufficient training data, missing features, or data leakage

---

## References

- Glickman, M. E., & Stern, H. S. (1998). A state-space model for National Football League scores. JASA.
- Silver, N. (2014). FiveThirtyEight NFL Elo Ratings.
- Walsh & Joshi (2024). Brier Score optimization for sports betting.
- nfeloapp research on EPA weighting (Offense 1.6x, Defense 1.0x)
- XGBoost Documentation: https://xgboost.readthedocs.io/
- PyMC Documentation: https://www.pymc.io/
- PyTorch Documentation: https://pytorch.org/docs/
