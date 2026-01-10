# /train - Model Training Command

## Description
Train one or more models in the NFL prediction ensemble.

## Usage
```
/train [model] [options]
```

## Models
- `elo` - Baseline Elo rating system
- `xgboost` - Gradient boosted trees (margin + totals)
- `bayesian` - Glickman-Stern state-space model
- `neural` - Shallow neural network
- `ensemble` - Full stacking meta-learner
- `all` - Train all models in sequence

## Options
- `--seasons` - Training seasons (e.g., `2019-2023`)
- `--holdout` - Holdout season for validation
- `--cv` - Use walk-forward cross-validation
- `--tune` - Run hyperparameter optimization
- `--track` - Log to MLflow

## Examples

### Train XGBoost with walk-forward CV
```
/train xgboost --seasons 2015-2023 --cv
```

### Train full ensemble with MLflow tracking
```
/train ensemble --seasons 2018-2023 --holdout 2024 --track
```

### Tune Bayesian model hyperparameters
```
/train bayesian --tune --seasons 2019-2023
```

## Implementation Notes

When executing this command:

1. **Data Loading**
   - Load features from feature store
   - Validate temporal correctness
   - Check for missing values

2. **Training Protocol**
   - Use walk-forward CV (NEVER random splits)
   - Train on seasons [t-n, t-1], validate on t
   - Report metrics for each fold

3. **Metrics to Report**
   - ATS Accuracy (target >52.4%)
   - RMSE vs actual margin
   - RMSE vs Vegas line
   - Brier Score
   - Feature importance (for tree models)

4. **Artifacts to Save**
   - Model weights/parameters
   - Training config
   - Validation metrics
   - Feature importance plots

5. **Post-Training Checks**
   - Compare train vs validation performance (overfitting check)
   - Verify calibration curve
   - Log to MLflow if --track specified

## Code Template

```python
from src.models import EloModel, XGBoostModel, BayesianModel, NeuralModel, Ensemble
from src.data import load_features
from src.validation import walk_forward_cv

def train_model(
    model_type: str,
    seasons: tuple[int, int],
    holdout: int | None = None,
    use_cv: bool = True,
    tune: bool = False,
    track: bool = False
):
    """Train specified model with validation."""
    
    # Load data with point-in-time correctness
    features = load_features(seasons)
    
    # Select model
    model_class = {
        'elo': EloModel,
        'xgboost': XGBoostModel,
        'bayesian': BayesianModel,
        'neural': NeuralModel,
        'ensemble': Ensemble
    }[model_type]
    
    model = model_class()
    
    if tune:
        model = tune_hyperparameters(model, features)
    
    if use_cv:
        results = walk_forward_cv(model, features)
    else:
        results = model.fit(features, holdout=holdout)
    
    if track:
        log_to_mlflow(model, results)
    
    return results
```
