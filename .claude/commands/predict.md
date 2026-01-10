# /predict - Generate Predictions Command

## Description
Generate predictions for upcoming NFL games with edge calculations and bet recommendations.

## Usage
```
/predict [scope] [options]
```

## Scope
- `week` - Current week's games (default)
- `game [team1] [team2]` - Specific matchup
- `season [year]` - Full season simulation
- `historical [date]` - Past game (for validation)

## Options
- `--model` - Model to use (`ensemble`, `xgboost`, `bayesian`, `elo`)
- `--threshold` - Edge threshold for bets (default: 3.0)
- `--kelly` - Kelly multiplier (default: 0.25)
- `--lines` - Include current Vegas lines
- `--confidence` - Show prediction intervals
- `--explain` - Include feature contributions

## Examples

### Predict this week's games with betting recommendations
```
/predict week --lines --threshold 3.5
```

### Predict specific matchup with explanation
```
/predict game KC BUF --explain --confidence
```

### Backtest 2024 season
```
/predict season 2024 --model ensemble
```

## Output Format

### Standard Prediction
```
┌─────────────────────────────────────────────────────────────┐
│ Week 18: Kansas City @ Buffalo                              │
├─────────────────────────────────────────────────────────────┤
│ Model Prediction:     BUF -2.5                              │
│ Vegas Line:           BUF -1.0                              │
│ Edge:                 +1.5 (NO BET - below threshold)       │
│ Total Prediction:     48.5                                  │
│ Vegas Total:          47.0                                  │
│ Total Edge:           +1.5 (NO BET)                         │
│ Confidence:           [BUF -5.5, BUF +0.5] (90% interval)   │
└─────────────────────────────────────────────────────────────┘
```

### With Bet Recommendation
```
┌─────────────────────────────────────────────────────────────┐
│ Week 18: Las Vegas @ Denver                                 │
├─────────────────────────────────────────────────────────────┤
│ Model Prediction:     DEN -7.5                              │
│ Vegas Line:           DEN -3.0                              │
│ Edge:                 +4.5 ✓ ACTIONABLE                     │
│ Recommended:          BET DEN -3.0                          │
│ Win Probability:      62.3%                                 │
│ Kelly Bet Size:       2.1% of bankroll                      │
│ Expected Value:       +4.8%                                 │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Notes

When executing this command:

1. **Data Requirements**
   - Load latest features (up to game date)
   - Fetch current Vegas lines if --lines
   - Check for injury updates

2. **Prediction Protocol**
   - Generate point estimate (margin of victory, total)
   - Calculate prediction interval (90% CI)
   - Compute edge vs Vegas

3. **Betting Logic**
   - Only recommend bets when |edge| >= threshold
   - Calculate Kelly fraction with conservative multiplier
   - Show expected value and win probability

4. **Explanation Mode**
   - SHAP values for XGBoost
   - Key feature contributions
   - Model disagreements in ensemble

## Code Template

```python
from src.models import load_trained_model
from src.data import get_current_features, get_vegas_lines
from src.betting import calculate_edge, kelly_fraction, should_bet

def predict(
    scope: str,
    model_name: str = 'ensemble',
    threshold: float = 3.0,
    kelly_mult: float = 0.25,
    include_lines: bool = True,
    include_confidence: bool = False,
    explain: bool = False
):
    """Generate predictions with betting recommendations."""
    
    # Load model
    model = load_trained_model(model_name)
    
    # Get games to predict
    games = get_games_for_scope(scope)
    
    predictions = []
    for game in games:
        # Get features (point-in-time correct)
        features = get_current_features(game)
        
        # Generate prediction
        pred = model.predict(features)
        
        # Get Vegas line if requested
        vegas_line = get_vegas_lines(game) if include_lines else None
        
        # Calculate edge
        edge = calculate_edge(pred['margin'], vegas_line)
        
        # Betting recommendation
        if should_bet(edge, threshold):
            win_prob = model.predict_proba(features)
            bet_size = kelly_fraction(win_prob, kelly_mult=kelly_mult)
            recommendation = {
                'bet': True,
                'side': 'home' if edge > 0 else 'away',
                'size': bet_size,
                'win_prob': win_prob,
                'ev': calculate_ev(win_prob, -110)
            }
        else:
            recommendation = {'bet': False, 'reason': f'Edge {edge:.1f} < threshold {threshold}'}
        
        predictions.append({
            'game': game,
            'prediction': pred,
            'vegas': vegas_line,
            'edge': edge,
            'recommendation': recommendation
        })
    
    return predictions
```
