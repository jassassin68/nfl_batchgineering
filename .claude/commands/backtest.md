# /backtest - Historical Backtesting Command

## Description
Run historical backtests to evaluate model performance and betting strategy profitability.

## Usage
```
/backtest [period] [options]
```

## Period
- `season [year]` - Single season backtest
- `range [start]-[end]` - Multi-season backtest
- `recent [n]` - Last n games

## Options
- `--model` - Model to test (`ensemble`, `xgboost`, etc.)
- `--threshold` - Edge threshold (default: 3.0)
- `--kelly` - Kelly multiplier (default: 0.25)
- `--bankroll` - Starting bankroll (default: 10000)
- `--markets` - Markets to test (`spread`, `total`, `both`)
- `--detailed` - Game-by-game breakdown
- `--compare` - Compare multiple models
- `--simulate` - Monte Carlo simulation (n=1000)

## Examples

### Backtest 2023 season with spread bets
```
/backtest season 2023 --markets spread --threshold 3.0
```

### Compare models over multiple seasons
```
/backtest range 2019-2023 --compare xgboost,bayesian,ensemble
```

### Detailed breakdown with simulation
```
/backtest season 2024 --detailed --simulate
```

## Output Format

### Summary Report
```
┌─────────────────────────────────────────────────────────────┐
│ BACKTEST RESULTS: 2019-2023 Seasons                         │
│ Model: Ensemble | Threshold: 3.0 pts | Kelly: 0.25x         │
├─────────────────────────────────────────────────────────────┤
│ SPREAD BETTING                                              │
│   Total Bets:          187 / 1,350 games (13.9% bet rate)   │
│   Record:              102-85 (54.5% ATS)                   │
│   Profit:              +$2,847 (+28.5% ROI)                 │
│   Max Drawdown:        -$1,203 (-12.0%)                     │
│   Sharpe Ratio:        1.42                                 │
├─────────────────────────────────────────────────────────────┤
│ TOTALS BETTING                                              │
│   Total Bets:          143 / 1,350 games (10.6% bet rate)   │
│   Record:              79-64 (55.2% O/U)                    │
│   Profit:              +$1,923 (+19.2% ROI)                 │
│   Max Drawdown:        -$876 (-8.8%)                        │
│   Sharpe Ratio:        1.28                                 │
├─────────────────────────────────────────────────────────────┤
│ COMBINED                                                    │
│   Total Profit:        +$4,770 (+47.7% ROI)                 │
│   Final Bankroll:      $14,770                              │
│   CAGR:                8.1%                                 │
└─────────────────────────────────────────────────────────────┘
```

### Model Comparison
```
┌─────────────────────────────────────────────────────────────┐
│ MODEL COMPARISON: 2019-2023                                 │
├─────────────────────────────────────────────────────────────┤
│ Model       │ ATS %  │ ROI    │ Sharpe │ Bets    │ Edge    │
├─────────────────────────────────────────────────────────────┤
│ Ensemble    │ 54.5%  │ +28.5% │ 1.42   │ 187     │ 4.2 avg │
│ XGBoost     │ 53.8%  │ +21.3% │ 1.18   │ 203     │ 3.9 avg │
│ Bayesian    │ 53.2%  │ +17.8% │ 1.05   │ 178     │ 4.0 avg │
│ Elo         │ 51.9%  │ +4.2%  │ 0.45   │ 156     │ 3.7 avg │
│ Vegas       │ 50.0%  │ -4.5%  │ -0.48  │ 1350    │ 0.0     │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Notes

When executing this command:

1. **Walk-Forward Protocol (CRITICAL)**
   - For each game, use only data available BEFORE that game
   - Re-train model at start of each season (or use pre-trained)
   - This simulates realistic betting scenario

2. **Bet Simulation Rules**
   - Apply edge threshold strictly
   - Use opening lines (not closing)
   - Calculate bet size via Kelly with conservative multiplier
   - Track cumulative bankroll

3. **Metrics to Calculate**
   ```
   - ATS Accuracy (wins / total bets)
   - ROI = (profit / total_wagered) * 100
   - Sharpe = mean(returns) / std(returns) * sqrt(n_bets)
   - Max Drawdown = max peak-to-trough decline
   - CAGR = (final/initial)^(1/years) - 1
   - Bet Rate = bets / total_games
   - Average Edge = mean(abs(edge) for bets placed)
   ```

4. **Monte Carlo Simulation**
   - Bootstrap sample from historical bets
   - Simulate 1000+ bankroll paths
   - Report: median outcome, 5th/95th percentile, ruin probability

5. **Validation Checks**
   - Verify no look-ahead bias in features
   - Check that bet dates match feature dates
   - Confirm opening lines used (not closing)

## Code Template

```python
from src.models import load_trained_model
from src.data import load_historical_games, get_opening_lines
from src.betting import calculate_edge, kelly_fraction, should_bet
from src.validation import walk_forward_iterator

def backtest(
    period: str,
    model_name: str = 'ensemble',
    threshold: float = 3.0,
    kelly_mult: float = 0.25,
    starting_bankroll: float = 10000,
    markets: str = 'both',
    detailed: bool = False,
    compare: list[str] | None = None,
    simulate: bool = False
):
    """Run historical backtest with betting simulation."""
    
    # Parse period
    start_season, end_season = parse_period(period)
    
    # Load historical data
    games = load_historical_games(start_season, end_season)
    
    # Initialize tracking
    bankroll = starting_bankroll
    bets = []
    results = []
    
    # Walk-forward through games
    for game, features in walk_forward_iterator(games):
        
        # Load model (trained on data before this game)
        model = get_model_for_date(model_name, game['date'])
        
        # Generate prediction
        pred = model.predict(features)
        
        # Get opening line
        vegas_line = get_opening_lines(game)
        
        # Calculate edge
        spread_edge = calculate_edge(pred['margin'], vegas_line['spread'])
        total_edge = calculate_edge(pred['total'], vegas_line['total'])
        
        # Spread bet
        if markets in ['spread', 'both'] and should_bet(spread_edge, threshold):
            bet = place_bet(
                game=game,
                market='spread',
                edge=spread_edge,
                kelly_mult=kelly_mult,
                bankroll=bankroll
            )
            bets.append(bet)
            
            # Settle bet
            result = settle_spread_bet(bet, game['actual_margin'])
            bankroll += result['profit']
            results.append(result)
        
        # Total bet (similar logic)
        if markets in ['total', 'both'] and should_bet(total_edge, threshold):
            # ... similar to spread
            pass
    
    # Calculate summary metrics
    summary = calculate_backtest_metrics(results, starting_bankroll)
    
    if simulate:
        monte_carlo = run_monte_carlo(bets, n_simulations=1000)
        summary['monte_carlo'] = monte_carlo
    
    if compare:
        comparison = run_comparison(period, compare, threshold, kelly_mult)
        summary['comparison'] = comparison
    
    return summary

def calculate_backtest_metrics(results: list, starting_bankroll: float) -> dict:
    """Calculate comprehensive backtest metrics."""
    wins = sum(1 for r in results if r['won'])
    losses = len(results) - wins
    total_wagered = sum(r['wager'] for r in results)
    profit = sum(r['profit'] for r in results)
    
    returns = [r['profit'] / r['wager'] for r in results]
    
    return {
        'record': f'{wins}-{losses}',
        'win_rate': wins / len(results) if results else 0,
        'total_wagered': total_wagered,
        'profit': profit,
        'roi': (profit / total_wagered) * 100 if total_wagered else 0,
        'sharpe': np.mean(returns) / np.std(returns) * np.sqrt(len(returns)),
        'max_drawdown': calculate_max_drawdown(results),
        'final_bankroll': starting_bankroll + profit
    }
```

## Critical Reminders

1. **NEVER** use closing lines - always opening lines
2. **NEVER** use features calculated from future data
3. **ALWAYS** use walk-forward validation
4. **BE SKEPTICAL** of results showing >55% accuracy
5. **CHECK** for data leakage if results seem too good
