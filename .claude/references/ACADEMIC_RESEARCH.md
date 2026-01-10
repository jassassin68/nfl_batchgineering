# Academic Research Reference

## Core Papers & Sources

### 1. Glickman & Stern (1998) - State-Space NFL Model
**Citation**: Glickman, M.E. & Stern, H.S. (1998). A State-Space Model for National Football League Scores. Journal of the American Statistical Association, 93(441), 25-35.

**URL**: https://glicko.net/research/nfl-chapter.pdf

**Key Contributions**:
- Team strength modeled as latent AR(1) process
- Two variance components: week-to-week (τ²) and season-to-season (σ²)
- Home field advantage as fixed effect (~2.5 points)
- Bayesian inference via MCMC

**Model Specification**:
```
Y_ij = θ_i - θ_j + h + ε_ij

where:
  Y_ij = point differential (home - away)
  θ_i = home team strength (latent)
  θ_j = away team strength (latent)
  h = home field advantage
  ε_ij ~ N(0, σ²_game)

Team strength evolution:
  θ_i,t+1 = ρ * θ_i,t + η_i,t
  η_i,t ~ N(0, τ²)
```

**Implementation Notes**:
- Use PyMC or Stan for Bayesian inference
- Estimate ρ (autocorrelation) from data
- Home field advantage has declined in recent years (~2.0-2.5 now)

---

### 2. Galekwa et al. (2024) - Systematic Review of ML in Sports Betting
**Citation**: Galekwa et al. (2024). A Systematic Review of Machine Learning in Sports Betting: Techniques, Challenges, and Future Directions.

**URL**: https://arxiv.org/html/2410.21484v1

**Key Findings**:
- Gradient boosted methods (XGBoost, LightGBM) dominate sports prediction
- NFL specifically benefits from ensemble methods and HMMs
- Feature engineering more important than model architecture
- Calibration optimization outperforms accuracy optimization

**Quoted**: "In American football, hidden Markov models and ensemble methods such as XGBoost have achieved high prediction accuracy."

---

### 3. Chen & Guestrin (2016) - XGBoost
**Citation**: Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16.

**Key Parameters for NFL**:
```python
xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 4,          # Shallow (small data)
    'learning_rate': 0.05,   # Low (prevent overfitting)
    'n_estimators': 200,     # With early stopping
    'reg_alpha': 0.1,        # L1 regularization
    'reg_lambda': 1.0,       # L2 regularization
    'subsample': 0.8,        # Row sampling
    'colsample_bytree': 0.8, # Feature sampling
}
```

---

### 4. Walsh & Joshi (2024) - Calibration vs Accuracy
**Finding**: Calibration-optimized models generate 69.86% higher average returns compared to accuracy-optimized models.

**Implication**: Optimize for Brier Score, not raw classification accuracy.

**Brier Score**:
```python
brier_score = mean((predicted_prob - actual_outcome)**2)
# Lower is better, 0 = perfect calibration
```

---

### 5. Gray & Gray (1997) - NFL Market Efficiency
**Citation**: Gray, P.K. & Gray, S.F. (1997). Testing Market Efficiency: Evidence from the NFL Sports Betting Market. Journal of Finance, 52(4), 1725-1737.

**Key Findings**:
- Average difference between spreads and actuals < 0.25 points
- Markets highly efficient but not perfectly so
- Betting underdogs >7 points showed 59.69% win rate (may be arbitraged now)

---

### 6. nfeloapp Research - EPA Weighting
**Finding**: Optimal weighting for predicting future net EPA:
- Offensive EPA: weight 1.6x
- Defensive EPA: weight 1.0x

**Rationale**: Offensive performance is more stable/predictive than defensive.

---

### 7. Zoltar System (Microsoft)
**Source**: Pure AI coverage of Microsoft Zoltar

**Key Insight**: Only bet when |model_margin - vegas_spread| > 4.0 points

**Performance**: Achieved ~65% ATS accuracy (exceptional, research context)

---

## Feature Engineering Research

### EPA (Expected Points Added)
**Source**: nflfastR / nflscrapR

**Definition**: The change in expected points (EP) from before a play to after.

**Calculation**:
```
EPA = EP_after - EP_before

where EP = f(down, distance, yard_line, time, score_diff)
```

**Best Practices**:
- Use per-play EPA, not total EPA (normalizes for pace)
- Separate pass EPA from rush EPA
- Rolling windows (4-game optimal per research)

### DVOA (Defense-adjusted Value Over Average)
**Source**: Football Outsiders

**Definition**: Compares team's success on each play to league average, adjusted for opponent and situation.

**Note**: Requires subscription or scraping; consider as secondary feature.

### Success Rate
**Definition**: Percentage of plays that gain 40%+ of needed yards on 1st down, 60%+ on 2nd down, or 100%+ on 3rd/4th down.

**Value**: More stable than EPA across small samples.

### CPOE (Completion Percentage Over Expected)
**Definition**: Actual completion % minus expected based on throw difficulty.

**Source**: nflfastR includes xpass_prob for calculation.

### RYOE (Rushing Yards Over Expected)
**Definition**: Actual rushing yards minus expected based on blocking/situation.

**Source**: nflfastR includes xpass_prob; rushing equivalent requires custom model.

---

## Market Efficiency Research

### Key Findings Across Literature

1. **NFL Spread Markets**: Among most efficient (~50% accuracy expected randomly)
2. **52.4% Break-Even**: Required accuracy at -110 juice
3. **Home Underdog Bias**: Slight inefficiency when home team is large underdog
4. **Public Money Bias**: Tendency to over-bet favorites/overs
5. **Weather Effects**: Cold/wind games may be underpriced unders

### Cautions

1. **Publication Bias**: Papers reporting 60%+ accuracy rarely replicate
2. **Temporal Decay**: Published edges disappear within 2-5 years
3. **Sample Size**: Claims require >1000 bet sample for significance
4. **Transaction Costs**: Academic studies often ignore juice, line movement

---

## Implementation Recommendations

### Model Selection Summary

| Model | Purpose | Complexity | Data Needs |
|-------|---------|------------|------------|
| Elo | Baseline, interpretable | Low | Minimal |
| XGBoost | Primary workhorse | Medium | Moderate |
| Bayesian State-Space | Dynamic team strength | High | Moderate |
| Shallow NN | Ensemble diversity | Medium | Moderate |
| LSTM/Transformer | NOT RECOMMENDED | Very High | Insufficient |

### Feature Priority

1. **Tier 1 (Essential)**
   - EPA/play (offense, defense, pass, rush)
   - Success rate
   - Rolling windows (4-game)

2. **Tier 2 (Important)**
   - Home field advantage
   - Rest differential
   - QB EPA/dropback

3. **Tier 3 (Helpful)**
   - DVOA (if available)
   - Weather conditions
   - Injury impact scores

4. **Tier 4 (Experimental)**
   - Market indicators (line movement, sharp money)
   - Social sentiment
   - Historical matchup features

### Validation Protocol

1. **Walk-Forward CV**: Train on past, test on future
2. **Out-of-Season**: Test on seasons not seen during development
3. **Bet Simulation**: Apply actual betting rules
4. **Monte Carlo**: Bootstrap confidence intervals

---

## Additional Resources

### Data Sources
- nflverse: https://github.com/nflverse/nflverse-data
- Football Outsiders: https://www.footballoutsiders.com
- Pro Football Reference: https://www.pro-football-reference.com
- The Odds API: https://the-odds-api.com

### Code Libraries
- nfl-data-py: Python wrapper for nflverse
- xgboost: Gradient boosting
- pymc: Bayesian modeling
- optuna: Hyperparameter optimization
- mlflow: Experiment tracking

### Related Papers (Further Reading)
- Lopez et al. (2018) - Cross-sport randomness comparison
- Warner (2010) - ML vs Vegas Line prediction
- Conrad (2024) - Quinnipiac capstone on NFL totals
