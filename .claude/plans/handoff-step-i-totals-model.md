# Handoff -- nfl_batchgineering Step I (totals model: over / under betting)

## Context for the fresh session

You are picking up the **NFL betting prediction system** at
`C:\Users\jasse\_GitHub_repositories\nfl_batchgineering`. Steps A-H
have shipped on `main`. Step I expands the **bet universe from spreads
only to spreads + totals**.

**The user stakes real money against this system. Reliability is the
#1 priority.** Totals are a separate model, not a tweak to the spread
model. Treat it as such.

### What's already done (do NOT redo)

- **Steps A-D**: tests, dbt guards, backtest harness, betting
  decision layer (spreads only).
- **Step E**: bet ledger + CLV tracking. Schema includes `side` for
  home/away; needs extension for over/under.
- **Step F**: champion-vs-challenger framework for spreads.
- **Step G**: pre-bet guardrails (bankroll persistence, stake caps,
  staleness, anomaly detection).
- **Step H**: weekly schedules, notifications, STATUS dashboard.

### Why Step I now

CLAUDE.md scope says "point spread AND totals betting." The system
currently only handles spreads. Totals roughly **double the bet
universe** (one bet per game becomes two), which lets the user spread
exposure across more independent events -- helpful for variance
reduction. Totals also have a different signal mix than spreads
(weather + pace matter much more than team strength), so they're
genuinely uncorrelated edge, not just more of the same.

---

## Step I scope

### I1. Totals target + feature emphasis

The mart `mart_game_prediction_features` already includes `vegas_total`
and `home_score + away_score` as the implicit target. New
`actual_total = home_score + away_score` derived field at the mart or
prep layer.

Feature emphasis for totals (different from spreads):

- **Pass-heavy:** offensive pass rate, defensive pass-EPA-allowed,
  explosive-play rates (totals are driven by chunk plays).
- **Pace:** plays per game, time-of-possession, no-huddle frequency.
- **Weather:** wind > 15mph and precipitation suppress scoring
  measurably. Existing `temp`, `wind`, `roof` columns are starting
  points; consider precipitation if the weather source has it.
- **Coaching tendencies:** pace differential, fourth-down
  aggressiveness, two-minute drill efficiency.

Most of these features already exist in the mart for spread purposes.
Step I doesn't need new feature engineering in v1 -- it needs the
existing features re-weighted for a different target. **Save new
feature work for after the baseline totals model is benchmarked.**

### I2. Train XGBoost totals model

Mirror `src/ml/models/spread_predictor.py` exactly. Same
hyperparameters initially (max_depth=4, lr=0.05, etc. per CLAUDE.md).
New module: `src/ml/models/total_predictor.py`.

The signal scale is very different: totals range 35-55 typically, vs
spreads -14 to +14. Don't blindly reuse the spread model's logistic
scale (5.5). For totals win prob you need a different mapping --
empirically derive from training data what spread (`pred_total -
vegas_total`) corresponds to what win rate. v1: skip win-prob entirely
for totals, just use point edge directly for sizing (treat -110 as
break-even probability of 52.4% and don't try to estimate true
calibrated prob).

### I3. Backtest harness extension

Add `backtest_totals_xgboost` to the `BACKTEST_REGISTRY` in
`src/ml/backtest.py`. The ATS-accuracy logic in `_ats_accuracy` is
spread-specific (it compares predicted_margin to actual_margin); write
the totals analog: did pred_total beat vegas_total in the direction
the model predicted?

Edge convention for totals:

- `total_edge = pred_total - vegas_total`
- Positive edge -> bet OVER. Negative -> bet UNDER.
- Same 4.0pt threshold as Step D spreads. Re-validate this is the
  right threshold for totals after the first backtest -- totals tend
  to have noisier scoring distributions so the threshold may need to
  be 5+ for totals.

Per-fold metrics tracked separately from spreads. The validation
report adds new rows.

### I4. Totals recommendation logic

Extend `src/betting/recommend.py`:

- `BetRecommendation.side` enum gains `'over'` and `'under'` in
  addition to `'home'/'away'/'pass'`. Or introduce a parallel
  `TotalRecommendation` dataclass -- discuss with user. Recommend
  unified `BetRecommendation` with a `bet_type` field
  (`'spread' | 'total'`) so the ledger schema stays clean.
- New CLI flag `--include-totals` (default true once shipped) so the
  weekly recommendation produces both bet types.
- Guardrails (Step G) apply identically: staleness, anomaly, stake
  cap. The cap now spans both bet types -- 10% of bankroll for the
  whole week, not 10% per bet type.

### I5. Schema updates

Step E's `BETS` table gains `bet_type` and renames `vegas_spread_*`
to `vegas_line_*` (or keep both columns, one nullable). The CLV
formula extends to totals (positive CLV = line moved in your favor
after you bet; same conceptual definition, different sign mechanics).

Document the CLV-for-totals derivation in the same place as the
CLV-for-spreads derivation in Step E. Unit-test both.

`mart_bet_performance` gains a `bet_type` partition so spread ROI and
totals ROI are tracked separately (and combined).

### I6. Champion-vs-challenger for totals

Step F's framework extends naturally: the totals model has its own
champion. They're separate registry entries
(`bet_type='spread'` vs `'total'`). Don't make a "stacked
spread+total" model -- they're different problems.

### I7. Tests

- `tests/test_totals_backtest.py` -- walk-forward over synthetic
  totals data, sign correctness on the totals edge calculation.
- `tests/test_totals_recommendation.py` -- over/under labeling,
  stake sizing, end-to-end on a synthetic frame including bet_type.
- `tests/test_totals_clv.py` -- canonical CLV cases for totals.
- Extension of `tests/test_betting.py` to assert spread and totals
  recommendations don't interfere (a week with both types produces
  the right counts, the cap spans both).

---

## Project conventions (non-negotiable)

Same as Steps A-H. Specifically:

- **No deep learning**. Totals modeling is also bounded by 5k games.
- **Brier on totals win-prob if you compute one**; otherwise use RMSE
  vs actual total as the primary scoring metric (totals are
  regression, not classification).
- **Polars over Pandas**.
- **Walk-forward CV only**.
- **Reuse, don't fork**. Totals share data loading, Snowflake auth,
  bet ledger schema (with `bet_type` extension), guardrails, and
  scheduling infrastructure with spreads.

## Critical files to read first

- `src/ml/models/spread_predictor.py` -- template for
  `total_predictor.py`.
- `src/ml/backtest.py` -- `BACKTEST_REGISTRY`, `_ats_accuracy`.
- `dbt_project/models/3_marts/mart_game_prediction_features.sql` --
  confirm `vegas_total`, `home_score`, `away_score` are present and
  not look-ahead.
- `src/betting/recommend.py` -- the `BetRecommendation` shape that
  Step E's ledger schema mirrors.
- Step E's `BETS` schema -- decide where `bet_type` slots in.
- `CLAUDE.md` feature-engineering rules (already lists totals-relevant
  features under "Situational" and "Context").

## Out of scope for Step I (deferred)

- Player props (anytime TD, rushing yards, receiving yards). Very
  different problem -- many more markets, smaller samples, sharper
  books. Step J+ candidate.
- Alt lines (e.g., over/under at non-standard totals). Same product
  area but separate market.
- Live in-game betting. Needs streaming data; out of scope for this
  architecture.
- Teasers, parlays. The expected value math on these in NFL is
  negative; don't build it.
- Quarter / half totals. Same justification as alt lines.

---

## First actions in the fresh session

1. `git log -5 --oneline`, `git status`, confirm state.
2. Confirm `vegas_total`, `home_score`, `away_score` exist in
   `mart_game_prediction_features` and that there's enough history
   to walk-forward through.
3. Read Step E's final `BETS` schema -- the `bet_type` extension
   needs to fit cleanly.
4. **Ask the user**: "Unified `BetRecommendation` with a `bet_type`
   field, or parallel `TotalRecommendation` dataclass?" Recommend
   unified for ledger / mart simplicity, but the user may have a
   downstream reason to want them separated.
5. Ask the user about the totals edge threshold: keep 4.0pt or
   tighten further given totals' noisier distribution? Default to
   5.0pt and let them push back.
6. Run the existing spread backtest at edge=4.0 as a sanity check
   that the harness still produces the same numbers (regression
   guard before extending it).
7. Enter plan mode and draft I1-I7.

## Verification

- `python src/ml/backtest.py --models xgboost,totals_xgboost`
  produces both per-fold tables in the markdown report.
- The totals model clears (or fails to clear) the 0.524 target at
  the chosen threshold; either way the result is reported and
  committed for review.
- `weekly_bet_recommendations` produces both spread and totals
  recommendations for a synthetic week; bet ledger gets the
  correct `bet_type` value on every row.
- Guardrails (Step G) fire correctly when totals + spreads
  together exceed the weekly cap.
- CLV computed correctly for an OVER bet whose line dropped (line
  moved in your favor) and for an UNDER bet whose line rose.
- Full test suite green.
