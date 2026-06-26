# Handoff -- nfl_batchgineering Step F (model iteration framework + first challenger)

## Context for the fresh session

You are picking up the **NFL betting prediction system** at
`C:\Users\jasse\_GitHub_repositories\nfl_batchgineering`. Steps A-E
have shipped on `main`. Step F is the **model iteration framework**:
champion-vs-challenger evaluation, plus shipping the first real
challenger to the borderline-positive XGBoost baseline.

**The user stakes real money against this system. Reliability is the
#1 priority.** When in doubt, ask. When you'd normally pause, make the
reasonable call and continue.

### What's already done (do NOT redo)

- **Step A+B** (merged): pytest, dbt guards, look-ahead-bias proofs.
- **Step C** (merged): walk-forward backtest harness
  (`src/ml/backtest.py`), baseline XGBoost ATS_edge **0.5269**,
  ROI **+0.89%**, 5/7 folds clearing the 0.524 target.
- **Step D** (merged): `src/betting/` decision layer with 4.0pt edge
  threshold and 0.25x Kelly. `weekly_bet_recommendations` Dagster asset.
- **Step E** (merged): `BETS`, `BET_CLOSING_LINES`, `BET_RESULTS`
  tables + `mart_bet_performance` view. `model_version` column on
  every bet so champion/challenger results stay separable. CLV tracked
  per bet.

### Why Step F now

The baseline is borderline-positive. Vegas adapts. You will eventually
need a candidate model that beats XGBoost, and the harness to prove it
beats XGBoost before promoting it. Step F builds the muscle of *swap
models safely* before the model decays in live betting, plus ships the
first real candidate to start the comparison.

---

## Step F scope

### F1. Champion registry

Persisted record of which model is currently the production champion.
Two options:

- **File-based** (simpler v1): `src/ml/models/CHAMPION.json` with
  `{model_name, version, trained_at, ats_edge_at_promotion,
   roi_at_promotion}`. `weekly_bet_recommendations` reads it on
  startup.
- **Snowflake-based**: `PRODUCTION_ANALYTICS.ML.CHAMPION_REGISTRY` table
  with full history. Audit trail for "what model picked this bet?"

Recommend Snowflake. Bet ledger already references `model_version`;
giving it a foreign key to a registry table makes the audit story
clean.

### F2. Challenger evaluation flow

New CLI + Dagster asset that:

1. Trains the challenger using the same walk-forward folds as the
   existing `backtest_xgboost` in `src/ml/backtest.py`.
2. Produces a side-by-side comparison report: per-fold ATS_edge,
   Brier, ROI, RMSE. Aggregate metrics. **Paired bootstrap test on
   per-fold ROI** (10k resamples) -- gives a p-value for "challenger
   ROI > champion ROI" without assuming normality on tiny samples.
3. Surfaces calibration curves so a challenger that ties on ATS but
   wins on Brier is visible.

The existing `BACKTEST_REGISTRY` in `src/ml/backtest.py:408` already
supports multiple models. Extend that registry; don't fork the harness.

### F3. Promotion gate

A challenger may only be promoted if **all** of:

- Challenger aggregate ATS_edge >= champion ATS_edge + 0.005 (half a
  percentage point margin to discount noise), AND
- Challenger aggregate ATS_edge >= 0.524 (clears -110 juice), AND
- Paired-bootstrap p-value on per-fold ROI <= 0.10 (one-sided), AND
- Challenger Brier <= champion Brier + 0.005 (no calibration
  regression), AND
- Challenger clears ATS target on at least as many folds as champion.

These are deliberately strict because the cost of a bad promotion
(live money on a worse model) >> the cost of holding a slightly-better
challenger back a few weeks. Make the thresholds config-driven so the
user can tune them.

### F4. First challenger candidate -- calibrated stacked ensemble

The architecture diagram in `CLAUDE.md` already calls for stacking
XGBoost + Bayesian + Neural + Elo with a ridge meta-learner. A
Bayesian state-space stub exists at `src/ml/models/bayesian.py` (check
implementation state -- it may be a skeleton).

First challenger to try:

1. **Stack** XGBoost (current) + Elo (current) + Bayesian state-space
   (Glickman-Stern, per CLAUDE.md). Ridge meta-learner on top, fit
   on out-of-fold predictions to avoid stacking leakage. Skip the
   neural net for v1 -- 5k games is too few for it to add real signal
   and it complicates the pipeline.
2. **Calibration layer**: Platt scaling (logistic regression on raw
   stacked output) or isotonic regression. CLAUDE.md emphasizes
   Brier-score optimization; this is the lever for it.
3. Train through the existing walk-forward folds. No new feature
   engineering in this step -- prove the stacking + calibration
   alone moves the needle before adding signal.

If the stacking challenger does NOT clear the gate, escalate to the
user before adding features. The user explicitly wants a go/no-go
checkpoint between model iterations.

### F5. Live shadowing

Once the champion is locked, also run the challenger silently each
week:

- `weekly_bet_recommendations` produces a `staked=true` bet row tagged
  with the champion's `model_version`.
- A parallel run produces a `staked=false` bet row tagged with the
  challenger's `model_version` (or a separate column to keep the
  shadow set obviously labeled -- decide based on `BETS` schema from
  Step E).
- Both run through the full reconciliation + CLV pipeline so
  `mart_bet_performance` can compare them on identical real-world
  weeks. After ~50 shadow bets you have a live-data signal layered on
  top of the backtest evidence.

### F6. Tests

- `tests/test_promotion_gate.py` -- every gate threshold exercised on
  synthetic results (challenger wins on every dim, loses on each dim
  one at a time, ties).
- `tests/test_bootstrap.py` -- known-distribution sanity check on the
  paired-bootstrap p-value implementation.
- `tests/test_ensemble.py` -- stacking meta-learner is fit on
  out-of-fold predictions, not in-fold (regression guard against
  leakage).

---

## Project conventions (non-negotiable)

Same as Steps A-E. Specifically for this step:

- **No deep learning architectures** (CLAUDE.md). LSTM/transformer
  candidates are off the table; sample size kills them.
- **Maximum 3 hidden layers** if a NN is part of any future stack.
- **Walk-forward CV only**; never random splits.
- **Polars over Pandas** in pipeline code; sklearn / pymc accept numpy
  arrays -- keep the Polars -> numpy conversion explicit and minimal.
- **Brier score is primary**; ATS accuracy is secondary. A challenger
  that wins on Brier and ties on ATS is still an improvement.
- **`from __future__ import annotations` rule**: fine in modeling code
  but NOT in any file defining a Dagster `Config` subclass.

## Critical files to read first

- `src/ml/backtest.py` -- entire file. `BACKTEST_REGISTRY` is where
  challengers register. `_walk_forward_season_splits` enforces temporal
  ordering; reuse it.
- `src/ml/models/spread_predictor.py` (or wherever the XGBoost model
  lives) -- the champion you're trying to beat.
- `src/ml/models/elo_model.py` -- the existing Elo baseline. Useful as
  a stack member.
- `src/ml/models/bayesian.py` -- check state. Glickman-Stern
  implementation may need to be completed before F4 can ship.
- `src/ml/models/ensemble.py` -- check state. Stacking meta-learner
  may also be a stub.
- `src/ml/utils/validation.py` -- `calculate_brier_score`,
  `calculate_roi`. Reuse these in the comparison harness.
- The Step E `BETS` + `mart_bet_performance` schema (whatever shape it
  ended up in) -- needed for F5 shadowing.

## Out of scope for Step F (deferred)

- New features / data sources. The first challenger must beat the
  baseline using the existing feature set; that proves the
  *architecture* helps. New features are for a later iteration.
- Online learning / continuous training. Train weekly at most.
- AutoML / hyperparameter search beyond a small grid -- 5k games
  doesn't support aggressive HPO without overfitting.
- Pre-bet guardrails, automation, totals model.

---

## First actions in the fresh session

1. `git log -5 --oneline`, `git status`, confirm state.
2. **Read `src/ml/models/bayesian.py` and `src/ml/models/ensemble.py`
   FIRST**. The challenger plan assumes these exist as working
   implementations. If they're stubs, the actual first step of Step F
   is finishing those modules, not building the promotion harness.
   Surface this to the user before drafting the rest of the plan.
3. Read the Step E `BETS` schema and `mart_bet_performance` -- F5
   shadowing depends on its column layout.
4. Run the existing backtest (`python src/ml/backtest.py
   --edge-threshold 4.0`) and capture the champion's per-fold numbers
   as the comparison baseline for the new harness.
5. Ask the user: "If the first challenger doesn't clear the gate, do
   you want to (a) add new features, (b) try a different ensemble
   composition, or (c) call it and hold the XGBoost baseline?"
6. Enter plan mode and draft F1-F6.

## Verification

- `python src/ml/compare_models.py --challenger stacked_calibrated`
  (or whatever you name the CLI) produces a side-by-side report with
  bootstrap p-values.
- All promotion gate thresholds are exercised by unit tests.
- If the first challenger passes the gate: champion registry is
  updated, `weekly_bet_recommendations` picks up the new model on the
  next materialization, and the previous champion enters shadow mode.
- If it fails: report is still produced and committed to
  `reports/comparisons/` so the failure is auditable, and the user
  decides next steps.
- Full test suite remains green.
