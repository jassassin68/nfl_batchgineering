# Handoff -- nfl_batchgineering Step E (CLV + live bet ledger)

## Context for the fresh session

You are picking up the **NFL betting prediction system** at
`C:\Users\jasse\_GitHub_repositories\nfl_batchgineering`. Steps A-D
have shipped on `main`. Step E is the **data foundation for measuring
whether the system is actually winning in production**.

**The user stakes real money against this system. Reliability is the
#1 priority.** When in doubt, ask. When you'd normally pause, make the
reasonable call and continue -- the user will redirect if needed.

### What's already done (do NOT redo)

- **Step A+B** (merged): pytest harness, dbt data-quality guards,
  look-ahead-bias proofs, Windows emoji crash fix.
- **Step C** (merged): walk-forward backtest harness (`src/ml/backtest.py`),
  `model_validation_report` Dagster asset. Baseline XGBoost ATS_edge
  0.5269, ROI +0.89% across 7 folds (borderline-positive).
- **Step D** (merged): `src/betting/` package with `calculate_edge`,
  `kelly_fraction`, `build_recommendations`. `weekly_bet_recommendations`
  Dagster asset. CLI at `src/betting/recommend.py`. Edge threshold
  tightened to 4.0pt, Kelly held at 0.25x. Fixed historical
  `bet_recommendation` inversion bug.

### Why Step E now

Without a record of placed bets and closing-line movement, the user is
flying blind in production. ATS hit rate over a half-season is
statistical noise; CLV gives a real edge signal in ~50 bets. Every
future improvement (model iteration, calibration drift detection,
champion vs challenger comparison) needs this table to exist first.

---

## Step E scope

### E1. Bet ledger schema

New Snowflake tables (or dbt-managed marts):

```
PRODUCTION_ANALYTICS.ML.BETS                  -- one row per recommended bet at recommendation time
  bet_id (uuid)
  game_id
  season, week
  recommended_at (timestamp, UTC)
  side ('home' | 'away')
  predicted_spread, vegas_spread_at_rec       -- the spread at recommendation time
  edge_points
  win_prob, kelly_fraction, stake_units
  bankroll_at_rec
  odds_at_rec
  model_version                               -- so champion vs challenger results stay separable
  staked (bool, default true)                 -- user can flip to false if they decided not to take it

PRODUCTION_ANALYTICS.ML.BET_CLOSING_LINES     -- one row per game, snapshotted ~5 min before kickoff
  game_id
  vegas_spread_close
  captured_at (timestamp)
  source                                       -- which odds feed

PRODUCTION_ANALYTICS.ML.BET_RESULTS            -- one row per resolved bet
  bet_id
  home_score, away_score
  margin (home - away)
  vegas_spread_close                           -- joined from BET_CLOSING_LINES
  outcome ('win' | 'loss' | 'push')
  profit_units                                 -- in same units as stake_units
  clv_points                                   -- (vegas_spread_close - vegas_spread_at_rec) * side_multiplier
  resolved_at
```

CLV sign convention: positive CLV = line moved in your favor after you
bet. If you bet HOME at -3 and it closed -4 (home got more favored),
CLV = +1 for the HOME bettor. If you bet AWAY at the same line and it
closed -4, CLV = -1 against you. Be careful with signs -- the handoff
for this work should re-derive the formula from scratch and unit-test
it on canonical cases.

### E2. Recording layer

Modify `src/betting/recommend.py::build_recommendations` (or add a
wrapper) so every produced recommendation is also persisted to
`BETS`. Keep `staked=True` as the default so the user only has to
intervene when *skipping* a recommended bet (default-yes is safer than
default-no for measuring system performance).

Reuse the Snowflake connection pattern from
`src/ml/predict.py::get_snowflake_connection` and the write pattern in
`write_to_snowflake`. Do not introduce a new connection helper -- keep
one source of truth for Snowflake auth.

### E3. Closing-line capture job

New Dagster asset / job that runs ~5 minutes before each game's
kickoff and writes a row to `BET_CLOSING_LINES`. Two design choices to
make early:

- **Source.** The current vegas line source is in
  `dbt_project/models/2_intermediate/int_game_vegas_lines.sql` -- find
  out whether it pulls history (so closing line is already in the
  warehouse) or only the latest snapshot. If only latest, you need a
  live odds-feed integration (The Odds API is in CLAUDE.md as a
  candidate; check what credentials, if any, already exist in `.env`).
- **Schedule.** Per-game timing means dynamic partitioning by game_id
  with a sensor that fires N minutes before each game's `gametime`.
  Simpler v1: a single schedule that runs Sunday 12:55 PM ET (5 min
  before the main slate) and captures every game starting in the next
  4 hours. Accept that Thursday/Monday/Saturday games need their own
  schedule entries.

### E4. Result reconciliation

New Dagster asset that runs Tuesday morning, scans `BETS` rows older
than 24 hours without a matching `BET_RESULTS` row, joins to final
scores from `mart_game_prediction_features` (which already has
`home_score`/`away_score` once games complete), computes outcome and
CLV, writes `BET_RESULTS`.

### E5. Performance mart

```
PRODUCTION_ANALYTICS.ML.MART_BET_PERFORMANCE   -- dbt view
  -- rolling windows: last_4_weeks, last_8_weeks, season_to_date, all_time
  -- metrics: n_bets, win_rate, roi, brier (vs win_prob), avg_clv, clv_positive_rate
  -- broken down by model_version so champion vs challenger comparisons stay clean
```

Build this as a dbt view in `dbt_project/models/3_marts/`. Mirror the
column/test conventions in `_marts__models.yml`.

### E6. Tests + CLI for inspection

- `tests/test_clv.py` -- canonical-case unit tests on the CLV formula
  (positive when line moved in your favor, sign correctness for both
  home and away bets, push handling).
- `tests/test_recording.py` -- mocked Snowflake write; verify
  `build_recommendations` produces a bet row per non-pass recommendation
  with correct `model_version` tagging.
- `src/betting/performance.py` -- thin CLI that queries
  `MART_BET_PERFORMANCE` and prints a markdown summary. Same Windows
  cp1252 ASCII rule as the rest of the betting layer.

---

## Project conventions (non-negotiable)

From `CLAUDE.md` and prior PRs:

- **Look-ahead bias**: never use future data. Point-in-time only.
- **Calibration > accuracy**: Brier score is the primary metric.
- **Polars over Pandas**. Type hints on all functions.
- **Windows cp1252 console**: ASCII-only in `print()`. Reuse the
  `sys.stdout.reconfigure` pattern in `src/betting/recommend.py`.
- **Snowflake key-pair auth** via `~/.dbt/profiles.yml`. Reuse
  `get_snowflake_connection` from `predict.py`. Don't print the token.
- **Never commit**: `.env`, `secrets.json`, `profiles.yml`, any pickle
  outside `src/ml/models/**`.
- **gh CLI is NOT installed**. PRs opened manually with compare URL.
- **`.venv` lives at the repo root**, not the worktree.
- **Conventional commits**, PR required for `main`.
- **Dagster Config classes**: NO `from __future__ import annotations`
  in any file defining a `Config` subclass (PEP 563 breaks introspection).
  See header comment in `dagster_project/assets/validation.py`.
- **Single source of truth**: any logic shared between CLI, Dagster asset,
  and library code lives in `src/` and is imported by all three.

## Critical files to read first

- `src/betting/recommend.py` -- the recording layer wraps this.
- `src/ml/predict.py:504-567` -- Snowflake write pattern to mirror.
- `dbt_project/models/2_intermediate/int_game_vegas_lines.sql` --
  closing-line source question.
- `dbt_project/models/3_marts/mart_game_prediction_features.sql` --
  joining final scores back to bets for reconciliation.
- `dagster_project/assets/validation.py` -- Dagster Config + asset
  pattern.
- `dagster_project/schedules/weekly_schedule.py` -- existing schedule
  pattern to mirror for closing-line and reconciliation jobs.

## Out of scope for Step E (deferred)

- Model iteration (Step F).
- Pre-bet guardrails (Step G).
- Weekly automation tying everything together (Step H).
- Totals model (Step I).

---

## First actions in the fresh session

1. `git log -5 --oneline` and `git status` to confirm state.
2. Read `int_game_vegas_lines.sql` and determine whether closing lines
   are already captured. **This is the biggest unknown** and shapes
   E3's design.
3. Ask the user: "Do you want every recommended bet auto-recorded as
   staked, or should there be a confirmation step?" Default to
   auto-record-staked unless they say otherwise.
4. Ask the user: "What odds feed do you actually plan to use in
   production for closing lines?" (Determines E3 integration shape.)
5. Enter plan mode, draft E1-E6 against the current codebase, get
   approval, then implement.

## Verification

- A `weekly_bet_recommendations` materialization writes N rows to `BETS`
  with `model_version` set.
- The closing-line job populates `BET_CLOSING_LINES` for every game in
  this week's slate.
- After games complete, the reconciliation job populates `BET_RESULTS`
  with non-null `clv_points`.
- `python src/betting/performance.py` prints a coherent summary.
- New tests pass; full suite remains green.
- Snowflake row counts: `BETS` >= sum of weekly recommendations,
  `BET_RESULTS` matches `BETS` once games complete (no orphans).
