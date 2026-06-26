# Handoff -- nfl_batchgineering Step G (pre-bet guardrails for real money)

## Context for the fresh session

You are picking up the **NFL betting prediction system** at
`C:\Users\jasse\_GitHub_repositories\nfl_batchgineering`. Steps A-F
have shipped on `main`. Step G is **safety rails between model output
and real money leaving the bank account**.

**The user stakes real money against this system. Reliability is the
#1 priority.** Defense in depth. When in doubt, err toward refusing
the bet, not placing it.

### What's already done (do NOT redo)

- **Step A+B** (merged): pytest, dbt guards.
- **Step C** (merged): walk-forward backtest harness.
- **Step D** (merged): `src/betting/` decision layer. Edge 4.0pt,
  Kelly 0.25x.
- **Step E** (merged): bet ledger (`BETS`, `BET_CLOSING_LINES`,
  `BET_RESULTS`), CLV tracking, `mart_bet_performance`.
- **Step F** (merged): champion-vs-challenger framework, first
  calibrated stacked ensemble may or may not have replaced the
  XGBoost baseline.

### Why Step G now

Steps A-F prove the system *could* be profitable. Step G is what
prevents a single bad week, a stale Vegas line, a model anomaly, or a
user typo from blowing up the bankroll. Every guardrail here exists
because of a specific failure mode that has bankrupted real-money
betting systems in production.

---

## Step G scope

### G1. Bankroll persistence

Replace `--bankroll` CLI flag (current state after Step D) with a
Snowflake-backed source of truth.

```
PRODUCTION_ANALYTICS.ML.BANKROLL_LEDGER
  ts (timestamp)
  bankroll_units
  delta_units                  -- + for deposit/win, - for loss/withdraw
  reason                       -- 'weekly_settlement', 'deposit', 'withdrawal', 'manual_adjust'
  source                       -- 'reconciliation_job' | 'cli' | 'manual'
```

`build_recommendations` (and the Dagster asset) read the latest row to
size new bets. After Step E's reconciliation job updates
`BET_RESULTS`, it also appends a settlement row to
`BANKROLL_LEDGER`. The CLI `--bankroll` flag stays as an override for
testing but defaults to "read from ledger."

Why: a CLI flag means the user types the bankroll every week. One
typo, one stale value (forgot last week's losses), and Kelly sizes
against the wrong number. Persistence makes this impossible.

### G2. Max weekly stake cap

Hard ceiling on `sum(stake_units)` across all recommendations in a
week, regardless of what Kelly says. Reasonable v1:

- Default: 10% of current bankroll per week.
- Config-driven: `--max-weekly-stake-pct` and `--max-single-stake-pct`
  CLI flags / Dagster config.
- Behavior when triggered: proportionally scale down every bet's
  `stake_units` so the sum equals the cap. Log a warning to stdout and
  to a `risk_events` table.

Why: Kelly is provably optimal under correct probability estimates;
real systems have wrong probability estimates. A cap is the cheap
insurance against a week where the model has 8 "high-edge"
recommendations and they're all correlated.

### G3. Correlation detection

Refuse / flag simultaneous bets that are not independent:

- **Same-game correlations**: never an issue for spreads-only (one bet
  per game). Becomes critical when Step I (totals) and beyond ship --
  spread + total on the same game are correlated through pace.
- **Cross-game correlations**: 6 home favorites on the same Sunday is
  not 6 independent bets -- they share weather, referee bias, and
  national-momentum effects. Compute a simple correlation score
  (e.g., count of same-side bets weighted by spread tier) and if
  above threshold, scale stakes down or flag for human review.
- **Same-side cluster**: more than N% of weekly bets on one side
  (home or away) is a model anomaly red flag -- log it for review.

Start conservative: just LOG correlation metrics in the recommendation
report. Don't auto-refuse bets in v1. Once you have a few weeks of
data on what your model's correlation profile looks like, decide
thresholds.

### G4. Line-staleness check

A `vegas_spread_at_rec` recorded N hours ago may be a stale snapshot.
Sharp action moves NFL lines several points in the hour before
kickoff. Refuse to size against a vegas spread that is:

- More than 4 hours stale by default (config), OR
- Captured before any sharp-money window known to move the line
  (Saturday afternoon for Sunday games, etc.) -- v1 just uses the
  4-hour wall-clock test; v2 can get fancier.

This requires Step E's `BET_CLOSING_LINES` infrastructure to know what
"current" Vegas is. Hook into the same data path.

### G5. Anomaly detection

Before placing any bet, check whether the model's predicted spread is
within N standard deviations of historical Vegas-line distribution for
this matchup type. If a model is predicting a 21-point spread and
Vegas has it at -3, something is wrong (feature pipeline broken,
training data corrupted, etc.) and the safest move is to refuse the
bet and alert.

Metrics to track:

- `|predicted_spread - vegas_spread| > 14` (more than two TDs of
  disagreement)
- `|predicted_spread| > 17` (predicted blowout that Vegas isn't
  pricing)
- `win_prob > 0.85 or win_prob < 0.15` (overconfidence beyond what
  NFL games support)

Any trigger: log to `risk_events`, set `staked=false` on the
recommended bet, surface in the weekly report with a human-readable
reason.

### G6. Risk events table + CLI

```
PRODUCTION_ANALYTICS.ML.RISK_EVENTS
  ts
  event_type                  -- 'stake_cap_triggered', 'stale_line_refused', 'anomaly_blocked', ...
  game_id (nullable)
  details (variant / json)
  resulted_in_skip (bool)
```

CLI: `python src/betting/risk.py --since 30d` -- summary of triggers,
how many bets were modified or skipped, which guardrails fired most.

### G7. Tests

- `tests/test_bankroll.py` -- ledger read / write, settlement
  arithmetic, override behavior.
- `tests/test_stake_caps.py` -- proportional scaling when cap is
  triggered, single-bet cap takes precedence, zero-stake edge case.
- `tests/test_staleness.py` -- staleness check honors clock, refuses
  >4h old lines, allows fresh lines.
- `tests/test_anomaly.py` -- each anomaly trigger fires when it
  should, doesn't fire when it shouldn't.
- `tests/test_risk_events.py` -- events get logged on trigger.

---

## Project conventions (non-negotiable)

Same as Steps A-F. Specifically:

- **Refuse > allow**: any guardrail's default behavior on uncertainty
  is to skip the bet, not place it.
- **Loggable**: every guardrail trigger MUST write a row to
  `risk_events`. No silent skips.
- **Config-driven thresholds**: every numerical threshold here (4
  hours, 10% bankroll, 14-point disagreement) is a config knob. The
  user will tune these once they have a few weeks of live data.
- **Idempotent**: a bet recommendation re-run for the same week must
  produce the same guardrail decisions given the same input data.
  Test for this explicitly.

## Critical files to read first

- `src/betting/recommend.py` -- guardrails attach here.
- `src/betting/kelly.py` -- sizing already clamps to [0, 1]; G2 is
  another layer on top.
- Step E's `BETS` and `BET_CLOSING_LINES` schemas.
- Step E's reconciliation asset -- G1 hooks into its settlement
  write.
- `dagster_project/assets/recommendations.py` -- weekly asset that
  needs to read bankroll and emit risk events.

## Out of scope for Step G (deferred)

- Sportsbook API integration. Guardrails operate on recommendations,
  not on actual placed bets. The user still places bets manually
  through their book.
- ML-based anomaly detection. G5 uses simple rules. A learned
  anomaly model is a Step J+ concern.
- Drawdown circuit-breaker (auto-pause after N% bankroll loss).
  Probably belongs in Step G but discuss with the user -- it's a
  policy decision, not a technical one.

---

## First actions in the fresh session

1. `git log -5 --oneline`, `git status`, confirm state.
2. Read `src/betting/recommend.py` to see what changed in Steps E and
   F. The guardrail hooks attach to whatever shape it has now.
3. Read Step E's `BETS` and `BET_CLOSING_LINES` schemas in
   `dbt_project/models/3_marts/`.
4. **Ask the user about a drawdown circuit-breaker**: "Do you want
   the system to auto-pause recommendations after a configurable %
   bankroll loss, or should that always be a human decision?" This
   shapes whether G6 needs a `paused` flag and how `weekly_bet_recommendations`
   reads it.
5. Ask the user for their initial threshold values (max weekly stake
   %, single bet %, staleness window). Recommend defaults of 10% / 3%
   / 4h and let them push back.
6. Enter plan mode and draft G1-G7.

## Verification

- Run `weekly_bet_recommendations` in a synthetic week where one
  recommendation has a stale line, one is anomalously large, and the
  Kelly-sized sum exceeds the cap. All three guardrails fire and
  produce `risk_events` rows. The final stake column reflects the
  scaling.
- Bankroll ledger round-trip: synthetic settlement after a week of
  bets reduces (or grows) `bankroll_units` by the expected amount;
  next week's recommendations Kelly-size against the new bankroll.
- `python src/betting/risk.py --since 7d` produces a clean summary.
- Full test suite green.
- Manual smoke test: try to recommend a bet with a 4.5h-old Vegas
  line. System refuses. Repeat with a fresh line. System accepts.
