# Step handoffs

Self-contained briefings for each planned step in the nfl_batchgineering
roadmap. Each file is designed to bootstrap a fresh Claude Code session
with no prior context.

## How to use

Open a fresh Claude session and paste:

> Read `C:\Users\jasse\_GitHub_repositories\nfl_batchgineering\.claude\plans\handoff-step-<X>-<name>.md`
> and treat it as your full briefing for Step <X>. Follow the "First
> actions" section -- start with the go/no-go / clarifying-questions
> conversation before any coding.

Each handoff includes context, what's already done, scope (broken into
sub-steps), project conventions, critical files to read first,
out-of-scope items, first actions, and verification criteria.

## Roadmap order (recommended)

1. **Step E -- CLV + bet ledger** ([handoff-step-e-clv-bet-ledger.md](handoff-step-e-clv-bet-ledger.md))
   Data foundation: record every bet, capture closing lines, reconcile
   results, build `mart_bet_performance`. Without this, every later
   improvement is unmeasurable.

2. **Step F -- Model iteration framework + first challenger** ([handoff-step-f-model-iteration.md](handoff-step-f-model-iteration.md))
   Champion-vs-challenger evaluation harness, promotion gate, live
   shadowing. First challenger: calibrated stacked ensemble (XGBoost +
   Elo + Bayesian state-space + Platt calibration).

3. **Step G -- Pre-bet guardrails** ([handoff-step-g-prebet-guardrails.md](handoff-step-g-prebet-guardrails.md))
   Bankroll persistence, weekly stake caps, line-staleness checks,
   anomaly detection, `risk_events` table. Safety rails between model
   output and real money.

4. **Step H -- Weekly automation** ([handoff-step-h-weekly-automation.md](handoff-step-h-weekly-automation.md))
   Dagster schedules for Thursday recommendations, Sunday closing-line
   snapshots, Tuesday reconciliation. Notifications (Slack), STATUS
   dashboard, failure alerting.

5. **Step I -- Totals model** ([handoff-step-i-totals-model.md](handoff-step-i-totals-model.md))
   Over / under betting alongside spreads. Roughly doubles the bet
   universe. Different feature emphasis (weather, pace) than spreads.

## Why this order

E unblocks measurement, which unblocks everything else. F gives the
muscle to swap models safely *before* the current model decays in
production. G is the safety layer that lets real money flow through
the system. H closes the manual-operation loop. I doubles the market
coverage. Each step's verification depends on the steps before it
shipping cleanly.

## Order can flex

- **G before F** if the user wants safety rails before any model
  changes go live. Recommended if model iteration feels risky.
- **H before G** is fine if the user is comfortable with manual
  guardrail enforcement for a few more weeks. Not recommended.
- **I before F** is fine -- totals is a clean expansion that
  doesn't depend on champion-vs-challenger machinery. Recommended
  if the user wants to grow the bet universe before iterating on
  model quality.

## Out of scope across all steps

- Player props, alt lines, live in-game betting, teasers, parlays.
- Sportsbook API auto-betting. The user always manually places bets.
- Custom web UI / dashboard. Markdown + Slack is enough.
- Deep learning architectures (NFL sample size doesn't support them).
