-- One-time backfill for inverted `bet_recommendation` values in
-- PRODUCTION_ANALYTICS.ML.PREDICTIONS.
--
-- Context: predict.py emitted the two BET HOME/BET AWAY string literals swapped
-- relative to the model's actual value pick from the time the table was first
-- populated until the fix landed (see audit_findings_handoff.md Defect A).
-- The `edge`, `predicted_spread`, `home_win_prob`, and `confidence` columns are
-- already correct — only the recommendation label needs to flip. 'NO BET' rows
-- are untouched.
--
-- Run ONCE, after deploying the Defect A code fix in src/ml/predict.py.
-- Re-running this script after the code fix would re-invert correct rows, so
-- gate it carefully (e.g. via a feature-flag table or a one-shot manual run).

BEGIN;

-- 1. Pre-flight: row counts by current label (capture before/after for audit).
SELECT bet_recommendation, COUNT(*) AS n
FROM PRODUCTION_ANALYTICS.ML.PREDICTIONS
GROUP BY 1
ORDER BY 1;

-- 2. Swap labels in place.
UPDATE PRODUCTION_ANALYTICS.ML.PREDICTIONS
SET bet_recommendation = CASE
    WHEN bet_recommendation = 'BET HOME' THEN 'BET AWAY'
    WHEN bet_recommendation = 'BET AWAY' THEN 'BET HOME'
    ELSE bet_recommendation
END
WHERE bet_recommendation IN ('BET HOME', 'BET AWAY');

-- 3. Post-flight: row counts (BET HOME ↔ BET AWAY totals should swap; NO BET unchanged).
SELECT bet_recommendation, COUNT(*) AS n
FROM PRODUCTION_ANALYTICS.ML.PREDICTIONS
GROUP BY 1
ORDER BY 1;

-- 4. Spot check sign convention: every BET HOME should have edge > 0, every BET AWAY edge < 0.
SELECT bet_recommendation,
       MIN(edge) AS min_edge,
       MAX(edge) AS max_edge,
       COUNT_IF(
           (bet_recommendation = 'BET HOME' AND edge <= 0)
        OR (bet_recommendation = 'BET AWAY' AND edge >= 0)
       ) AS n_mismatched
FROM PRODUCTION_ANALYTICS.ML.PREDICTIONS
WHERE bet_recommendation IN ('BET HOME', 'BET AWAY')
GROUP BY 1
ORDER BY 1;
-- Expected: n_mismatched = 0 for both groups, min_edge >= 3.0 for BET HOME,
-- max_edge <= -3.0 for BET AWAY.

COMMIT;
