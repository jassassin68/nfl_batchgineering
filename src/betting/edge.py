"""Edge calculation and bet-or-pass gating.

Sign convention (matches src/ml/predict.py and src/ml/backtest.py):
  edge = model_pred - vegas_line
  positive edge -> model favors home more than Vegas (bet home)
  negative edge -> model favors away more than Vegas (bet away)
"""


def calculate_edge(model_pred: float, vegas_line: float) -> float:
    """Return the model's spread disagreement with Vegas in points.

    Args:
        model_pred: Model-predicted spread (home margin).
        vegas_line: Vegas spread (positive = home favored).

    Returns:
        edge in points. Positive favors home; negative favors away.
    """
    return float(model_pred) - float(vegas_line)


def should_bet(edge: float, threshold: float = 4.0) -> bool:
    """Return True iff |edge| meets the minimum threshold.

    The default 4.0 is the upper end of CLAUDE.md's 3-4pt range and was
    selected for Step D after a borderline-positive baseline (ATS 0.5269).
    """
    return abs(float(edge)) >= float(threshold)
