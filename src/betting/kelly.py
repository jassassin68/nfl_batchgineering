"""Fractional Kelly bet sizing.

CLAUDE.md mandates 0.25-0.5x Kelly, never full Kelly. This module enforces
a multiplier and clamps the result to [0, 1] so a positive recommendation
can never exceed bankroll.
"""


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds.

    -110 -> 1.909..., +150 -> 2.5. Matches the conversion used in
    src/ml/utils/validation.py::calculate_roi so ROI sims and bet sizing
    agree on the payout per unit.
    """
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds < 0:
        return 1.0 + (100.0 / abs(odds))
    return 1.0 + (odds / 100.0)


def kelly_fraction(
    win_prob: float,
    odds: float = -110.0,
    kelly_mult: float = 0.25,
) -> float:
    """Compute fractional Kelly bet size as a fraction of bankroll.

    Kelly formula: f* = (b*p - q) / b
        where b = decimal_odds - 1, p = win_prob, q = 1 - p.

    The raw f* is multiplied by ``kelly_mult`` (default 0.25 = quarter
    Kelly per CLAUDE.md floor) and clamped to [0, 1]. A negative raw
    Kelly (negative-EV bet) returns 0 -- callers should already have
    gated on edge, but this is a defense in depth.

    Args:
        win_prob: Model's estimated win probability for the side bet.
                  Must be in (0, 1) exclusive.
        odds: American odds (default -110, the standard ATS price).
        kelly_mult: Fraction of full Kelly to use. Must be > 0.

    Returns:
        Fraction of bankroll to stake, clamped to [0, 1].

    Raises:
        ValueError: if win_prob is not in (0, 1) or kelly_mult <= 0.
    """
    win_prob = float(win_prob)
    if not (0.0 < win_prob < 1.0):
        raise ValueError(
            f"win_prob must be in (0, 1); got {win_prob}"
        )
    if kelly_mult <= 0.0:
        raise ValueError(
            f"kelly_mult must be > 0; got {kelly_mult}"
        )

    b = american_to_decimal(odds) - 1.0  # net profit per unit staked
    p = win_prob
    q = 1.0 - p
    raw = (b * p - q) / b
    scaled = raw * float(kelly_mult)
    return max(0.0, min(1.0, scaled))
