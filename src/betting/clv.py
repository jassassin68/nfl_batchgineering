"""Closing-line value (CLV) and bet settlement.

Single source of truth for resolving a recorded bet into an outcome,
profit, and CLV. Imported by the reconciliation Dagster asset
(dagster_project/assets/reconciliation.py) and unit-tested in
tests/test_clv.py -- there is one settlement definition, not one per
caller.

Spread sign convention (matches src/betting/edge.py, src/ml/predict.py
and int_game_vegas_lines.sql): ``vegas_spread`` is positive when the
HOME team is favored. ``margin`` is ``home_score - away_score``.

CLV convention (confirmed in the Step E plan): positive CLV means the
line moved in the bettor's favor after the bet was placed.

  clv_points = (vegas_spread_close - vegas_spread_at_rec) * side_multiplier
      side_multiplier = +1 for a HOME bet, -1 for an AWAY bet

A HOME bettor profits from the home team becoming *more* favored
(vegas_spread rises); an AWAY bettor profits from the home team becoming
*less* favored (vegas_spread falls).
"""

from src.betting.kelly import american_to_decimal

SIDE_HOME = "home"
SIDE_AWAY = "away"

OUTCOME_WIN = "win"
OUTCOME_LOSS = "loss"
OUTCOME_PUSH = "push"


def _side_multiplier(side: str) -> int:
    """Return +1 for a HOME bet, -1 for an AWAY bet.

    Raises:
        ValueError: if ``side`` is not 'home' or 'away'.
    """
    s = str(side).lower()
    if s == SIDE_HOME:
        return 1
    if s == SIDE_AWAY:
        return -1
    raise ValueError(f"side must be 'home' or 'away'; got {side!r}")


def clv_points(
    vegas_spread_at_rec: float,
    vegas_spread_close: float,
    side: str,
) -> float:
    """Closing-line value in points for a single bet.

    Positive = the line moved in the bettor's favor after they bet.

    Args:
        vegas_spread_at_rec: Spread (home-favored positive) when the bet
            was recommended/placed.
        vegas_spread_close: Spread at kickoff (the closing line).
        side: 'home' or 'away'.

    Returns:
        CLV in points. Sign is relative to the side bet.
    """
    delta = float(vegas_spread_close) - float(vegas_spread_at_rec)
    return delta * _side_multiplier(side)


def settle_outcome(
    margin: float,
    vegas_spread_at_rec: float,
    side: str,
) -> str:
    """Resolve a bet against the spread it was placed at.

    The bet is graded against ``vegas_spread_at_rec`` (the line the bettor
    actually took), not the closing line. A HOME bet wins when the home
    team beats the spread; an AWAY bet wins when it does not. Landing
    exactly on the number is a push.

    Args:
        margin: Final game margin, ``home_score - away_score``.
        vegas_spread_at_rec: Spread (home-favored positive) the bet took.
        side: 'home' or 'away'.

    Returns:
        'win', 'loss', or 'push'.
    """
    # home_cover > 0 means the home team beat the spread.
    home_cover = float(margin) - float(vegas_spread_at_rec)
    if home_cover == 0:
        return OUTCOME_PUSH
    home_covered = home_cover > 0
    bet_home = _side_multiplier(side) == 1
    return OUTCOME_WIN if home_covered == bet_home else OUTCOME_LOSS


def profit_units(
    outcome: str,
    stake_units: float,
    odds: float = -110.0,
) -> float:
    """Net profit/loss for a settled bet, in the same units as ``stake_units``.

    Win pays ``stake * (decimal_odds - 1)``; loss returns ``-stake``; push
    returns 0. Uses the same American->decimal conversion as bet sizing so
    settlement and Kelly agree on the payout per unit.

    Args:
        outcome: 'win', 'loss', or 'push'.
        stake_units: Amount staked (same units as bankroll).
        odds: American odds the bet was taken at (default -110).

    Returns:
        Net profit (positive) or loss (negative) in stake units.

    Raises:
        ValueError: if ``outcome`` is not a recognized value.
    """
    stake = float(stake_units)
    if outcome == OUTCOME_WIN:
        return stake * (american_to_decimal(odds) - 1.0)
    if outcome == OUTCOME_LOSS:
        return -stake
    if outcome == OUTCOME_PUSH:
        return 0.0
    raise ValueError(
        f"outcome must be 'win', 'loss', or 'push'; got {outcome!r}"
    )


def settle_bet(
    margin: float,
    vegas_spread_at_rec: float,
    vegas_spread_close: float,
    side: str,
    stake_units: float,
    odds: float = -110.0,
) -> dict:
    """Fully resolve one bet: outcome, profit, and CLV.

    Convenience wrapper composing :func:`settle_outcome`,
    :func:`profit_units`, and :func:`clv_points` so the reconciliation
    asset has a single call per bet.

    Returns:
        dict with keys ``outcome``, ``profit_units``, ``clv_points``.
        ``clv_points`` is ``None`` when ``vegas_spread_close`` is ``None``
        (closing line not yet captured).
    """
    outcome = settle_outcome(margin, vegas_spread_at_rec, side)
    profit = profit_units(outcome, stake_units, odds=odds)
    clv = (
        None
        if vegas_spread_close is None
        else clv_points(vegas_spread_at_rec, vegas_spread_close, side)
    )
    return {
        "outcome": outcome,
        "profit_units": profit,
        "clv_points": clv,
    }
