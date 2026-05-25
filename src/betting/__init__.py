"""Betting decision layer: turn model spread predictions into sized bets.

Single source of truth for edge calculation, Kelly sizing, and the
home/away/pass recommendation logic consumed by both src/ml/predict.py
and the weekly_bet_recommendations Dagster asset.
"""

from src.betting.edge import calculate_edge, should_bet
from src.betting.kelly import american_to_decimal, kelly_fraction
from src.betting.recommend import (
    BetRecommendation,
    build_recommendations,
    DEFAULT_BANKROLL,
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_KELLY_MULT,
    DEFAULT_ODDS,
)

__all__ = [
    "BetRecommendation",
    "american_to_decimal",
    "build_recommendations",
    "calculate_edge",
    "kelly_fraction",
    "should_bet",
    "DEFAULT_BANKROLL",
    "DEFAULT_EDGE_THRESHOLD",
    "DEFAULT_KELLY_MULT",
    "DEFAULT_ODDS",
]
