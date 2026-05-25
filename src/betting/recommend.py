"""Build sized bet recommendations from model predictions.

One entry point (``build_recommendations``) used by both:
  - src/ml/predict.py (refactored to use this package)
  - dagster_project/assets/recommendations.py (weekly_bet_recommendations)
  - this module's own CLI (``python src/betting/recommend.py``)

Output is a Polars frame so it round-trips cleanly through CSV and Snowflake.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.betting.edge import calculate_edge, should_bet
from src.betting.kelly import kelly_fraction

DEFAULT_EDGE_THRESHOLD = 4.0  # tightened from 3.0 for Step D
DEFAULT_KELLY_MULT = 0.25     # CLAUDE.md floor
DEFAULT_BANKROLL = 1000.0
DEFAULT_ODDS = -110.0

# Logistic scale used everywhere for spread -> win-prob mapping.
# Must match src/ml/predict.py and src/ml/backtest.py.
_LOGISTIC_SCALE = 5.5

SIDE_HOME = "home"
SIDE_AWAY = "away"
SIDE_PASS = "pass"


@dataclass
class BetRecommendation:
    """One sized recommendation per game.

    win_prob is the model's probability for the *side being bet* (not
    always the home side), so it can feed Kelly directly.
    """

    game_id: str
    side: str  # 'home' | 'away' | 'pass'
    edge_points: float
    home_win_prob: float
    win_prob: float
    kelly_fraction: float
    stake_units: float
    vegas_spread: float
    predicted_spread: float


def _home_win_prob_from_spread(predicted_spread: float) -> float:
    """Logistic mapping matching predict.py:471 and backtest.py:186."""
    return 1.0 / (1.0 + math.exp(-float(predicted_spread) / _LOGISTIC_SCALE))


def recommend_one(
    game_id: str,
    predicted_spread: float,
    vegas_spread: float,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    kelly_mult: float = DEFAULT_KELLY_MULT,
    bankroll: float = DEFAULT_BANKROLL,
    odds: float = DEFAULT_ODDS,
) -> BetRecommendation:
    """Compute a single BetRecommendation."""
    edge = calculate_edge(predicted_spread, vegas_spread)
    home_wp = _home_win_prob_from_spread(predicted_spread)

    if not should_bet(edge, edge_threshold):
        return BetRecommendation(
            game_id=str(game_id),
            side=SIDE_PASS,
            edge_points=round(edge, 3),
            home_win_prob=round(home_wp, 4),
            win_prob=round(home_wp, 4),
            kelly_fraction=0.0,
            stake_units=0.0,
            vegas_spread=float(vegas_spread),
            predicted_spread=float(predicted_spread),
        )

    side = SIDE_HOME if edge > 0 else SIDE_AWAY
    win_prob = home_wp if side == SIDE_HOME else (1.0 - home_wp)
    # Defense-in-depth: logistic never returns 0 or 1 for finite spreads,
    # but clamp to keep kelly_fraction's domain check happy.
    win_prob = min(max(win_prob, 1e-6), 1.0 - 1e-6)
    kelly = kelly_fraction(win_prob, odds=odds, kelly_mult=kelly_mult)
    stake = kelly * float(bankroll)

    return BetRecommendation(
        game_id=str(game_id),
        side=side,
        edge_points=round(edge, 3),
        home_win_prob=round(home_wp, 4),
        win_prob=round(win_prob, 4),
        kelly_fraction=round(kelly, 6),
        stake_units=round(stake, 4),
        vegas_spread=float(vegas_spread),
        predicted_spread=float(predicted_spread),
    )


def build_recommendations(
    df: pl.DataFrame,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    kelly_mult: float = DEFAULT_KELLY_MULT,
    bankroll: float = DEFAULT_BANKROLL,
    odds: float = DEFAULT_ODDS,
    *,
    game_id_col: str = "game_id",
    predicted_col: str = "predicted_spread",
    vegas_col: str = "vegas_spread",
) -> pl.DataFrame:
    """Build sized recommendations from a frame with predictions + Vegas lines.

    Args:
        df: Polars DataFrame containing one row per upcoming game with at
            least ``game_id``, ``predicted_spread``, ``vegas_spread``.
        edge_threshold: Minimum |edge| in points required to place a bet.
        kelly_mult: Fractional Kelly multiplier (CLAUDE.md: 0.25-0.5).
        bankroll: Bankroll size in the user's preferred units (dollars,
            units, etc.) -- stake_units is in the same unit.
        odds: American odds.
        game_id_col / predicted_col / vegas_col: column name overrides.

    Returns:
        Polars DataFrame with one row per input row, in the same order, with
        columns: game_id, side, edge_points, home_win_prob, win_prob,
        kelly_fraction, stake_units, vegas_spread, predicted_spread.
    """
    required = {game_id_col, predicted_col, vegas_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"build_recommendations: input frame missing columns {sorted(missing)}"
        )

    recs: List[BetRecommendation] = []
    for row in df.iter_rows(named=True):
        pred = row[predicted_col]
        vegas = row[vegas_col]
        if pred is None or vegas is None:
            # Treat missing inputs as a pass rather than crashing -- weekly
            # runs occasionally include games where the line hasn't posted.
            recs.append(
                BetRecommendation(
                    game_id=str(row[game_id_col]),
                    side=SIDE_PASS,
                    edge_points=float("nan"),
                    home_win_prob=float("nan"),
                    win_prob=float("nan"),
                    kelly_fraction=0.0,
                    stake_units=0.0,
                    vegas_spread=float(vegas) if vegas is not None else float("nan"),
                    predicted_spread=float(pred) if pred is not None else float("nan"),
                )
            )
            continue
        recs.append(
            recommend_one(
                game_id=row[game_id_col],
                predicted_spread=pred,
                vegas_spread=vegas,
                edge_threshold=edge_threshold,
                kelly_mult=kelly_mult,
                bankroll=bankroll,
                odds=odds,
            )
        )

    return pl.DataFrame([asdict(r) for r in recs])


# ---------------------------------------------------------------------------
# Report writers (shared by CLI and Dagster asset)
# ---------------------------------------------------------------------------


def write_recommendations_report(
    recs: pl.DataFrame,
    out_dir: Path,
    season: int,
    week: int,
    config: Dict,
) -> tuple[Path, Path]:
    """Write a markdown + CSV pair under ``out_dir``. Returns (md, csv) paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"recs_{season}_wk{week:02d}_{stamp}"
    csv_path = out_dir / f"{base}.csv"
    md_path = out_dir / f"{base}.md"

    recs.write_csv(str(csv_path))

    bets = recs.filter(pl.col("side") != SIDE_PASS)
    n_total = recs.height
    n_bets = bets.height
    total_stake = float(bets["stake_units"].sum()) if n_bets else 0.0
    max_stake = float(bets["stake_units"].max()) if n_bets else 0.0
    avg_edge = (
        float(bets["edge_points"].abs().mean()) if n_bets else 0.0
    )

    lines: List[str] = []
    lines.append(f"# Weekly bet recommendations -- season {season}, week {week}")
    lines.append("")
    lines.append(f"Generated (UTC): {stamp}")
    lines.append("")
    lines.append("## Run configuration")
    lines.append("")
    for key in sorted(config):
        lines.append(f"- **{key}**: {config[key]}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Games considered: {n_total}")
    lines.append(f"- Bets recommended: {n_bets}")
    lines.append(f"- Total stake: {total_stake:.2f}")
    lines.append(f"- Max single stake: {max_stake:.2f}")
    lines.append(f"- Avg |edge| on bets: {avg_edge:.2f} pts")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append(
        "| game_id | side | edge_pts | predicted | vegas | win_prob | "
        "kelly_frac | stake_units |"
    )
    lines.append(
        "|---------|:----:|---------:|----------:|------:|---------:|"
        "-----------:|------------:|"
    )
    for r in recs.iter_rows(named=True):
        lines.append(
            f"| {r['game_id']} | {r['side']} | "
            f"{r['edge_points']:+.2f} | {r['predicted_spread']:+.2f} | "
            f"{r['vegas_spread']:+.2f} | {r['win_prob']:.3f} | "
            f"{r['kelly_fraction']:.5f} | {r['stake_units']:.4f} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_predictions_frame(
    season: int, week: int, historical: bool = False
) -> pl.DataFrame:
    """Load predictions for a season/week.

    Re-runs the same prediction pipeline used by src/ml/predict.py rather
    than reading a cached ML.PREDICTIONS row -- avoids stale-row hazards
    if predict has not been rerun this week.

    Args:
        season: NFL season.
        week: Week number.
        historical: If True, loads completed games from
            ``mart_game_prediction_features`` (useful for dry-runs when no
            upcoming week is available). If False (default), loads from
            ``mart_upcoming_game_predictions``.
    """
    # Import lazily and avoid pulling dagster_project/__init__.py (which
    # loads the entire asset graph + dbt manifest) just to find MODEL_DIR.
    from src.ml.predict import (
        generate_predictions,
        load_ensemble_model,
        load_historical_games,
        load_upcoming_games,
    )

    model_dir = _REPO_ROOT / "src" / "ml" / "models" / "ensemble"

    if historical:
        games_df = load_historical_games(week, season)
        source = "historical"
    else:
        games_df = load_upcoming_games(week, season)
        source = "upcoming"

    if games_df.is_empty():
        raise RuntimeError(
            f"No {source} games found for season {season} week {week}. "
            f"{'Try a different week.' if historical else 'Pass --historical to dry-run against a completed week.'}"
        )
    models = load_ensemble_model(str(model_dir))
    if not models:
        raise RuntimeError(f"No models found in {model_dir}")
    results = generate_predictions(games_df, models)
    if results.is_empty():
        raise RuntimeError("Prediction generation returned empty results.")
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build sized bet recommendations for an NFL week."
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=DEFAULT_EDGE_THRESHOLD,
        help=f"Minimum |edge| pts to bet (default: {DEFAULT_EDGE_THRESHOLD}).",
    )
    parser.add_argument(
        "--kelly-mult",
        type=float,
        default=DEFAULT_KELLY_MULT,
        help=f"Fractional Kelly multiplier (default: {DEFAULT_KELLY_MULT}).",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=DEFAULT_BANKROLL,
        help=f"Bankroll size in user units (default: {DEFAULT_BANKROLL}).",
    )
    parser.add_argument(
        "--odds",
        type=float,
        default=DEFAULT_ODDS,
        help=f"American odds for sizing (default: {DEFAULT_ODDS}).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=_REPO_ROOT / "reports" / "recommendations",
        help="Directory to write recs_*.md and recs_*.csv.",
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help=(
            "Dry-run mode: use completed games from "
            "mart_game_prediction_features instead of mart_upcoming_game_predictions. "
            "Useful when no upcoming week is available (offseason)."
        ),
    )
    args = parser.parse_args(argv)

    mode = "historical" if args.historical else "upcoming"
    print(
        f"Loading {mode} predictions for season {args.season} week {args.week}",
        flush=True,
    )
    predictions = _load_predictions_frame(
        args.season, args.week, historical=args.historical
    )
    print(f"Loaded {predictions.height} predicted games", flush=True)

    recs = build_recommendations(
        predictions,
        edge_threshold=args.edge_threshold,
        kelly_mult=args.kelly_mult,
        bankroll=args.bankroll,
        odds=args.odds,
    )

    config = {
        "season": args.season,
        "week": args.week,
        "edge_threshold": args.edge_threshold,
        "kelly_mult": args.kelly_mult,
        "bankroll": args.bankroll,
        "odds": args.odds,
    }
    md_path, csv_path = write_recommendations_report(
        recs, args.report_dir, args.season, args.week, config
    )

    n_bets = int(recs.filter(pl.col("side") != SIDE_PASS).height)
    print(f"Wrote {md_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    print(f"Bets recommended: {n_bets} / {recs.height}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
