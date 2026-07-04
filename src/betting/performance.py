"""CLI: print the production bet-performance scorecard.

Queries the ``mart_bet_performance`` view and prints a markdown summary of
win rate, ROI, Brier, and closing-line value by rolling window and model
version.

ASCII-only output (Windows cp1252 console rule, same as the rest of the
betting layer). Reuses ``query_df`` from src/pipeline/snowflake -- no new
Snowflake helper.

Usage:
    python src/betting/performance.py
    python src/betting/performance.py --model-version step_d_xgb_v1
    python src/betting/performance.py --window season_to_date
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import polars as pl

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Render windows newest-to-widest regardless of the order rows come back in.
WINDOW_ORDER = ["last_4_weeks", "last_8_weeks", "season_to_date", "all_time"]


def _fmt(value, spec: str) -> str:
    """Format a possibly-NULL numeric cell as ASCII text."""
    if value is None:
        return "n/a"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def render_summary(df: pl.DataFrame) -> str:
    """Render the performance frame as an ASCII markdown report.

    Args:
        df: rows from mart_bet_performance (lower-cased columns).

    Returns:
        Markdown string. Safe to print on a cp1252 console.
    """
    lines: List[str] = ["# Bet performance scorecard", ""]

    if df.is_empty():
        lines.append("No settled bets yet -- the ledger is empty.")
        lines.append("")
        return "\n".join(lines)

    # Stable ordering: known windows first, then model_version.
    order_idx = {w: i for i, w in enumerate(WINDOW_ORDER)}
    df = df.with_columns(
        pl.col("rolling_window")
        .replace_strict(order_idx, default=len(WINDOW_ORDER))
        .alias("_order")
    ).sort(["_order", "model_version"])

    lines.append(
        "| window | model_version | n_bets | win_rate | roi | brier | "
        "avg_clv | clv_pos_rate |"
    )
    lines.append(
        "|--------|---------------|-------:|---------:|----:|------:|"
        "--------:|-------------:|"
    )
    for r in df.iter_rows(named=True):
        lines.append(
            f"| {r['rolling_window']} | {r['model_version']} | "
            f"{int(r['n_bets'])} | {_fmt(r.get('win_rate'), '.3f')} | "
            f"{_fmt(r.get('roi'), '+.3f')} | {_fmt(r.get('brier'), '.4f')} | "
            f"{_fmt(r.get('avg_clv'), '+.2f')} | "
            f"{_fmt(r.get('clv_positive_rate'), '.3f')} |"
        )
    lines.append("")
    return "\n".join(lines)


def load_performance(
    model_version: Optional[str] = None,
    window: Optional[str] = None,
) -> pl.DataFrame:
    """Read mart_bet_performance, optionally filtered."""
    from src.pipeline.snowflake import marts_table, query_df

    table = marts_table("mart_bet_performance")
    where: List[str] = []
    if model_version:
        where.append(f"model_version = '{model_version}'")
    if window:
        where.append(f"rolling_window = '{window}'")
    clause = f" WHERE {' AND '.join(where)}" if where else ""
    return query_df(f"SELECT * FROM {table}{clause}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the production bet-performance scorecard."
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Filter to a single model_version.",
    )
    parser.add_argument(
        "--window",
        default=None,
        choices=WINDOW_ORDER,
        help="Filter to a single rolling window.",
    )
    args = parser.parse_args(argv)

    df = load_performance(model_version=args.model_version, window=args.window)
    print(render_summary(df), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
