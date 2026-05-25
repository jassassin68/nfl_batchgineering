"""Walk-forward backtesting harness for NFL spread models.

Loads game features from ``mart_game_prediction_features``, runs walk-forward
cross-validation across seasons, and emits per-(model, test_season) metrics:
ATS accuracy, Brier score, RMSE vs actual, RMSE vs Vegas, and Kelly-sized ROI.

Two reports are written to ``reports/``:
  - ``backtest_<UTC-timestamp>.md`` -- human-readable summary table.
  - ``backtest_<UTC-timestamp>.csv`` -- long-form per-fold rows.

Purpose: establish a measured baseline. This is the v1 system, not the final
betting system. Future model iterations should be evaluated by re-running this
same harness so results are directly comparable.

CLI:
    python src/ml/backtest.py --start-season 2014 --end-season 2025
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import polars as pl

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# Make ``src`` importable when running this file directly via the CLI.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ml.models.elo_model import EloModel
from src.ml.models.spread_predictor import SpreadPredictor
from src.ml.utils.feature_engineering import prepare_training_data
from src.ml.utils.validation import (
    calculate_brier_score,
    calculate_roi,
)

# Standard targets pulled from CLAUDE.md.
ATS_BREAKEVEN = 0.5238  # break-even ATS rate at -110 odds (110 / 210)
ATS_TARGET = 0.524  # CLAUDE.md threshold to "beat the juice"
DEFAULT_EDGE_THRESHOLD = 4.0  # Step D: tightened from 3.0 for live betting
DEFAULT_KELLY_MULT = 0.25
DEFAULT_N_SEASONS_TRAIN = 5
DEFAULT_AMERICAN_ODDS = -110


@dataclass
class FoldMetrics:
    """Per-fold (model, test_season) result row.

    All fields are JSON-serializable so the same dataclass round-trips through
    CSV and markdown without a separate schema.
    """

    model: str
    test_season: int
    n_games: int
    train_seasons: str  # "2014-2018" style
    ats_accuracy_all: float
    ats_accuracy_edge: float
    n_bets_edge: int
    brier_score: float
    rmse_vs_actual: float
    rmse_vs_vegas_actual: float
    roi: float
    wins: int
    losses: int
    pushes: int
    profit_units: float
    cleared_ats_target: bool


def load_training_data(
    start_season: int,
    end_season: int,
    table_fqn: Optional[str] = None,
) -> pl.DataFrame:
    """Load ``mart_game_prediction_features`` rows for the given season range.

    Args:
        start_season: Inclusive lower bound.
        end_season: Inclusive upper bound.
        table_fqn: Override the fully-qualified table name (used by tests).

    Returns:
        Polars DataFrame ordered by (season, week, game_id) with completed
        games only (non-null scores).
    """
    from src.pipeline.snowflake import marts_table, query_df

    fqn = table_fqn or marts_table("mart_game_prediction_features")
    sql = f"""
        SELECT *
        FROM {fqn}
        WHERE season BETWEEN {int(start_season)} AND {int(end_season)}
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY season, week, game_id
    """
    df = query_df(sql)
    if df.is_empty():
        raise RuntimeError(
            f"No rows in {fqn} between seasons {start_season} and {end_season}. "
            "Run training_data_loader.py + dbt build first."
        )
    # Snowflake returns season as VARCHAR/Int8 depending on the source. Force
    # int64 so walk-forward CV and is_in() comparisons against Python ints work.
    return df.with_columns(pl.col("season").cast(pl.Int64))


def _ats_accuracy(
    predictions: np.ndarray,
    actuals: np.ndarray,
    vegas_spreads: np.ndarray,
    edge_threshold: float = 0.0,
) -> tuple[float, int]:
    """Return (ATS accuracy, count of bets placed at the edge threshold).

    Returns (NaN, 0) if no bets meet the threshold -- callers should special-case.
    """
    edges = predictions - vegas_spreads
    if edge_threshold > 0.0:
        mask = np.abs(edges) >= edge_threshold
    else:
        mask = np.ones_like(edges, dtype=bool)

    n_bets = int(mask.sum())
    if n_bets == 0:
        return float("nan"), 0

    # Push games are not wins or losses -- treat them as 0.5 (half-credit) so the
    # ATS rate is bookmaker-comparable. With nflverse data integer spreads are
    # common in older seasons, so this matters.
    margin = actuals[mask] - vegas_spreads[mask]
    model_picks_home = edges[mask] > 0
    home_covers = margin > 0
    away_covers = margin < 0
    pushes = margin == 0

    correct = (model_picks_home & home_covers) | (~model_picks_home & away_covers)
    score = correct.astype(float) + pushes.astype(float) * 0.5
    return float(score.mean()), n_bets


def _summarize_fold(
    model_name: str,
    test_season: int,
    train_seasons: List[int],
    predictions: np.ndarray,
    actuals: np.ndarray,
    vegas_spreads: np.ndarray,
    edge_threshold: float,
    odds: float,
) -> FoldMetrics:
    """Compute one FoldMetrics row from aligned arrays.

    Aligned arrays are required (same length, same ordering) -- callers are
    responsible for that invariant.
    """
    assert len(predictions) == len(actuals) == len(vegas_spreads), (
        f"length mismatch: preds={len(predictions)} actuals={len(actuals)} "
        f"vegas={len(vegas_spreads)}"
    )

    ats_all, _ = _ats_accuracy(predictions, actuals, vegas_spreads, 0.0)
    ats_edge, n_bets_edge = _ats_accuracy(
        predictions, actuals, vegas_spreads, edge_threshold
    )

    # Brier score on win-prob predictions: logistic mapping with scale 5.5
    # matches base.BasePredictor.predict_proba and predict.py.
    pred_win_prob = 1.0 / (1.0 + np.exp(-predictions / 5.5))
    actual_home_win = (actuals > 0).astype(float)
    brier = float(calculate_brier_score(pred_win_prob, actual_home_win))

    rmse_actual = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
    rmse_vegas_actual = float(np.sqrt(np.mean((vegas_spreads - actuals) ** 2)))

    roi_result = calculate_roi(
        predictions, actuals, vegas_spreads, edge_threshold, odds
    )

    train_range = (
        f"{min(train_seasons)}-{max(train_seasons)}" if train_seasons else "n/a"
    )

    cleared = not np.isnan(ats_edge) and ats_edge >= ATS_TARGET

    return FoldMetrics(
        model=model_name,
        test_season=int(test_season),
        n_games=int(len(predictions)),
        train_seasons=train_range,
        ats_accuracy_all=float(ats_all),
        ats_accuracy_edge=float(ats_edge),
        n_bets_edge=int(n_bets_edge),
        brier_score=brier,
        rmse_vs_actual=rmse_actual,
        rmse_vs_vegas_actual=rmse_vegas_actual,
        roi=float(roi_result["roi"]),
        wins=int(roi_result["wins"]),
        losses=int(roi_result["losses"]),
        pushes=int(roi_result["pushes"]),
        profit_units=float(roi_result["profit"]),
        cleared_ats_target=bool(cleared),
    )


def _walk_forward_season_splits(
    seasons: List[int], n_seasons_train: int
) -> List[tuple[List[int], int]]:
    """Yield (train_seasons, test_season) pairs using an expanding window.

    Mirrors ``utils.validation.walk_forward_cv`` but returns season lists
    instead of slicing a DataFrame so callers can use this for both Polars and
    model-specific fits.
    """
    sorted_seasons = sorted(set(int(s) for s in seasons))
    if len(sorted_seasons) < n_seasons_train + 1:
        raise ValueError(
            f"Need at least {n_seasons_train + 1} seasons for walk-forward "
            f"CV, got {len(sorted_seasons)}: {sorted_seasons}"
        )

    splits = []
    for i in range(n_seasons_train, len(sorted_seasons)):
        train = sorted_seasons[:i]
        test = sorted_seasons[i]
        assert max(train) < test, (
            "walk-forward invariant violated: train must precede test "
            f"(train={train}, test={test})"
        )
        splits.append((train, test))
    return splits


def backtest_xgboost(
    df: pl.DataFrame,
    n_seasons_train: int,
    edge_threshold: float,
    odds: float,
) -> List[FoldMetrics]:
    """Walk-forward backtest the XGBoost SpreadPredictor.

    Trains a fresh SpreadPredictor on each fold's train seasons, predicts on
    the held-out test season, and records metrics. No information from the
    test season is ever exposed to the model during fit.
    """
    # Use the existing prep pipeline so we exercise the same code path as
    # production training. Returns prepared df, feature list, target column.
    prepared, feature_cols, target_col = prepare_training_data(
        df, target_type="spread", include_derived_features=True
    )

    seasons = sorted(set(int(s) for s in prepared["season"].to_list()))
    splits = _walk_forward_season_splits(seasons, n_seasons_train)

    results: List[FoldMetrics] = []
    for train_seasons, test_season in splits:
        train_df = prepared.filter(pl.col("season").is_in(train_seasons))
        test_df = prepared.filter(pl.col("season") == test_season)

        if test_df.is_empty():
            print(
                f"[xgboost] skipping {test_season}: no test rows after prep",
                flush=True,
            )
            continue

        X_train = train_df.select(feature_cols).to_numpy()
        y_train = train_df[target_col].to_numpy()
        X_test = test_df.select(feature_cols).to_numpy()
        y_test = test_df[target_col].to_numpy()
        vegas_test = test_df["vegas_spread"].to_numpy().astype(float)

        model = SpreadPredictor()
        # Train without an explicit val split here -- early stopping needs eval
        # data, but holding out part of the train seasons within walk-forward
        # would just shift the boundary. Use the default n_estimators instead.
        model.train(
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_cols,
            early_stopping_rounds=0,
            verbose=False,
        )
        preds = model.predict(X_test)

        fold = _summarize_fold(
            model_name="xgboost",
            test_season=test_season,
            train_seasons=train_seasons,
            predictions=preds,
            actuals=y_test,
            vegas_spreads=vegas_test,
            edge_threshold=edge_threshold,
            odds=odds,
        )
        results.append(fold)
        print(
            f"[xgboost] {test_season}: ATS_all={fold.ats_accuracy_all:.3f} "
            f"ATS_edge={fold.ats_accuracy_edge:.3f} ({fold.n_bets_edge} bets) "
            f"Brier={fold.brier_score:.4f} ROI={fold.roi:+.3f}",
            flush=True,
        )

    return results


def backtest_elo(
    df: pl.DataFrame,
    n_seasons_train: int,
    edge_threshold: float,
    odds: float,
) -> List[FoldMetrics]:
    """Walk-forward backtest the Elo baseline.

    Fits Elo on train seasons (chronological updates), then snapshots ratings
    and predicts every test-season game using only the pre-test-season ratings.
    This keeps each test-season prediction independent of intra-season Elo
    updates, mirroring how a season-opening model would behave.
    """
    seasons = sorted(set(int(s) for s in df["season"].to_list()))
    splits = _walk_forward_season_splits(seasons, n_seasons_train)

    required_cols = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "vegas_spread",
    ]
    elo_df = df.select(required_cols).filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )

    results: List[FoldMetrics] = []
    for train_seasons, test_season in splits:
        train_df = elo_df.filter(pl.col("season").is_in(train_seasons))
        test_df = elo_df.filter(pl.col("season") == test_season)

        if test_df.is_empty():
            print(
                f"[elo] skipping {test_season}: no test rows", flush=True
            )
            continue

        model = EloModel()
        model.fit(train_df)

        # Apply between-season regression once for the upcoming test season,
        # matching what fit() would do at the next season boundary.
        model.regress_to_mean(test_season)

        preds = np.array(
            [
                model.predict_spread(row["home_team"], row["away_team"])
                for row in test_df.iter_rows(named=True)
            ]
        )
        actuals = (test_df["home_score"] - test_df["away_score"]).to_numpy()
        vegas = test_df["vegas_spread"].to_numpy().astype(float)

        fold = _summarize_fold(
            model_name="elo",
            test_season=test_season,
            train_seasons=train_seasons,
            predictions=preds,
            actuals=actuals,
            vegas_spreads=vegas,
            edge_threshold=edge_threshold,
            odds=odds,
        )
        results.append(fold)
        print(
            f"[elo] {test_season}: ATS_all={fold.ats_accuracy_all:.3f} "
            f"ATS_edge={fold.ats_accuracy_edge:.3f} ({fold.n_bets_edge} bets) "
            f"Brier={fold.brier_score:.4f} ROI={fold.roi:+.3f}",
            flush=True,
        )

    return results


# Registry of available backtests. Keep small and additive -- v2/v3 candidate
# models should add an entry here so the same CLI surface and report shape
# applies. Bayesian/Neural/Ensemble are intentionally not in the default set
# because they require per-fold MCMC / SGD that would dominate runtime; they
# can be added later behind an opt-in flag once their reliability is proven on
# this baseline framework.
BACKTEST_REGISTRY: Dict[str, Callable[..., List[FoldMetrics]]] = {
    "xgboost": backtest_xgboost,
    "elo": backtest_elo,
}


def run_full_backtest(
    df: pl.DataFrame,
    models: Optional[List[str]] = None,
    n_seasons_train: int = DEFAULT_N_SEASONS_TRAIN,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    odds: float = DEFAULT_AMERICAN_ODDS,
) -> Dict[str, List[FoldMetrics]]:
    """Run every requested backtest and return a dict keyed by model name.

    Args:
        df: ``mart_game_prediction_features`` rows.
        models: Subset of ``BACKTEST_REGISTRY`` keys; default = all.
        n_seasons_train: Walk-forward train window (expanding).
        edge_threshold: Minimum |pred - vegas| points to count a bet.
        odds: American odds for ROI sim (default -110).
    """
    chosen = models or list(BACKTEST_REGISTRY.keys())
    unknown = set(chosen) - set(BACKTEST_REGISTRY)
    if unknown:
        raise ValueError(
            f"Unknown models {sorted(unknown)}; "
            f"available: {sorted(BACKTEST_REGISTRY)}"
        )

    results: Dict[str, List[FoldMetrics]] = {}
    for name in chosen:
        print(f"\n=== Backtesting {name} ===", flush=True)
        fn = BACKTEST_REGISTRY[name]
        results[name] = fn(
            df,
            n_seasons_train=n_seasons_train,
            edge_threshold=edge_threshold,
            odds=odds,
        )
    return results


def _aggregate(fold_metrics: List[FoldMetrics]) -> Dict[str, float]:
    """Aggregate per-fold metrics into a single summary row per model."""
    if not fold_metrics:
        return {}

    total_games = sum(f.n_games for f in fold_metrics)
    total_bets = sum(f.n_bets_edge for f in fold_metrics)
    total_wins = sum(f.wins for f in fold_metrics)
    total_losses = sum(f.losses for f in fold_metrics)
    total_pushes = sum(f.pushes for f in fold_metrics)
    total_profit = sum(f.profit_units for f in fold_metrics)

    # Game-weighted means for accuracy/calibration so seasons with more games
    # carry proportionally more weight.
    weights = np.array([f.n_games for f in fold_metrics], dtype=float)

    def wmean(values: List[float]) -> float:
        arr = np.array(values, dtype=float)
        m = ~np.isnan(arr)
        if not m.any():
            return float("nan")
        return float(np.average(arr[m], weights=weights[m]))

    return {
        "n_folds": len(fold_metrics),
        "total_games": total_games,
        "total_bets_edge": total_bets,
        "ats_accuracy_all": wmean([f.ats_accuracy_all for f in fold_metrics]),
        "ats_accuracy_edge": wmean([f.ats_accuracy_edge for f in fold_metrics]),
        "brier_score": wmean([f.brier_score for f in fold_metrics]),
        "rmse_vs_actual": wmean([f.rmse_vs_actual for f in fold_metrics]),
        "rmse_vs_vegas_actual": wmean(
            [f.rmse_vs_vegas_actual for f in fold_metrics]
        ),
        "roi": total_profit / total_bets if total_bets > 0 else 0.0,
        "wins": total_wins,
        "losses": total_losses,
        "pushes": total_pushes,
        "folds_clearing_ats_target": sum(
            1 for f in fold_metrics if f.cleared_ats_target
        ),
    }


def write_report(
    results: Dict[str, List[FoldMetrics]],
    out_dir: Path,
    config: Dict,
) -> tuple[Path, Path]:
    """Write a markdown summary and a CSV of every fold row.

    Both files are timestamped (UTC, second-precision) so multiple runs in a
    session don't overwrite each other.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"backtest_{stamp}.csv"
    md_path = out_dir / f"backtest_{stamp}.md"

    # CSV: one row per (model, test_season).
    all_rows: List[FoldMetrics] = []
    for folds in results.values():
        all_rows.extend(folds)

    fieldnames = list(asdict(all_rows[0]).keys()) if all_rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(asdict(row))

    # Markdown: per-model summary table + per-fold detail tables.
    lines: List[str] = []
    lines.append(f"# Backtest report -- {stamp}")
    lines.append("")
    lines.append("Run configuration:")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(config, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append(
        f"ATS target: {ATS_TARGET:.4f} (clear -110 juice). "
        f"Break-even: {ATS_BREAKEVEN:.4f}."
    )
    lines.append("")

    lines.append("## Per-model summary (game-weighted)")
    lines.append("")
    lines.append(
        "| model | folds | games | bets@edge | ATS_all | ATS_edge | Brier "
        "| RMSE_vs_actual | RMSE_vs_vegas | ROI | folds>=ATS_target |"
    )
    lines.append(
        "|-------|-------|-------|-----------|---------|----------|-------|----------------|---------------|-----|--------------------|"
    )
    for model, folds in results.items():
        agg = _aggregate(folds)
        if not agg:
            lines.append(f"| {model} | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | 0 |")
            continue
        lines.append(
            f"| {model} | {agg['n_folds']} | {agg['total_games']} | "
            f"{agg['total_bets_edge']} | {agg['ats_accuracy_all']:.4f} | "
            f"{agg['ats_accuracy_edge']:.4f} | {agg['brier_score']:.4f} | "
            f"{agg['rmse_vs_actual']:.3f} | {agg['rmse_vs_vegas_actual']:.3f} | "
            f"{agg['roi']:+.4f} | {agg['folds_clearing_ats_target']} |"
        )
    lines.append("")

    lines.append("## Per-fold detail")
    lines.append("")
    for model, folds in results.items():
        lines.append(f"### {model}")
        lines.append("")
        if not folds:
            lines.append("_No folds completed._")
            lines.append("")
            continue
        lines.append(
            "| test_season | train | games | bets@edge | ATS_all | ATS_edge "
            "| Brier | RMSE | ROI | cleared |"
        )
        lines.append(
            "|------------:|-------|------:|----------:|--------:|---------:|"
            "------:|-----:|----:|:-------:|"
        )
        for f in folds:
            ats_e = (
                "n/a" if np.isnan(f.ats_accuracy_edge) else f"{f.ats_accuracy_edge:.4f}"
            )
            lines.append(
                f"| {f.test_season} | {f.train_seasons} | {f.n_games} | "
                f"{f.n_bets_edge} | {f.ats_accuracy_all:.4f} | {ats_e} | "
                f"{f.brier_score:.4f} | {f.rmse_vs_actual:.2f} | "
                f"{f.roi:+.4f} | {'yes' if f.cleared_ats_target else 'no'} |"
            )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, csv_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest for NFL spread models."
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=2014,
        help="Inclusive lower bound of seasons to load (default: 2014).",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=2025,
        help="Inclusive upper bound of seasons to load (default: 2025).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(BACKTEST_REGISTRY),
        default=None,
        help="Subset of models to backtest (default: all).",
    )
    parser.add_argument(
        "--n-seasons-train",
        type=int,
        default=DEFAULT_N_SEASONS_TRAIN,
        help=(
            "Number of seasons in the initial training window "
            f"(default: {DEFAULT_N_SEASONS_TRAIN})."
        ),
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=DEFAULT_EDGE_THRESHOLD,
        help=(
            "Minimum |pred - vegas| points required to count a bet "
            f"(default: {DEFAULT_EDGE_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--odds",
        type=float,
        default=DEFAULT_AMERICAN_ODDS,
        help=f"American odds for ROI sim (default: {DEFAULT_AMERICAN_ODDS}).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=_REPO_ROOT / "reports",
        help="Directory to write backtest_*.md and backtest_*.csv into.",
    )
    args = parser.parse_args(argv)

    print(
        f"Loading game features for seasons {args.start_season}-{args.end_season}",
        flush=True,
    )
    df = load_training_data(args.start_season, args.end_season)
    print(f"Loaded {len(df)} games across {df['season'].n_unique()} seasons", flush=True)

    results = run_full_backtest(
        df,
        models=args.models,
        n_seasons_train=args.n_seasons_train,
        edge_threshold=args.edge_threshold,
        odds=args.odds,
    )

    config = {
        "start_season": args.start_season,
        "end_season": args.end_season,
        "models": args.models or list(BACKTEST_REGISTRY.keys()),
        "n_seasons_train": args.n_seasons_train,
        "edge_threshold": args.edge_threshold,
        "odds": args.odds,
        "ats_target": ATS_TARGET,
    }
    md_path, csv_path = write_report(results, args.report_dir, config)
    print(f"\nWrote {md_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
