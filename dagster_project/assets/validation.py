"""Model validation asset: walk-forward backtest report.

Materializes a fresh ``backtest_*.md`` / ``backtest_*.csv`` pair under
``reports/`` (gitignored) and surfaces top-line metrics (ATS, Brier,
folds-clearing-target) as asset metadata so the Dagster UI can be used as a
performance dashboard across runs.

Note: NO ``from __future__ import annotations`` here -- Dagster's pythonic
Config introspection cannot resolve PEP 563 stringified annotations and
raises DagsterInvalidPythonicConfigDefinitionError.
"""

import sys
from pathlib import Path

from dagster import AssetKey, Config, MaterializeResult, MetadataValue, asset

from dagster_project.constants import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.backtest import (
    ATS_TARGET,
    BACKTEST_REGISTRY,
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_KELLY_MULT,
    DEFAULT_N_SEASONS_TRAIN,
    DEFAULT_AMERICAN_ODDS,
    _aggregate,
    load_training_data,
    run_full_backtest,
    write_report,
)


class ValidationConfig(Config):
    """Run config for the model_validation_report asset."""

    start_season: int = 2014
    end_season: int = 2025
    n_seasons_train: int = 5
    edge_threshold: float = 3.0
    odds: float = -110.0


@asset(
    group_name="validation",
    compute_kind="python",
    deps=[AssetKey("mart_game_prediction_features")],
)
def model_validation_report(
    context, config: ValidationConfig
) -> MaterializeResult:
    """Run walk-forward backtest and emit a report under reports/.

    Reuses ``src.ml.backtest`` so the Dagster surface and the CLI surface
    produce identical artifacts -- there is one backtest definition, two
    invocations.
    """
    context.log.info(
        f"Backtesting seasons {config.start_season}-{config.end_season} "
        f"(train window={config.n_seasons_train}, "
        f"edge={config.edge_threshold}, odds={config.odds})"
    )

    df = load_training_data(config.start_season, config.end_season)
    context.log.info(
        f"Loaded {len(df)} games across {df['season'].n_unique()} seasons"
    )

    results = run_full_backtest(
        df,
        n_seasons_train=config.n_seasons_train,
        edge_threshold=config.edge_threshold,
        odds=config.odds,
    )

    report_dir = PROJECT_ROOT / "reports"
    run_config = {
        "start_season": config.start_season,
        "end_season": config.end_season,
        "models": list(BACKTEST_REGISTRY.keys()),
        "n_seasons_train": config.n_seasons_train,
        "edge_threshold": config.edge_threshold,
        "odds": config.odds,
        "ats_target": ATS_TARGET,
        "kelly_mult": DEFAULT_KELLY_MULT,
    }
    md_path, csv_path = write_report(results, report_dir, run_config)
    context.log.info(f"Wrote {md_path}")
    context.log.info(f"Wrote {csv_path}")

    # Surface every model's top-line aggregate so the UI shows a
    # cross-model comparison at a glance.
    metadata: dict = {
        "report_markdown": MetadataValue.path(str(md_path)),
        "report_csv": MetadataValue.path(str(csv_path)),
        "seasons_range": MetadataValue.text(
            f"{config.start_season}-{config.end_season}"
        ),
        "ats_target": MetadataValue.float(ATS_TARGET),
    }

    total_folds_clearing = 0
    for model_name, folds in results.items():
        agg = _aggregate(folds)
        if not agg:
            continue
        metadata[f"{model_name}_ats_edge"] = MetadataValue.float(
            round(agg["ats_accuracy_edge"], 4)
        )
        metadata[f"{model_name}_ats_all"] = MetadataValue.float(
            round(agg["ats_accuracy_all"], 4)
        )
        metadata[f"{model_name}_brier"] = MetadataValue.float(
            round(agg["brier_score"], 4)
        )
        metadata[f"{model_name}_roi"] = MetadataValue.float(
            round(agg["roi"], 4)
        )
        metadata[f"{model_name}_folds"] = MetadataValue.int(agg["n_folds"])
        metadata[f"{model_name}_folds_clearing_target"] = MetadataValue.int(
            agg["folds_clearing_ats_target"]
        )
        total_folds_clearing += agg["folds_clearing_ats_target"]

    metadata["total_folds_clearing_ats_target"] = MetadataValue.int(
        total_folds_clearing
    )

    return MaterializeResult(metadata=metadata)
