"""
NFL Game Spread Prediction Script
Generates predictions for upcoming games using trained ensemble model.

Usage:
    python src/ml/predict.py --week 5 --season 2025 --output predictions_week5.csv
    python src/ml/predict.py --week 5 --season 2025 --output predictions.csv --snowflake
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

import polars as pl
import numpy as np
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.models.elo_model import EloModel
from src.ml.models.spread_predictor import SpreadPredictor
from src.ml.utils.feature_engineering import select_spread_features

# Try to import optional models
try:
    from src.ml.models.bayesian import BayesianStateSpace
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

try:
    from src.ml.models.neural import NeuralNetPredictor
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False

try:
    from src.ml.models.ensemble import StackingEnsemble
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False


def get_snowflake_connection():
    """Create Snowflake connection using key pair authentication."""
    import snowflake.connector
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    # Load environment variables
    load_dotenv()

    # Load private key
    private_key_text = os.getenv('SNOWFLAKE_KEYPAIR_PRIVATE_KEY')
    passphrase = os.getenv('SNOWFLAKE_KEYPAIR_PASSPHRASE')

    # Replace literal \n with actual newlines
    private_key_text = private_key_text.replace('\\n', '\n')

    # Decode the private key
    private_key = serialization.load_pem_private_key(
        private_key_text.encode(),
        password=passphrase.encode() if passphrase else None,
        backend=default_backend()
    )

    pkb = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        private_key=pkb,
        database="PRODUCTION_ANALYTICS",
        schema="ANALYTICS",
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        role=os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
    )

    return conn


def load_upcoming_games(week: int, season: int) -> pl.DataFrame:
    """Load upcoming games from Snowflake mart_upcoming_game_predictions."""
    print(f"Loading upcoming games for Week {week}, Season {season}...")

    conn = get_snowflake_connection()

    query = f"""
        SELECT *
        FROM PRODUCTION_ANALYTICS.ANALYTICS.MART_UPCOMING_GAME_PREDICTIONS
        WHERE season = {season}
          AND week = {week}
        ORDER BY gameday, gametime
    """

    cursor = conn.cursor()
    cursor.execute(query)

    # Get column names
    columns = [desc[0].lower() for desc in cursor.description]

    # Fetch all rows
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    if not rows:
        print(f"No upcoming games found for Week {week}, Season {season}")
        return pl.DataFrame()

    # Create DataFrame
    df = pl.DataFrame({col: [row[i] for row in rows] for i, col in enumerate(columns)})

    print(f"Loaded {len(df)} upcoming games")
    return df


def load_historical_games(week: int, season: int) -> pl.DataFrame:
    """Load a completed historical week from mart_game_prediction_features.

    Used by TEST mode to replay a known week when no upcoming games exist
    (e.g. the offseason). Includes home_score/away_score so predictions can be
    scored against actual results.

    Args:
        week: NFL week number.
        season: NFL season year.

    Returns:
        Polars DataFrame of historical games with the full feature set.
    """
    from src.pipeline.snowflake import marts_table, query_df

    table = marts_table("mart_game_prediction_features")
    print(f"Loading historical games for Week {week}, Season {season} from {table}...")
    df = query_df(
        f"select * from {table} "
        f"where season = {season} and week = {week} "
        f"order by gameday, gametime"
    )
    print(f"Loaded {len(df)} historical games")
    return df


def compare_to_actuals(results: pl.DataFrame, games_df: pl.DataFrame) -> pl.DataFrame:
    """Score TEST-mode predictions against actual game results.

    Adds actual_spread, model_error and ats_hit columns. ats_hit is True when
    the model picked the correct side of the Vegas line -- the sign of
    (predicted - Vegas) matches the sign of (actual - Vegas). Games that landed
    exactly on the line, or that have no Vegas line, are treated as pushes
    (ats_hit = null).

    Args:
        results: prediction output from generate_predictions().
        games_df: the historical games, carrying home_score/away_score.

    Returns:
        results with actual_spread, model_error and ats_hit columns added.
    """
    scores = games_df.select(["game_id", "home_score", "away_score"])
    merged = results.join(scores, on="game_id", how="left")

    merged = merged.with_columns(
        (pl.col("home_score") - pl.col("away_score")).alias("actual_spread")
    )
    merged = merged.with_columns(
        (pl.col("predicted_spread") - pl.col("actual_spread")).alias("model_error")
    )

    if "vegas_spread" in merged.columns:
        model_side = (pl.col("predicted_spread") - pl.col("vegas_spread")).sign()
        actual_side = (pl.col("actual_spread") - pl.col("vegas_spread")).sign()
        merged = merged.with_columns(
            pl.when(pl.col("vegas_spread").is_null() | (actual_side == 0))
            .then(None)
            .otherwise(model_side == actual_side)
            .alias("ats_hit")
        )

    return merged


def summarize_test_results(merged: pl.DataFrame) -> Dict:
    """Print and return accuracy metrics for a TEST-mode run.

    Args:
        merged: output of compare_to_actuals().

    Returns:
        Dict with game count, MAE, and ATS accuracy.
    """
    n_games = len(merged)
    mae = (
        merged.select(pl.col("model_error").abs().mean()).item()
        if n_games
        else 0.0
    )

    ats_acc = None
    ats_graded = 0
    ats_hits = 0
    if "ats_hit" in merged.columns:
        graded = merged.filter(pl.col("ats_hit").is_not_null())
        ats_graded = len(graded)
        if ats_graded:
            ats_hits = int(graded.select(pl.col("ats_hit").sum()).item() or 0)
            ats_acc = ats_hits / ats_graded

    print("\n" + "=" * 60)
    print("TEST MODE -- PREDICTED VS ACTUAL")
    print("=" * 60)
    print(f"Games scored:      {n_games}")
    print(f"Spread MAE:        {mae:.2f} points")
    if ats_acc is not None:
        print(f"ATS accuracy:      {ats_hits}/{ats_graded} = {ats_acc:.1%} "
              f"(break-even 52.4%)")
    else:
        print("ATS accuracy:      n/a (no gradeable games)")
    print("=" * 60)

    return {
        "games_scored": n_games,
        "spread_mae": mae,
        "ats_graded": ats_graded,
        "ats_hits": ats_hits,
        "ats_accuracy": ats_acc,
    }


def load_vegas_lines(vegas_file: Optional[str] = None) -> Optional[pl.DataFrame]:
    """Load Vegas lines from CSV file if provided."""
    if not vegas_file:
        return None

    vegas_path = Path(vegas_file)
    if not vegas_path.exists():
        print(f"Vegas lines file not found: {vegas_file}")
        return None

    print(f"Loading Vegas lines from: {vegas_file}")
    df = pl.read_csv(vegas_path)
    print(f"Loaded Vegas lines for {len(df)} games")
    return df


def load_ensemble_model(model_dir: str) -> Dict:
    """Load trained models from model directory."""
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")

    models = {}

    # Load XGBoost model
    xgb_dir = model_dir / 'xgboost'
    if xgb_dir.exists():
        xgb_path = xgb_dir / 'spread_predictor.json'
        if xgb_path.exists():
            models['xgboost'] = SpreadPredictor(model_path=str(xgb_path))
            print(f"Loaded XGBoost model from {xgb_path}")

    # Load Elo model
    elo_dir = model_dir / 'elo'
    if elo_dir.exists():
        elo_files = list(elo_dir.glob('*.json'))
        if elo_files:
            models['elo'] = EloModel()
            models['elo'].load_model(str(elo_files[0]))
            print(f"Loaded Elo model from {elo_files[0]}")

    # Load Bayesian model if available
    if BAYESIAN_AVAILABLE:
        bayesian_dir = model_dir / 'bayesian'
        if bayesian_dir.exists():
            bayesian_files = list(bayesian_dir.glob('*.pkl'))
            if bayesian_files:
                models['bayesian'] = BayesianStateSpace()
                models['bayesian'].load_model(bayesian_files[0])
                print(f"Loaded Bayesian model from {bayesian_files[0]}")

    # Load Neural model if available
    if NEURAL_AVAILABLE:
        neural_dir = model_dir / 'neural'
        if neural_dir.exists():
            neural_files = list(neural_dir.glob('*.pt'))
            if neural_files:
                models['neural'] = NeuralNetPredictor()
                models['neural'].load_model(neural_files[0])
                print(f"Loaded Neural model from {neural_files[0]}")

    # Load ensemble meta-learner if available
    if ENSEMBLE_AVAILABLE:
        ensemble_file = model_dir / 'stacking_ensemble.pkl'
        if ensemble_file.exists():
            models['ensemble'] = StackingEnsemble()
            models['ensemble'].load_model(ensemble_file)

            # Get required base models from ensemble metadata
            required_models = models['ensemble'].get_required_base_models()

            # Validate all required models are loaded
            missing_models = [name for name in required_models if name not in models]
            if missing_models:
                print(f"Warning: Ensemble requires models {required_models}")
                print(f"Missing models: {missing_models}")
                print("Ensemble predictions will be unavailable.")
                del models['ensemble']
            else:
                # Add ONLY the required base models to ensemble (in correct order)
                for name in required_models:
                    models[name].is_fitted = True  # Mark as fitted since loaded from disk
                    models['ensemble'].add_base_model(name, models[name])
                print(f"Ensemble ready with models: {required_models}")

    # Fail loudly if no spread-capable model loaded. The XGBoost model is the
    # ensemble's primary base model and the fallback when no meta-learner is
    # present; without it generate_predictions cannot produce predicted_spread
    # and would otherwise write null predictions to Snowflake silently.
    if 'xgboost' not in models:
        raise RuntimeError(
            f"No XGBoost model found under '{model_dir}'. Expected "
            f"'{model_dir / 'xgboost' / 'spread_predictor.json'}'. "
            f"Loaded only: {sorted(models.keys()) or 'nothing'}. "
            "Train models before predicting (see src/ml/WEEKLY_WORKFLOW.md)."
        )

    return models


def generate_predictions(
    games_df: pl.DataFrame,
    models: Dict,
    vegas_lines_df: Optional[pl.DataFrame] = None
) -> pl.DataFrame:
    """Generate predictions for upcoming games."""

    if games_df.is_empty():
        return pl.DataFrame()

    # Merge Vegas lines if provided (overrides defaults from schedule)
    if vegas_lines_df is not None:
        games_df = games_df.join(
            vegas_lines_df.select(['game_id', 'vegas_spread', 'vegas_total']),
            on='game_id',
            how='left',
            suffix='_csv'
        )
        # Use CSV values if available
        games_df = games_df.with_columns([
            pl.when(pl.col('vegas_spread_csv').is_not_null())
              .then(pl.col('vegas_spread_csv'))
              .otherwise(pl.col('vegas_spread'))
              .alias('vegas_spread'),
            pl.when(pl.col('vegas_total_csv').is_not_null())
              .then(pl.col('vegas_total_csv'))
              .otherwise(pl.col('vegas_total'))
              .alias('vegas_total')
        ])

    # Select features for prediction
    feature_cols = select_spread_features(games_df)

    # Filter to available features
    available_features = [f for f in feature_cols if f in games_df.columns]
    missing_features = [f for f in feature_cols if f not in games_df.columns]

    if missing_features:
        print(f"Warning: Missing features (using 0): {missing_features}")

    # Prepare feature matrix
    X = games_df.select(available_features).to_numpy()

    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0)

    # Generate predictions
    predictions = {}

    # XGBoost prediction
    if 'xgboost' in models:
        xgb_preds = models['xgboost'].predict(X)
        predictions['xgboost_spread'] = xgb_preds

    # Elo prediction (needs team names)
    if 'elo' in models:
        elo_preds = []
        for i in range(len(games_df)):
            home = games_df['home_team'][i]
            away = games_df['away_team'][i]
            try:
                pred = models['elo'].predict_spread(home, away)
                elo_preds.append(pred)
            except Exception:
                elo_preds.append(0.0)
        predictions['elo_spread'] = np.array(elo_preds)

    # Bayesian prediction (needs team names)
    if 'bayesian' in models and BAYESIAN_AVAILABLE:
        try:
            home_teams = games_df['home_team'].to_list()
            away_teams = games_df['away_team'].to_list()
            bayesian_preds = models['bayesian'].predict_batch(home_teams, away_teams)
            predictions['bayesian_spread'] = bayesian_preds
        except Exception as e:
            print(f"Bayesian prediction failed: {e}")

    # Neural prediction
    if 'neural' in models and NEURAL_AVAILABLE:
        try:
            neural_preds = models['neural'].predict(X)
            predictions['neural_spread'] = neural_preds
        except Exception as e:
            print(f"Neural prediction failed: {e}")

    # Ensemble prediction
    if 'ensemble' in models and ENSEMBLE_AVAILABLE:
        try:
            home_teams = games_df['home_team'].to_list()
            away_teams = games_df['away_team'].to_list()
            ensemble_preds = models['ensemble'].predict(X, home_teams, away_teams)
            predictions['predicted_spread'] = ensemble_preds
        except Exception as e:
            print(f"Ensemble prediction failed, using XGBoost: {e}")
            if 'xgboost_spread' in predictions:
                predictions['predicted_spread'] = predictions['xgboost_spread']
    elif 'xgboost' in models:
        # Fallback to XGBoost if no ensemble
        predictions['predicted_spread'] = predictions['xgboost_spread']

    # Refuse to emit results without a spread prediction rather than writing
    # null spreads/edges/recommendations that look like a successful run.
    if 'predicted_spread' not in predictions:
        raise RuntimeError(
            "No model produced a spread prediction (need the XGBoost model or a "
            "complete stacking ensemble). Refusing to write null predictions."
        )

    # Build results DataFrame
    results = games_df.select([
        'game_id', 'season', 'week', 'gameday', 'gametime',
        'home_team', 'away_team', 'vegas_spread', 'vegas_total'
    ])

    # Add predictions
    for col, values in predictions.items():
        results = results.with_columns(
            pl.Series(name=col, values=values.flatten() if hasattr(values, 'flatten') else values)
        )

    # Calculate edge and betting recommendation
    if 'predicted_spread' in predictions:
        pred_spread = predictions['predicted_spread'].flatten()
        vegas_spread = games_df['vegas_spread'].to_numpy()

        # Edge = how much model disagrees with Vegas
        edge = pred_spread - vegas_spread

        # Home win probability (logistic approximation)
        home_win_prob = 1 / (1 + np.exp(-pred_spread / 5.5))

        # Betting recommendation based on edge threshold
        recommendations = []
        for e in edge:
            if e >= 3.0:
                recommendations.append('BET HOME')
            elif e <= -3.0:
                recommendations.append('BET AWAY')
            else:
                recommendations.append('NO BET')

        # Confidence level
        confidence = []
        for e in edge:
            abs_edge = abs(e)
            if abs_edge >= 5.0:
                confidence.append('High')
            elif abs_edge >= 3.0:
                confidence.append('Medium')
            else:
                confidence.append('Low')

        results = results.with_columns([
            pl.Series(name='edge', values=np.round(edge, 2)),
            pl.Series(name='home_win_prob', values=np.round(home_win_prob, 3)),
            pl.Series(name='bet_recommendation', values=recommendations),
            pl.Series(name='confidence', values=confidence)
        ])

    return results


def write_to_snowflake(df: pl.DataFrame, week: int, season: int):
    """Write predictions to Snowflake ML.PREDICTIONS table."""
    print("Writing predictions to Snowflake...")

    conn = get_snowflake_connection()
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS PRODUCTION_ANALYTICS.ML.PREDICTIONS (
            game_id VARCHAR,
            season NUMBER,
            week NUMBER,
            gameday DATE,
            gametime VARCHAR,
            home_team VARCHAR,
            away_team VARCHAR,
            vegas_spread FLOAT,
            vegas_total FLOAT,
            predicted_spread FLOAT,
            edge FLOAT,
            home_win_prob FLOAT,
            bet_recommendation VARCHAR,
            confidence VARCHAR,
            prediction_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)

    # Delete existing predictions for this week/season
    cursor.execute(f"""
        DELETE FROM PRODUCTION_ANALYTICS.ML.PREDICTIONS
        WHERE season = {season} AND week = {week}
    """)

    # Insert new predictions
    for row in df.iter_rows(named=True):
        cursor.execute(f"""
            INSERT INTO PRODUCTION_ANALYTICS.ML.PREDICTIONS
            (game_id, season, week, gameday, gametime, home_team, away_team,
             vegas_spread, vegas_total, predicted_spread, edge, home_win_prob,
             bet_recommendation, confidence)
            VALUES (
                '{row.get("game_id", "")}',
                {row.get("season", 0)},
                {row.get("week", 0)},
                '{row.get("gameday", "")}',
                '{row.get("gametime", "")}',
                '{row.get("home_team", "")}',
                '{row.get("away_team", "")}',
                {row.get("vegas_spread") or 'NULL'},
                {row.get("vegas_total") or 'NULL'},
                {row.get("predicted_spread") or 'NULL'},
                {row.get("edge") or 'NULL'},
                {row.get("home_win_prob") or 'NULL'},
                '{row.get("bet_recommendation", "")}',
                '{row.get("confidence", "")}'
            )
        """)

    conn.commit()
    cursor.close()
    conn.close()

    print(f"Wrote {len(df)} predictions to PRODUCTION_ANALYTICS.ML.PREDICTIONS")


def main():
    parser = argparse.ArgumentParser(
        description="Generate NFL spread predictions for upcoming games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/ml/predict.py --week 5 --season 2025 --output predictions_week5.csv
  python src/ml/predict.py --week 5 --season 2025 --output predictions.csv --snowflake
  python src/ml/predict.py --week 5 --season 2025 --vegas-file data/vegas_lines.csv
        """
    )

    parser.add_argument('--week', type=int, required=True,
                        help='Week number to predict')
    parser.add_argument('--season', type=int, required=True,
                        help='Season year')
    parser.add_argument('--mode', type=str, default='prod',
                        choices=['prod', 'test'],
                        help='prod = upcoming games; test = replay a completed '
                             'historical week and score vs actual results')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV file path (default: predictions.csv)')
    parser.add_argument('--model-dir', type=str, default='src/ml/models/ensemble',
                        help='Directory containing trained models')
    parser.add_argument('--vegas-file', type=str, default=None,
                        help='CSV file with Vegas lines (optional, overrides schedule)')
    parser.add_argument('--snowflake', action='store_true',
                        help='Also write predictions to Snowflake ML.PREDICTIONS table')

    args = parser.parse_args()

    print("=" * 60)
    print("NFL SPREAD PREDICTION")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Week: {args.week}")
    print(f"Season: {args.season}")
    print(f"Output: {args.output}")
    print(f"Model Directory: {args.model_dir}")
    print("=" * 60)

    test_mode = args.mode == 'test'

    # Load games. TEST mode replays a completed historical week (and runs
    # preflight validation first); PROD mode loads upcoming games.
    if test_mode:
        from src.pipeline.config import PipelineMode, PipelineRunConfig
        from src.pipeline.preflight import PreflightError, run_preflight

        try:
            plan = run_preflight(
                PipelineRunConfig(
                    mode=PipelineMode.TEST, week=args.week, season=args.season
                )
            )
        except PreflightError as exc:
            print(f"\nPreflight validation FAILED:\n{exc}")
            return 1
        print("\n" + plan.render())
        games_df = load_historical_games(args.week, args.season)
    else:
        games_df = load_upcoming_games(args.week, args.season)

    if games_df.is_empty():
        print("No games to predict. Exiting.")
        return 1

    # Load Vegas lines if provided
    vegas_df = load_vegas_lines(args.vegas_file)

    # Load models
    print("\nLoading models...")
    models = load_ensemble_model(args.model_dir)

    if not models:
        print("No models loaded. Exiting.")
        return 1

    # Generate predictions
    print("\nGenerating predictions...")
    results = generate_predictions(games_df, models, vegas_df)

    if results.is_empty():
        print("No predictions generated. Exiting.")
        return 1

    # TEST mode: score predictions against actual historical results.
    if test_mode:
        results = compare_to_actuals(results, games_df)
        summarize_test_results(results)

    # Write to CSV (ensure the output directory exists)
    print(f"\nWriting {len(results)} predictions to {args.output}...")
    output_path = Path(args.output)
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    results.write_csv(str(output_path))
    print(f"Predictions saved to: {args.output}")

    # Write to Snowflake if requested. PROD only -- test runs never touch the
    # production ML.PREDICTIONS table.
    if args.snowflake and not test_mode:
        write_to_snowflake(results, args.week, args.season)
    elif args.snowflake and test_mode:
        print("Skipping Snowflake write: --snowflake is ignored in test mode.")

    # Print summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)

    bet_home = results.filter(pl.col('bet_recommendation') == 'BET HOME')
    bet_away = results.filter(pl.col('bet_recommendation') == 'BET AWAY')
    no_bet = results.filter(pl.col('bet_recommendation') == 'NO BET')

    print(f"Total games: {len(results)}")
    print(f"BET HOME: {len(bet_home)}")
    print(f"BET AWAY: {len(bet_away)}")
    print(f"NO BET: {len(no_bet)}")

    if len(bet_home) + len(bet_away) > 0:
        print("\nRecommended bets:")
        recommended = results.filter(pl.col('bet_recommendation') != 'NO BET')
        for row in recommended.iter_rows(named=True):
            print(f"  {row['away_team']} @ {row['home_team']}: "
                  f"{row['bet_recommendation']} (edge: {row['edge']:.1f}, "
                  f"conf: {row['confidence']})")

    print("\n" + "=" * 60)
    print("Open the output CSV in Excel to review all predictions.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
