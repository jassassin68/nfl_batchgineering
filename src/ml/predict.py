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
                recommendations.append('BET AWAY')
            elif e <= -3.0:
                recommendations.append('BET HOME')
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
    print(f"Week: {args.week}")
    print(f"Season: {args.season}")
    print(f"Output: {args.output}")
    print(f"Model Directory: {args.model_dir}")
    print("=" * 60)

    # Load upcoming games
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

    # Write to CSV
    print(f"\nWriting {len(results)} predictions to {args.output}...")
    results.write_csv(args.output)
    print(f"Predictions saved to: {args.output}")

    # Write to Snowflake if requested
    if args.snowflake:
        write_to_snowflake(results, args.week, args.season)

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
