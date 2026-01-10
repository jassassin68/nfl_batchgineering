"""
Train Elo rating model and establish baseline performance.
This provides a simple baseline to compare against more complex ML models.
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
from dotenv import load_dotenv
import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

from src.ml.models.elo_model import EloModel
from src.ml.utils.evaluation import evaluate_spread_model, create_performance_report


def load_data_from_snowflake(query: str, limit: int = None) -> pl.DataFrame:
    """
    Load training data from Snowflake using Polars.

    Args:
        query: SQL query to execute
        limit: Optional row limit for testing

    Returns:
        Polars DataFrame with game data
    """
    # Load environment variables
    load_dotenv()

    # Load private key for key pair authentication
    private_key_text = os.getenv('SNOWFLAKE_KEYPAIR_PRIVATE_KEY')
    passphrase = os.getenv('SNOWFLAKE_KEYPAIR_PASSPHRASE')

    if not private_key_text:
        raise ValueError("Missing SNOWFLAKE_KEYPAIR_PRIVATE_KEY environment variable")

    # Replace literal \n with actual newlines (needed for .env file format)
    private_key_text = private_key_text.replace('\\n', '\n')

    # Decode the private key
    try:
        private_key = serialization.load_pem_private_key(
            private_key_text.encode(),
            password=passphrase.encode() if passphrase else None,
            backend=default_backend()
        )
    except Exception as e:
        raise ValueError(
            f"Failed to load private key: {e}\n"
            "Make sure your SNOWFLAKE_KEYPAIR_PRIVATE_KEY in .env is formatted correctly.\n"
            "The key should be on a single line with \\n for newlines, like:\n"
            'SNOWFLAKE_KEYPAIR_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\\nMIIE...\\n-----END PRIVATE KEY-----"'
        )

    # Get private key bytes in DER format
    pkb = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    # Get Snowflake credentials from environment
    conn_params = {
        'account': os.getenv('SNOWFLAKE_ACCOUNT'),
        'user': os.getenv('SNOWFLAKE_USER'),
        'private_key': pkb,  # Use key pair authentication instead of password
        'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
        'database': os.getenv('SNOWFLAKE_DATABASE'),
        'schema': os.getenv('SNOWFLAKE_SCHEMA'),
        'role': os.getenv('SNOWFLAKE_ROLE')
    }

    # Validate credentials
    required_params = ['account', 'user', 'warehouse', 'database', 'schema', 'role']
    missing = [k for k in required_params if not conn_params.get(k)]
    if missing:
        raise ValueError(f"Missing Snowflake credentials: {missing}")

    print(f"Connecting to Snowflake (account: {conn_params['account']}) using key pair authentication...")

    # Add limit to query if specified
    if limit:
        query = f"{query.rstrip(';')} LIMIT {limit}"

    # Connect and fetch data
    with snowflake.connector.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            columns = [desc[0].lower() for desc in cur.description]  # Convert to lowercase
            data = cur.fetchall()

    df = pl.DataFrame(data, schema=columns, orient="row")
    print(f"Loaded {len(df):,} games from Snowflake")

    return df


def main(args):
    """Train Elo model on historical games."""

    print("=" * 80)
    print("ELO RATING MODEL TRAINING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Minimum season: {args.min_season}")
    print(f"K-factor: {args.k_factor}")
    print(f"Home advantage: {args.home_advantage}")
    print("=" * 80)

    # Build query for game data
    query = f"""
        SELECT
            game_id,
            season,
            week,
            home_team,
            away_team,
            home_score,
            away_score,
            vegas_spread
        FROM production_analytics.analytics.mart_game_prediction_features
        WHERE season >= {args.min_season}
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY season, week, game_id
    """

    # Load data from Snowflake
    print("\nLoading data from Snowflake...")
    df = load_data_from_snowflake(query, limit=args.limit)

    # Convert numeric columns from strings (Snowflake returns everything as VARCHAR)
    print("Converting data types...")
    df = df.with_columns([
        pl.col('season').cast(pl.Int32),
        pl.col('week').cast(pl.Int32),
        pl.col('home_score').cast(pl.Int32),
        pl.col('away_score').cast(pl.Int32),
        pl.col('vegas_spread').cast(pl.Float64, strict=False)
    ])

    # Validate data
    print("\nData validation:")
    print(f"  Total games: {len(df):,}")
    print(f"  Seasons: {df['season'].min()} - {df['season'].max()}")
    print(f"  Unique teams: {df['home_team'].n_unique()}")
    print(f"  Weeks per season: {df.group_by('season').agg(pl.col('week').n_unique())}")

    # Split into train/test (last season as test)
    test_season = df['season'].max()
    train_df = df.filter(pl.col('season') < test_season)
    test_df = df.filter(pl.col('season') == test_season)

    print(f"\nTrain/Test Split:")
    print(f"  Train: {len(train_df):,} games ({train_df['season'].min()}-{train_df['season'].max()})")
    print(f"  Test:  {len(test_df):,} games (season {test_season})")

    # Initialize Elo model
    print("\n" + "=" * 80)
    print("TRAINING ELO MODEL")
    print("=" * 80)

    elo = EloModel(
        k_factor=args.k_factor,
        home_advantage=args.home_advantage,
        initial_rating=args.initial_rating,
        regression_factor=args.regression_factor,
        mean_rating=args.mean_rating
    )

    print(f"\nModel configuration:")
    print(f"  K-factor: {elo.k_factor}")
    print(f"  Home advantage: {elo.home_advantage} Elo (~{elo.home_advantage/25:.1f} point spread)")
    print(f"  Initial rating: {elo.initial_rating}")
    print(f"  Regression factor: {elo.regression_factor}")
    print(f"  Mean rating: {elo.mean_rating}")

    # Train Elo model
    print("\nTraining Elo model on historical games...")
    elo.fit(train_df)

    # Display current ratings
    print("\n" + "=" * 80)
    print("CURRENT ELO RATINGS (Top 10)")
    print("=" * 80)
    current_ratings = elo.get_current_ratings()
    print(current_ratings.head(10))

    # Make predictions on test set
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)
    test_with_preds = elo.predict(test_df)

    # Calculate actual spreads
    test_with_preds = test_with_preds.with_columns(
        (pl.col('home_score') - pl.col('away_score')).alias('actual_spread')
    )

    # Evaluate model
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    # Convert numeric columns from strings (Snowflake returns as VARCHAR)
    test_with_preds = test_with_preds.with_columns([
        pl.col('actual_spread').cast(pl.Float64),
        pl.col('elo_predicted_spread').cast(pl.Float64)
    ])

    y_true = test_with_preds['actual_spread'].to_numpy()
    y_pred = test_with_preds['elo_predicted_spread'].to_numpy()

    # Get evaluation metrics
    metrics = evaluate_spread_model(y_true, y_pred, verbose=True)

    # Evaluate against Vegas line if available
    if 'vegas_spread' in test_with_preds.columns:
        # Convert vegas_spread to float, handling nulls
        test_with_preds = test_with_preds.with_columns([
            pl.col('vegas_spread').cast(pl.Float64, strict=False)
        ])
        vegas_spread = test_with_preds['vegas_spread'].to_numpy()
        valid_mask = ~np.isnan(vegas_spread)

        if valid_mask.sum() > 0:
            print("\n" + "=" * 80)
            print("VEGAS COMPARISON (Games with Vegas lines)")
            print("=" * 80)

            # Elo vs Vegas
            elo_vs_vegas = evaluate_spread_model(
                y_true[valid_mask],
                y_pred[valid_mask],
                verbose=False
            )

            vegas_baseline = evaluate_spread_model(
                y_true[valid_mask],
                vegas_spread[valid_mask],
                verbose=False
            )

            print(f"\n  Elo MAE:   {elo_vs_vegas['mae']:.2f} points")
            print(f"  Vegas MAE: {vegas_baseline['mae']:.2f} points")
            print(f"  Improvement: {vegas_baseline['mae'] - elo_vs_vegas['mae']:.2f} points")

            # Calculate ATS accuracy (Against The Spread)
            # Pick side that disagrees most with Vegas
            spread_diff = y_pred[valid_mask] - vegas_spread[valid_mask]
            elo_picks = np.where(spread_diff > 0, 1, -1)  # 1 = pick home, -1 = pick away
            actual_vs_vegas = y_true[valid_mask] - vegas_spread[valid_mask]
            correct = (elo_picks * actual_vs_vegas) > 0

            ats_accuracy = correct.mean()
            print(f"\n  ATS Accuracy: {ats_accuracy:.1%} ({correct.sum()}/{len(correct)} games)")
            print(f"  (Threshold to beat -110 juice: 52.4%)")

    # Save model
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    output_path = elo.save_model(args.output_dir, 'elo_baseline')

    # Save rating history
    rating_history = elo.get_rating_history()
    if len(rating_history) > 0:
        history_path = Path(args.output_dir) / 'elo_rating_history.parquet'
        rating_history.write_parquet(history_path)
        print(f"Rating history saved to: {history_path}")

    # Save test predictions
    pred_path = Path(args.output_dir) / 'elo_test_predictions.parquet'
    test_with_preds.write_parquet(pred_path)
    print(f"Test predictions saved to: {pred_path}")

    # Create performance report
    if args.create_report:
        print("\n" + "=" * 80)
        print("CREATING PERFORMANCE REPORT")
        print("=" * 80)

        report_data = {
            'model_type': 'Elo Baseline',
            'train_games': len(train_df),
            'test_games': len(test_df),
            'test_season': test_season,
            'metrics': metrics,
            'hyperparameters': {
                'k_factor': elo.k_factor,
                'home_advantage': elo.home_advantage,
                'initial_rating': elo.initial_rating,
                'regression_factor': elo.regression_factor,
                'mean_rating': elo.mean_rating
            }
        }

        report_path = create_performance_report(
            y_true=y_true,
            y_pred=y_pred,
            model_name='elo_baseline',
            output_dir=args.output_dir,
            metadata=report_data
        )
        print(f"Performance report saved to: {report_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Elo rating model for NFL spread predictions'
    )

    # Data parameters
    parser.add_argument(
        '--min-season',
        type=int,
        default=2020,
        help='Minimum season to include in training (default: 2020)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of games for testing (default: no limit)'
    )

    # Elo hyperparameters
    parser.add_argument(
        '--k-factor',
        type=float,
        default=20.0,
        help='Elo K-factor learning rate (default: 20.0)'
    )
    parser.add_argument(
        '--home-advantage',
        type=float,
        default=48.0,
        help='Home field advantage in Elo points (default: 48.0)'
    )
    parser.add_argument(
        '--initial-rating',
        type=float,
        default=1500.0,
        help='Initial Elo rating for new teams (default: 1500.0)'
    )
    parser.add_argument(
        '--regression-factor',
        type=float,
        default=0.33,
        help='Season-to-season regression factor (default: 0.33)'
    )
    parser.add_argument(
        '--mean-rating',
        type=float,
        default=1505.0,
        help='Mean rating for regression (default: 1505.0)'
    )

    # Output parameters
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/elo_baseline',
        help='Directory to save model artifacts (default: models/elo_baseline)'
    )
    parser.add_argument(
        '--create-report',
        action='store_true',
        help='Create detailed performance report with visualizations'
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run training
    try:
        metrics = main(args)
        print("\n✓ Training completed successfully")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
