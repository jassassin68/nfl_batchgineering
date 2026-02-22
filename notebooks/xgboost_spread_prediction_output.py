"""
NFL Spread Prediction Script
Loads model, queries Snowflake for upcoming games, generates predictions with full diagnostics.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import polars as pl
import numpy as np
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.models.spread_predictor import SpreadPredictor


DEFAULT_MODEL_PATH = "C:/Users/jasse/_GitHub_repositories/nfl_batchgineering/src/ml/models/ensemble/base_models/xgboost/spread_predictor.json"


# =============================================================================
# SNOWFLAKE CONNECTION
# =============================================================================

def get_snowflake_connection():
    """Create Snowflake connection using key pair authentication."""
    import snowflake.connector
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    load_dotenv()

    private_key_text = os.getenv('SNOWFLAKE_KEYPAIR_PRIVATE_KEY')
    passphrase = os.getenv('SNOWFLAKE_KEYPAIR_PASSPHRASE')

    if not private_key_text:
        raise ValueError("SNOWFLAKE_KEYPAIR_PRIVATE_KEY not found in environment")

    # Replace literal \n with actual newlines
    private_key_text = private_key_text.replace('\\n', '\n')

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
    print(f"\nQuerying Snowflake for Week {week}, Season {season}...")

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

    columns = [desc[0].lower() for desc in cursor.description]
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame({col: [row[i] for row in rows] for i, col in enumerate(columns)})
    print(f"✅ Loaded {len(df)} upcoming games")

    return df


# =============================================================================
# PREDICTION & DIAGNOSTICS
# =============================================================================

def run_predictions(
    model_path: str,
    week: int,
    season: int,
    output_path: Optional[str] = None
) -> Optional[pl.DataFrame]:
    """
    Full prediction pipeline with diagnostics.
    """

    # -------------------------------------------------------------------------
    # 1. Load the trained model
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)

    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return None

    predictor = SpreadPredictor(model_path=model_path)

    # Get feature columns FROM the model
    feature_columns = predictor.feature_names

    print(f"\nModel expects {len(feature_columns)} features:")
    for i, feat in enumerate(feature_columns[:10], 1):
        print(f"  {i:2d}. {feat}")
    if len(feature_columns) > 10:
        print(f"  ... and {len(feature_columns) - 10} more")

    # -------------------------------------------------------------------------
    # 2. Print model summary
    # -------------------------------------------------------------------------
    predictor.summary()

    # -------------------------------------------------------------------------
    # 3. Load upcoming games from Snowflake
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("LOADING UPCOMING GAMES")
    print("="*70)

    games_df = load_upcoming_games(week, season)

    if games_df.is_empty():
        print(f"❌ No games found for Week {week}, Season {season}")
        return None

    print(f"\nAvailable columns in mart: {len(games_df.columns)}")

    # -------------------------------------------------------------------------
    # 4. Validate that mart contains required features
    # -------------------------------------------------------------------------
    available_columns = set(games_df.columns)
    missing = [f for f in feature_columns if f not in available_columns]

    if missing:
        print(f"\n❌ FATAL: Mart is missing {len(missing)} required features:")
        for feat in missing:
            print(f"     - {feat}")
        print("\nYour mart schema doesn't match what the model was trained on.")
        print("Either retrain the model or update your dbt mart.")
        return None

    print(f"\n✅ All {len(feature_columns)} required features found in mart")

    # -------------------------------------------------------------------------
    # 5. Make predictions
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("PREDICTIONS")
    print("="*70)

    results_df = predictor.predict_games(games_df, feature_columns)

    # Display predictions for each game
    for row in results_df.iter_rows(named=True):
        away = row.get('away_team', row.get('away_team_abbr', 'AWAY'))
        home = row.get('home_team', row.get('home_team_abbr', 'HOME'))
        gameday = row.get('gameday', '')

        print(f"\n{away} @ {home}  ({gameday})")
        print(f"  Predicted Spread: {row['predicted_spread']:+.1f}")
        print(f"  Confidence: {row['prediction_confidence']:.1%}")

        if row['predicted_spread'] > 0:
            print(f"  → {home} favored by {row['predicted_spread']:.1f}")
        elif row['predicted_spread'] < 0:
            print(f"  → {away} favored by {abs(row['predicted_spread']):.1f}")
        else:
            print(f"  → Pick'em")

    # -------------------------------------------------------------------------
    # 6. Feature importance
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE")
    print("="*70)

    for importance_type in ['gain', 'weight', 'cover']:
        print(f"\n--- By {importance_type.upper()} ---")
        importance = predictor.get_feature_importance(importance_type=importance_type)
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        for i, (feature, score) in enumerate(sorted_features[:15], 1):
            print(f"  {i:2d}. {feature:40s} {score:10.2f}")

    # -------------------------------------------------------------------------
    # 7. Prediction statistics
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("PREDICTION STATISTICS")
    print("="*70)

    spreads = results_df['predicted_spread'].to_numpy()
    confidences = results_df['prediction_confidence'].to_numpy()

    print(f"  Games predicted:        {len(spreads)}")
    print(f"  Mean spread:            {np.mean(spreads):+.2f}")
    print(f"  Spread std dev:         {np.std(spreads):.2f}")
    print(f"  Min spread:             {np.min(spreads):+.2f}")
    print(f"  Max spread:             {np.max(spreads):+.2f}")
    print(f"  Mean confidence:        {np.mean(confidences):.1%}")
    print(f"  Home favorites:         {np.sum(spreads > 0)}")
    print(f"  Away favorites:         {np.sum(spreads < 0)}")
    print(f"  Pick'ems (within 1 pt): {np.sum(np.abs(spreads) < 1)}")

    # -------------------------------------------------------------------------
    # 8. Save results if requested
    # -------------------------------------------------------------------------
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results_df.write_parquet(output_path)
        print(f"\n✅ Results saved to {output_path}")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70 + "\n")

    return results_df


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate NFL spread predictions for upcoming games"
    )

    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="NFL week number (1-18 for regular season, 19-22 for playoffs)"
    )

    parser.add_argument(
        "--season",
        type=int,
        default=datetime.now().year,
        help="NFL season year (defaults to current year)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to trained model file (default: {DEFAULT_MODEL_PATH})"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions (parquet format)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("\n" + "="*70)
    print(f"NFL SPREAD PREDICTOR - Week {args.week}, {args.season} Season")
    print("="*70)

    results = run_predictions(
        model_path=args.model,
        week=args.week,
        season=args.season,
        output_path=args.output
    )

    if results is None:
        sys.exit(1)

    return results


if __name__ == "__main__":
    main()