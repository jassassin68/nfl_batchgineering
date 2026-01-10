"""
Main training script for NFL spread prediction model.
Loads data from Snowflake, trains XGBoost model, evaluates performance, and saves artifacts.
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
from sklearn.model_selection import train_test_split
import snowflake.connector
from dotenv import load_dotenv
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

from src.ml.models.spread_predictor import SpreadPredictor
from src.ml.utils.feature_engineering import prepare_training_data, get_feature_importance_names
from src.ml.utils.evaluation import create_performance_report, evaluate_spread_model


def load_data_from_snowflake(
    query: str,
    limit: int = None
) -> pl.DataFrame:
    """
    Load training data from Snowflake using Polars.

    Args:
        query: SQL query to execute
        limit: Optional row limit for testing

    Returns:
        Polars DataFrame with game features
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
        'schema': os.getenv('SNOWFLAKE_SCHEMA', 'marts'),
        'role': os.getenv('SNOWFLAKE_ROLE', 'ACCOUNTADMIN')
    }

    # Verify all required params exist
    required_params = ['account', 'user', 'warehouse', 'database']
    missing = [k for k in required_params if not conn_params.get(k)]
    if missing:
        raise ValueError(f"Missing Snowflake credentials in .env: {missing}")

    print(f"Connecting to Snowflake using key pair authentication...")
    print(f"  Account: {conn_params['account']}")
    print(f"  Database: {conn_params['database']}")
    print(f"  Schema: {conn_params['schema']}")

    # Connect to Snowflake
    conn = snowflake.connector.connect(**conn_params)
    cursor = conn.cursor()

    # Add LIMIT if specified
    final_query = query
    if limit:
        final_query = f"{query.rstrip(';')} LIMIT {limit}"

    print(f"\nExecuting query...")
    print(f"  Query preview: {final_query[:200]}...")

    # Execute query
    cursor.execute(final_query)

    # Fetch column names (convert to lowercase to match Python conventions)
    columns = [col[0].lower() for col in cursor.description]

    # Fetch all rows
    rows = cursor.fetchall()

    # Close connection
    cursor.close()
    conn.close()

    print(f"✅ Loaded {len(rows):,} rows with {len(columns)} columns")

    # Convert to Polars DataFrame
    df = pl.DataFrame(rows, schema=columns, orient='row')

    return df


def main(args):
    """Main training pipeline."""

    print("\n" + "="*80)
    print("NFL SPREAD PREDICTION MODEL TRAINING")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # =========================================================================
    # 1. LOAD DATA FROM SNOWFLAKE
    # =========================================================================

    print("STEP 1: Loading training data from Snowflake...")
    print("-" * 80)

    query = """
    SELECT
        game_id,
        season,
        week,
        home_team,
        away_team,
        home_score,
        away_score,

        -- Home team features
        home_epa_adj,
        home_epa_l4w,
        home_success_rate,
        home_success_l4w,
        home_explosive_rate,
        home_pass_epa,
        home_run_epa,
        home_def_epa,
        home_def_rank,
        home_def_pass_epa,
        home_def_run_epa,
        home_def_epa_l4w,
        home_rz_td_rate,
        home_third_conv,
        home_two_min_epa,

        -- Away team features
        away_epa_adj,
        away_epa_l4w,
        away_success_rate,
        away_success_l4w,
        away_explosive_rate,
        away_pass_epa,
        away_run_epa,
        away_def_epa,
        away_def_rank,
        away_def_pass_epa,
        away_def_run_epa,
        away_def_epa_l4w,
        away_rz_td_rate,
        away_third_conv,
        away_two_min_epa,

        -- Context features
        temp,
        wind,
        div_game,
        playoff

    FROM marts.mart_game_prediction_features
    WHERE season >= {min_season}
        AND home_score IS NOT NULL  -- Only completed games
        AND away_score IS NOT NULL
    ORDER BY season, week, game_id
    """.format(min_season=args.min_season)

    df = load_data_from_snowflake(query, limit=args.limit)

    print(f"\nDataset summary:")
    print(f"  Seasons: {df['season'].min()} - {df['season'].max()}")
    print(f"  Games: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # =========================================================================
    # 2. PREPARE FEATURES AND TARGET
    # =========================================================================

    print("\n" + "="*80)
    print("STEP 2: Feature engineering and data preparation...")
    print("-" * 80)

    df_prepared, feature_cols, target_col = prepare_training_data(
        df,
        target_type='spread',
        include_derived_features=args.include_derived
    )

    # =========================================================================
    # 3. TRAIN/TEST SPLIT (TIME-BASED)
    # =========================================================================

    print("\n" + "="*80)
    print("STEP 3: Splitting data (time-based)...")
    print("-" * 80)

    # Use last season as test set
    test_season = df_prepared['season'].max()
    train_mask = df_prepared['season'] < test_season

    df_train = df_prepared.filter(train_mask)
    df_test = df_prepared.filter(~train_mask)

    print(f"Training set:")
    print(f"  Seasons: {df_train['season'].min()} - {df_train['season'].max()}")
    print(f"  Games: {len(df_train):,}")

    print(f"\nTest set:")
    print(f"  Season: {test_season}")
    print(f"  Games: {len(df_test):,}")

    # Extract features and target
    X_train = df_train.select(feature_cols).to_numpy()
    y_train = df_train[target_col].to_numpy()

    X_test = df_test.select(feature_cols).to_numpy()
    y_test = df_test[target_col].to_numpy()

    # Further split training into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42
    )

    print(f"\nFinal split:")
    print(f"  Training: {len(X_train):,} games")
    print(f"  Validation: {len(X_val):,} games")
    print(f"  Test: {len(X_test):,} games")

    # =========================================================================
    # 4. TRAIN MODEL
    # =========================================================================

    print("\n" + "="*80)
    print("STEP 4: Training XGBoost model...")
    print("-" * 80)

    # Initialize model with hyperparameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'n_estimators': args.n_estimators,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'tree_method': 'hist',
        'eval_metric': 'mae'
    }

    print(f"Hyperparameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    model = SpreadPredictor(params=params)

    # Train model
    print(f"\nTraining model...")
    evals_result = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_cols,
        early_stopping_rounds=args.early_stopping,
        verbose=True
    )

    # =========================================================================
    # 5. EVALUATE MODEL
    # =========================================================================

    print("\n" + "="*80)
    print("STEP 5: Evaluating model performance...")
    print("-" * 80)

    # Predictions on test set
    y_pred = model.predict(X_test)

    # Comprehensive evaluation
    feature_importance = model.model.get_score(importance_type='gain')
    importance_array = np.array([feature_importance.get(f, 0.0) for f in feature_cols])

    metrics = create_performance_report(
        y_test,
        y_pred,
        feature_cols,
        importance_array,
        output_dir=args.output_dir
    )

    # =========================================================================
    # 6. SAVE MODEL
    # =========================================================================

    print("\n" + "="*80)
    print("STEP 6: Saving model artifacts...")
    print("-" * 80)

    model_name = f"spread_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model.save_model(
        model_dir=args.output_dir,
        model_name=model_name
    )

    # Save training summary
    summary_path = f"{args.output_dir}/training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("NFL SPREAD PREDICTION MODEL - TRAINING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Name: {model_name}\n\n")
        f.write("Dataset:\n")
        f.write(f"  Training games: {len(X_train):,}\n")
        f.write(f"  Validation games: {len(X_val):,}\n")
        f.write(f"  Test games: {len(X_test):,}\n")
        f.write(f"  Features: {len(feature_cols)}\n\n")
        f.write("Hyperparameters:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTest Set Performance:\n")
        f.write(f"  MAE: {metrics['mae']:.2f} points\n")
        f.write(f"  RMSE: {metrics['rmse']:.2f} points\n")
        f.write(f"  R²: {metrics['r_squared']:.3f}\n")
        f.write(f"  Directional Accuracy: {metrics['directional_accuracy']:.1%}\n")
        f.write(f"  ATS Accuracy: {metrics['ats_accuracy']:.1%}\n")
        f.write(f"  Betting ROI: {metrics['betting_roi']:.2%}\n")
        f.write("\nTop 10 Features:\n")
        sorted_importance = sorted(
            zip(feature_cols, importance_array),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for i, (feat, imp) in enumerate(sorted_importance, 1):
            f.write(f"  {i:2d}. {feat:30s} {imp:8.2f}\n")

    print(f"Training summary saved to {summary_path}")

    # =========================================================================
    # 7. FINAL SUMMARY
    # =========================================================================

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\n✅ Model successfully trained and saved!")
    print(f"\n📊 Key Results:")
    print(f"  MAE: {metrics['mae']:.2f} points")
    print(f"  Directional Accuracy: {metrics['directional_accuracy']:.1%}")
    print(f"  Betting ROI: {metrics['betting_roi']:.2%}")
    print(f"\n📁 Artifacts saved to: {args.output_dir}/")
    print(f"  - Model: {model_name}.json")
    print(f"  - Metadata: {model_name}_metadata.json")
    print(f"  - Features: {model_name}_features.json")
    print(f"  - Plots: predictions_vs_actual.png, residuals_distribution.png, feature_importance.png")
    print(f"  - Summary: training_summary.txt")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NFL spread prediction model")

    # Data parameters
    parser.add_argument('--min-season', type=int, default=2015,
                        help='Minimum season to include in training (default: 2015)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of rows for testing (default: None)')

    # Feature engineering
    parser.add_argument('--include-derived', action='store_true', default=True,
                        help='Include derived/interaction features (default: True)')

    # Model hyperparameters
    parser.add_argument('--max-depth', type=int, default=5,
                        help='XGBoost max tree depth (default: 5)')
    parser.add_argument('--learning-rate', type=float, default=0.05,
                        help='XGBoost learning rate (default: 0.05)')
    parser.add_argument('--n-estimators', type=int, default=300,
                        help='Number of boosting rounds (default: 300)')
    parser.add_argument('--early-stopping', type=int, default=50,
                        help='Early stopping rounds (default: 50)')

    # Output
    parser.add_argument('--output-dir', type=str, default='ml_models',
                        help='Output directory for model artifacts (default: ml_models)')

    args = parser.parse_args()

    # Run training
    try:
        main(args)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
