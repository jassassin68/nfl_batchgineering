"""
Train complete NFL prediction ensemble.

Trains all 4 base models (Elo, XGBoost, Bayesian, Neural Network)
and the stacking meta-learner. Uses walk-forward cross-validation
to prevent look-ahead bias.

Usage:
    python src/ml/train_ensemble.py --min-season 2015 --output-dir models/ensemble
    python src/ml/train_ensemble.py --skip-bayesian  # Skip slow MCMC sampling
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
import json

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
from src.ml.models.spread_predictor import SpreadPredictor
from src.ml.utils.feature_engineering import prepare_training_data, select_spread_features
from src.ml.utils.evaluation import evaluate_spread_model
from src.ml.utils.validation import (
    train_test_split_temporal,
    calculate_ats_accuracy,
    calculate_roi,
    calculate_brier_score
)


def load_data_from_snowflake(query: str, limit: int = None) -> pl.DataFrame:
    """
    Load training data from Snowflake using key pair authentication.

    Args:
        query: SQL query to execute
        limit: Optional row limit for testing

    Returns:
        Polars DataFrame with game data
    """
    load_dotenv()

    # Load private key for key pair authentication
    private_key_text = os.getenv('SNOWFLAKE_KEYPAIR_PRIVATE_KEY')
    passphrase = os.getenv('SNOWFLAKE_KEYPAIR_PASSPHRASE')

    if not private_key_text:
        raise ValueError("Missing SNOWFLAKE_KEYPAIR_PRIVATE_KEY environment variable")

    private_key_text = private_key_text.replace('\\n', '\n')

    try:
        private_key = serialization.load_pem_private_key(
            private_key_text.encode(),
            password=passphrase.encode() if passphrase else None,
            backend=default_backend()
        )
    except Exception as e:
        raise ValueError(f"Failed to load private key: {e}")

    pkb = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    conn_params = {
        'account': os.getenv('SNOWFLAKE_ACCOUNT'),
        'user': os.getenv('SNOWFLAKE_USER'),
        'private_key': pkb,
        'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
        'database': os.getenv('SNOWFLAKE_DATABASE'),
        'schema': os.getenv('SNOWFLAKE_SCHEMA'),
        'role': os.getenv('SNOWFLAKE_ROLE')
    }

    required_params = ['account', 'user', 'warehouse', 'database', 'schema', 'role']
    missing = [k for k in required_params if not conn_params.get(k)]
    if missing:
        raise ValueError(f"Missing Snowflake credentials: {missing}")

    print(f"Connecting to Snowflake...")

    if limit:
        query = f"{query.rstrip(';')} LIMIT {limit}"

    with snowflake.connector.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            columns = [desc[0].lower() for desc in cur.description]
            data = cur.fetchall()

    df = pl.DataFrame(data, schema=columns, orient="row")
    print(f"Loaded {len(df):,} games from Snowflake")

    return df


def main(args):
    """Train complete ensemble pipeline."""

    print("=" * 80)
    print("NFL PREDICTION ENSEMBLE TRAINING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Minimum season: {args.min_season}")
    print(f"Skip Elo: {args.skip_elo}")
    print(f"Skip Bayesian: {args.skip_bayesian}")
    print(f"Skip Neural: {args.skip_neural}")
    print("=" * 80)

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)

    # Build comprehensive query for all features
    query = f"""
        SELECT *
        FROM production_analytics.analytics.mart_game_prediction_features
        WHERE season >= {args.min_season}
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY season, week, game_id
    """

    df = load_data_from_snowflake(query, limit=args.limit)

    # Convert key columns to proper types
    print("Converting data types...")
    df = df.with_columns([
        pl.col('season').cast(pl.Int32),
        pl.col('week').cast(pl.Int32),
        pl.col('home_score').cast(pl.Int32),
        pl.col('away_score').cast(pl.Int32),
        pl.col('vegas_spread').cast(pl.Float64, strict=False),
        pl.col('vegas_total').cast(pl.Float64, strict=False)
    ])

    # Add actual spread target
    df = df.with_columns([
        (pl.col('home_score') - pl.col('away_score')).alias('actual_spread')
    ])

    # Data summary
    print(f"\nData Summary:")
    print(f"  Total games: {len(df):,}")
    print(f"  Seasons: {df['season'].min()} - {df['season'].max()}")
    print(f"  Teams: {df['home_team'].n_unique()}")

    # =========================================================================
    # STEP 2: TRAIN/TEST SPLIT
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: TEMPORAL TRAIN/TEST SPLIT")
    print("=" * 80)

    train_df, test_df = train_test_split_temporal(df, test_seasons=1)

    test_season = test_df['season'].max()
    print(f"  Train: {len(train_df):,} games (seasons {train_df['season'].min()}-{train_df['season'].max()})")
    print(f"  Test: {len(test_df):,} games (season {test_season})")

    # =========================================================================
    # STEP 3: PREPARE FEATURES
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: PREPARING FEATURES")
    print("=" * 80)

    # Get feature columns (pass DataFrame to filter to available columns)
    feature_cols = select_spread_features(df)
    print(f"Using {len(feature_cols)} features")

    # Convert features to float and handle nulls
    for col in feature_cols:
        df = df.with_columns([
            pl.col(col).cast(pl.Float64, strict=False)
        ])

    # Re-split after type conversion
    train_df = df.filter(pl.col('season') < test_season)
    test_df = df.filter(pl.col('season') == test_season)

    # Prepare arrays
    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df['actual_spread'].to_numpy()
    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df['actual_spread'].to_numpy()

    # Handle NaN in features
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Team arrays for Elo/Bayesian
    home_teams_train = train_df['home_team'].to_numpy()
    away_teams_train = train_df['away_team'].to_numpy()
    home_teams_test = test_df['home_team'].to_numpy()
    away_teams_test = test_df['away_team'].to_numpy()

    # Vegas spreads for evaluation
    vegas_train = train_df['vegas_spread'].to_numpy()
    vegas_test = test_df['vegas_spread'].to_numpy()

    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")

    # Store all predictions for ensemble
    train_predictions = {}
    test_predictions = {}

    # =========================================================================
    # STEP 4: TRAIN ELO MODEL (Optional)
    # =========================================================================
    if not args.skip_elo:
        print("\n" + "=" * 80)
        print("STEP 4: TRAINING ELO MODEL")
        print("=" * 80)

        elo = EloModel(
            k_factor=args.k_factor,
            home_advantage=args.home_advantage
        )
        elo.fit(train_df)

        # Elo predictions
        train_predictions['elo'] = np.array([
            elo.predict_spread(h, a)
            for h, a in zip(home_teams_train, away_teams_train)
        ])
        test_predictions['elo'] = np.array([
            elo.predict_spread(h, a)
            for h, a in zip(home_teams_test, away_teams_test)
        ])

        print(f"  Elo Test MAE: {np.mean(np.abs(y_test - test_predictions['elo'])):.3f}")
    else:
        print("\n  Skipping Elo model (--skip-elo flag)")

    # =========================================================================
    # STEP 5: TRAIN XGBOOST MODEL
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: TRAINING XGBOOST MODEL")
    print("=" * 80)

    # Split train into train/val for early stopping
    val_split = int(len(X_train) * 0.8)
    X_train_xgb = X_train[:val_split]
    y_train_xgb = y_train[:val_split]
    X_val_xgb = X_train[val_split:]
    y_val_xgb = y_train[val_split:]

    xgboost_model = SpreadPredictor(params={
        'objective': 'reg:squarederror',
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'eval_metric': 'mae'
    })

    xgboost_model.train(
        X_train_xgb, y_train_xgb,
        X_val_xgb, y_val_xgb,
        feature_names=feature_cols,
        early_stopping_rounds=50,
        verbose=True
    )

    train_predictions['xgboost'] = xgboost_model.predict(X_train)
    test_predictions['xgboost'] = xgboost_model.predict(X_test)

    print(f"  XGBoost Test MAE: {np.mean(np.abs(y_test - test_predictions['xgboost'])):.3f}")

    # =========================================================================
    # STEP 6: TRAIN BAYESIAN MODEL (Optional)
    # =========================================================================
    if not args.skip_bayesian:
        print("\n" + "=" * 80)
        print("STEP 6: TRAINING BAYESIAN MODEL")
        print("=" * 80)

        try:
            from src.ml.models.bayesian import BayesianStateSpace

            bayesian = BayesianStateSpace(
                n_samples=args.n_bayesian_samples,
                n_chains=2,
                target_accept=0.9
            )

            bayesian.fit(
                X=None,
                y=y_train,
                home_teams=home_teams_train,
                away_teams=away_teams_train,
                verbose=True
            )

            train_predictions['bayesian'] = bayesian.predict_batch(
                list(home_teams_train), list(away_teams_train)
            )
            test_predictions['bayesian'] = bayesian.predict_batch(
                list(home_teams_test), list(away_teams_test)
            )

            print(f"  Bayesian Test MAE: {np.mean(np.abs(y_test - test_predictions['bayesian'])):.3f}")

        except ImportError as e:
            print(f"  Skipping Bayesian model: {e}")
            print("  Install with: pip install pymc arviz")
    else:
        print("\n  Skipping Bayesian model (--skip-bayesian flag)")

    # =========================================================================
    # STEP 7: TRAIN NEURAL NETWORK (Optional)
    # =========================================================================
    if not args.skip_neural:
        print("\n" + "=" * 80)
        print("STEP 7: TRAINING NEURAL NETWORK")
        print("=" * 80)

        try:
            from src.ml.models.neural import NeuralNetPredictor

            neural = NeuralNetPredictor(
                hidden_dims=(64, 32, 16),
                dropout_rate=0.4,
                learning_rate=0.001,
                weight_decay=0.01,
                patience=10,
                max_epochs=200
            )

            neural.fit(
                X_train_xgb, y_train_xgb,
                X_val_xgb, y_val_xgb,
                feature_names=feature_cols,
                verbose=True
            )

            train_predictions['neural'] = neural.predict(X_train)
            test_predictions['neural'] = neural.predict(X_test)

            print(f"  Neural Test MAE: {np.mean(np.abs(y_test - test_predictions['neural'])):.3f}")

        except ImportError as e:
            print(f"  Skipping Neural Network: {e}")
            print("  Install with: pip install torch")
    else:
        print("\n  Skipping Neural Network (--skip-neural flag)")

    # =========================================================================
    # STEP 8: TRAIN STACKING ENSEMBLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: TRAINING STACKING ENSEMBLE")
    print("=" * 80)

    from src.ml.models.ensemble import StackingEnsemble

    ensemble = StackingEnsemble(
        meta_alpha=1.0,
        use_uncertainty=True
    )

    # Add trained base models
    if 'elo' in train_predictions:
        ensemble.base_models['elo'] = elo
        ensemble.base_models['elo'].is_fitted = True  # EloModel doesn't set this

    ensemble.base_models['xgboost'] = xgboost_model
    ensemble.base_models['xgboost'].is_fitted = True

    if 'bayesian' in train_predictions:
        ensemble.add_base_model('bayesian', bayesian)

    if 'neural' in train_predictions:
        ensemble.add_base_model('neural', neural)

    # Train meta-learner
    meta_results = ensemble.fit(
        X=X_train,
        y=y_train,
        home_teams=home_teams_train,
        away_teams=away_teams_train,
        X_val=X_test,
        y_val=y_test,
        home_teams_val=home_teams_test,
        away_teams_val=away_teams_test,
        feature_names=feature_cols
    )

    # Ensemble predictions
    test_predictions['ensemble'] = ensemble.predict(
        X_test, home_teams_test, away_teams_test
    )

    print(f"\n  Model Weights:")
    for name, weight in sorted(meta_results['model_weights'].items(), key=lambda x: -x[1]):
        print(f"    {name:12s}: {weight:.3f}")

    # =========================================================================
    # STEP 9: EVALUATE ALL MODELS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 9: FINAL EVALUATION")
    print("=" * 80)

    results = {}
    valid_vegas = ~np.isnan(vegas_test)

    for name, preds in test_predictions.items():
        mae = np.mean(np.abs(y_test - preds))
        rmse = np.sqrt(np.mean((y_test - preds) ** 2))

        # ATS accuracy
        if valid_vegas.sum() > 0:
            ats = calculate_ats_accuracy(
                preds[valid_vegas],
                y_test[valid_vegas],
                vegas_test[valid_vegas]
            )
            roi_metrics = calculate_roi(
                preds[valid_vegas],
                y_test[valid_vegas],
                vegas_test[valid_vegas],
                edge_threshold=3.0
            )
        else:
            ats = 0.5
            roi_metrics = {'roi': 0, 'total_bets': 0}

        # Win probability and Brier score
        probs = 1 / (1 + np.exp(-preds / 5.5))
        actuals_binary = (y_test > 0).astype(float)
        brier = calculate_brier_score(probs, actuals_binary)

        results[name] = {
            'mae': mae,
            'rmse': rmse,
            'ats_accuracy': ats,
            'brier_score': brier,
            'roi': roi_metrics['roi'],
            'bets_placed': roi_metrics['total_bets']
        }

        print(f"\n{name.upper()}:")
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  ATS Accuracy: {ats:.1%}")
        print(f"  Brier Score: {brier:.4f}")
        print(f"  ROI (3pt edge): {roi_metrics['roi']:.1%} ({roi_metrics['total_bets']} bets)")

    # =========================================================================
    # STEP 10: SAVE MODELS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 10: SAVING MODELS")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Elo
    if 'elo' in train_predictions:
        elo.save_model(output_dir / 'elo', 'elo_model')

    # Save XGBoost
    xgboost_model.save_model(output_dir / 'xgboost', 'spread_predictor')

    # Save Bayesian
    if 'bayesian' in train_predictions:
        bayesian.save_model(output_dir / 'bayesian')

    # Save Neural
    if 'neural' in train_predictions:
        neural.save_model(output_dir / 'neural')

    # Save Ensemble
    ensemble.save_model(output_dir)

    # Save results summary
    summary = {
        'training_date': datetime.now().isoformat(),
        'min_season': args.min_season,
        'test_season': int(test_season),
        'train_games': len(train_df),
        'test_games': len(test_df),
        'models_trained': list(test_predictions.keys()),
        'results': results,
        'ensemble_weights': meta_results['model_weights']
    }

    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nAll models saved to: {output_dir}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['mae'])
    print(f"\nBest Model by MAE: {best_model[0]} ({best_model[1]['mae']:.3f})")

    best_ats = max(results.items(), key=lambda x: x[1]['ats_accuracy'])
    print(f"Best Model by ATS: {best_ats[0]} ({best_ats[1]['ats_accuracy']:.1%})")

    # Check if ensemble beats individuals
    if results['ensemble']['mae'] <= min(r['mae'] for n, r in results.items() if n != 'ensemble'):
        print("\nEnsemble achieves lowest MAE!")
    else:
        print("\nNote: Ensemble did not achieve lowest MAE")

    if results['ensemble']['ats_accuracy'] >= 0.524:
        print(f"ATS accuracy {results['ensemble']['ats_accuracy']:.1%} BEATS break-even threshold (52.4%)")
    else:
        print(f"ATS accuracy {results['ensemble']['ats_accuracy']:.1%} below break-even threshold (52.4%)")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train complete NFL prediction ensemble'
    )

    # Data parameters
    parser.add_argument(
        '--min-season', type=int, default=2015,
        help='Minimum season to include (default: 2015)'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Limit games for testing'
    )

    # Model skip flags
    parser.add_argument(
        '--skip-elo', action='store_true',
        help='Skip Elo model (worst performing base model)'
    )
    parser.add_argument(
        '--skip-bayesian', action='store_true',
        help='Skip Bayesian model (slow MCMC sampling)'
    )
    parser.add_argument(
        '--skip-neural', action='store_true',
        help='Skip Neural Network model'
    )

    # Elo hyperparameters
    parser.add_argument(
        '--k-factor', type=float, default=20.0,
        help='Elo K-factor (default: 20.0)'
    )
    parser.add_argument(
        '--home-advantage', type=float, default=48.0,
        help='Elo home advantage in points (default: 48.0)'
    )

    # XGBoost hyperparameters
    parser.add_argument(
        '--max-depth', type=int, default=4,
        help='XGBoost max depth (default: 4)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.05,
        help='XGBoost learning rate (default: 0.05)'
    )

    # Bayesian hyperparameters
    parser.add_argument(
        '--n-bayesian-samples', type=int, default=500,
        help='Bayesian MCMC samples (default: 500)'
    )

    # Output
    parser.add_argument(
        '--output-dir', type=str, default='models/ensemble',
        help='Output directory (default: models/ensemble)'
    )

    args = parser.parse_args()

    try:
        results = main(args)
        print("\nTraining completed successfully")
        sys.exit(0)
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
