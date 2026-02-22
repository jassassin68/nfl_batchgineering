"""
Spread Predictor Model Class
Encapsulates XGBoost model for NFL spread predictions with prediction confidence
"""

import xgboost as xgb
import numpy as np
import polars as pl
import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime


class SpreadPredictor:
    """
    XGBoost-based model for predicting NFL game spreads.
    Provides methods for training, prediction, and model persistence.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        params: Optional[Dict] = None
    ):
        """
        Initialize the Spread Predictor.

        Args:
            model_path: Path to saved model file (if loading existing model)
            params: XGBoost hyperparameters (if training new model)
        """
        self.model: Optional[xgb.Booster] = None
        self.feature_names: List[str] = []
        self.metadata: Dict = {}

        if model_path:
            self.load_model(model_path)
        elif params:
            self.params = params
        else:
            # Default hyperparameters
            self.params = self._get_default_params()

    @staticmethod
    def _get_default_params() -> Dict:
        """Returns default XGBoost hyperparameters for spread prediction."""
        return {
            'objective': 'reg:squarederror',
            'max_depth': 5,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'random_state': 42,
            'tree_method': 'hist',  # Faster training
            'eval_metric': 'mae'
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ) -> Dict:
        """
        Train the XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels (spreads)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: List of feature names
            early_stopping_rounds: Stop if validation doesn't improve for N rounds
            verbose: Whether to print training progress

        Returns:
            Dictionary with training history
        """

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)

        # Prepare eval list
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, 'validation'))

        # Extract n_estimators from params
        num_boost_round = self.params.pop('n_estimators', 300)

        # Train model
        evals_result = {}
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
            evals_result=evals_result,
            verbose_eval=verbose
        )

        # Store metadata
        self.metadata = {
            'train_date': datetime.now().isoformat(),
            'n_train_samples': X_train.shape[0],
            'n_features': X_train.shape[1],
            'n_val_samples': X_val.shape[0] if X_val is not None else 0,
            'feature_names': self.feature_names,
            'params': self.params,
            'num_boost_round': num_boost_round,
            'best_iteration': self.model.best_iteration if X_val is not None else num_boost_round
        }

        if verbose:
            print(f"\n✅ Model trained successfully!")
            print(f"   Training samples: {self.metadata['n_train_samples']:,}")
            print(f"   Validation samples: {self.metadata['n_val_samples']:,}")
            print(f"   Features: {self.metadata['n_features']}")
            print(f"   Best iteration: {self.metadata['best_iteration']}")

        return evals_result

    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = False
    ) -> np.ndarray:
        """
        Make spread predictions.

        Args:
            X: Feature array
            return_confidence: If True, also return prediction confidence

        Returns:
            Array of predicted spreads (and optionally confidence scores)
        """

        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")

        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        predictions = self.model.predict(dmatrix)

        if return_confidence:
            # Calculate confidence based on magnitude of prediction
            # Larger predicted spreads = more confident
            confidence = np.abs(predictions) / 14.0  # Normalize by typical max spread
            confidence = np.clip(confidence, 0.0, 1.0)
            return predictions, confidence

        return predictions

    def predict_games(
        self,
        games_df: pl.DataFrame,
        feature_columns: List[str]
    ) -> pl.DataFrame:
        """
        Predict spreads for a DataFrame of games.

        Args:
            games_df: DataFrame with game features
            feature_columns: List of column names to use as features

        Returns:
            DataFrame with predictions added
        """

        X = games_df.select(feature_columns).to_numpy()
        predictions, confidence = self.predict(X, return_confidence=True)

        # Add predictions to DataFrame
        result_df = games_df.with_columns([
            pl.Series('predicted_spread', predictions),
            pl.Series('prediction_confidence', confidence)
        ])

        return result_df

    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')

        Returns:
            Dictionary mapping feature names to importance scores
        """

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        importance_dict = self.model.get_score(importance_type=importance_type)

        # Map to feature names
        return {name: importance_dict.get(name, 0.0) for name in self.feature_names}

    def save_model(
        self,
        model_dir: str,
        model_name: str = "spread_predictor"
    ) -> None:
        """
        Save model and metadata to disk.

        Args:
            model_dir: Directory to save model files
            model_name: Base name for model files
        """

        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        model_path = f"{model_dir}/{model_name}.json"
        self.model.save_model(model_path)

        # Save metadata
        metadata_path = f"{model_dir}/{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Save feature names
        features_path = f"{model_dir}/{model_name}_features.json"
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)

        print(f"\n✅ Model saved successfully!")
        print(f"   Model: {model_path}")
        print(f"   Metadata: {metadata_path}")
        print(f"   Features: {features_path}")

    def load_model(
        self,
        model_path: str
    ) -> None:
        """
        Load model from disk.

        Args:
            model_path: Path to saved model JSON file
        """

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

        # Load metadata
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

        # Load feature names
        features_path = model_path.parent / f"{model_path.stem}_features.json"
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)

        print(f"✅ Model loaded from {model_path}")
        print(f"   Trained: {self.metadata.get('train_date', 'Unknown')}")
        print(f"   Features: {len(self.feature_names)}")

    def summary(self) -> None:
        """Print model summary information."""

        if self.model is None:
            print("⚠️  No model loaded or trained")
            return

        print("\n" + "="*60)
        print("SPREAD PREDICTOR MODEL SUMMARY")
        print("="*60)
        print(f"Trained: {self.metadata.get('train_date', 'Unknown')}")
        print(f"Training samples: {self.metadata.get('n_train_samples', 0):,}")
        print(f"Validation samples: {self.metadata.get('n_val_samples', 0):,}")
        print(f"Features: {self.metadata.get('n_features', 0)}")
        print(f"Best iteration: {self.metadata.get('best_iteration', 0)}")
        print("\nTop 10 Features by Importance:")

        importance = self.get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        for i, (feature, score) in enumerate(sorted_features, 1):
            print(f"  {i:2d}. {feature:30s} {score:8.2f}")

        print("="*60 + "\n")
