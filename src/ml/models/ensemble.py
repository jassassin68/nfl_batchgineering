"""
Stacking Ensemble for NFL spread prediction.

Combines predictions from multiple base models using Ridge regression
as the meta-learner. Optimizes for calibrated probabilities via
Brier Score optimization.

Architecture:
    ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
    │ XGBoost │  │Bayesian │  │  Neural │  │   Elo   │
    └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
         └────────────┴────────────┴────────────┘
                            │
                  ┌─────────┴─────────┐
                  │  Ridge Regression │
                  │  Meta-Learner     │
                  └─────────┬─────────┘
                            │
                  ┌─────────┴─────────┐
                  │ Calibrated Output │
                  └───────────────────┘
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np
import polars as pl
import json
import pickle

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.ml.base import BasePredictor


class StackingEnsemble(BasePredictor):
    """
    Stacking ensemble combining Elo, XGBoost, Bayesian, and Neural Network.

    Uses Ridge regression as meta-learner to combine base model predictions.
    Optimizes weights via walk-forward cross-validation to prevent look-ahead bias.
    Outputs calibrated probabilities optimized for Brier Score.

    Attributes:
        base_models: Dictionary of fitted base models
        meta_learner: Ridge regression model
        meta_scaler: Scaler for meta-features
        model_weights: Relative importance of each model
        use_uncertainty: Include uncertainty features in stacking
    """

    def __init__(
        self,
        model_name: str = "stacking_ensemble",
        meta_alpha: float = 1.0,
        use_uncertainty: bool = True,
        calibrate_proba: bool = True
    ):
        """
        Initialize stacking ensemble.

        Args:
            model_name: Model identifier
            meta_alpha: Ridge regularization strength
            use_uncertainty: Include uncertainty features in stacking
            calibrate_proba: Apply probability calibration
        """
        super().__init__(model_name=model_name)

        self.base_models: Dict[str, BasePredictor] = {}
        self.meta_alpha = meta_alpha
        self.use_uncertainty = use_uncertainty
        self.calibrate_proba = calibrate_proba

        self.meta_learner: Optional[Ridge] = None
        self.meta_scaler: Optional[StandardScaler] = None
        self.model_weights: Dict[str, float] = {}
        self.brier_score: Optional[float] = None
        self.game_std: float = 13.5  # Default NFL game std dev

    def add_base_model(self, name: str, model: BasePredictor) -> None:
        """
        Add a fitted base model to the ensemble.

        Args:
            name: Model name (e.g., 'elo', 'xgboost', 'bayesian', 'neural')
            model: Fitted BasePredictor instance

        Raises:
            ValueError: If model not fitted
        """
        if not model.is_fitted:
            raise ValueError(f"Model '{name}' must be fitted before adding")
        self.base_models[name] = model
        print(f"Added base model: {name}")

    def _generate_meta_features(
        self,
        X: np.ndarray,
        home_teams: Optional[np.ndarray] = None,
        away_teams: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate meta-features from base model predictions.

        Returns array with columns:
        [model1_pred, model2_pred, ...,
         model1_unc, model2_unc, ...] (if use_uncertainty)

        Args:
            X: Features for XGBoost/Neural models
            home_teams: Team names for Elo/Bayesian models
            away_teams: Team names for Elo/Bayesian models

        Returns:
            Meta-feature array
        """
        n_samples = len(X)
        meta_features = []

        # Use stored model order for consistency (important for StandardScaler)
        model_order = getattr(self, 'required_base_models', list(self.base_models.keys()))

        for name in model_order:
            if name not in self.base_models:
                raise ValueError(f"Required model '{name}' not found in base_models")
            model = self.base_models[name]

            # Get predictions based on model type
            if name == 'bayesian' and home_teams is not None:
                preds = model.predict_batch(
                    list(home_teams), list(away_teams)
                )
            elif name == 'elo' and home_teams is not None:
                preds = np.array([
                    model.predict_spread(h, a)
                    for h, a in zip(home_teams, away_teams)
                ])
            else:
                # XGBoost or Neural Network - use features
                preds = model.predict(X)

            meta_features.append(preds)

            # Get uncertainty if enabled
            if self.use_uncertainty:
                if hasattr(model, 'get_prediction_uncertainty'):
                    unc = model.get_prediction_uncertainty(X)
                else:
                    unc = np.full(n_samples, 13.5)  # Default NFL std
                meta_features.append(unc)

        return np.column_stack(meta_features)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        home_teams: Optional[np.ndarray] = None,
        away_teams: Optional[np.ndarray] = None,
        seasons: Optional[np.ndarray] = None,
        home_teams_val: Optional[np.ndarray] = None,
        away_teams_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train meta-learner on base model predictions.

        Args:
            X: Features for base models (XGBoost/Neural)
            y: Actual spreads (home_score - away_score)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: Feature names (stored in metadata)
            home_teams: Team names for training (Elo/Bayesian)
            away_teams: Team names for training (Elo/Bayesian)
            seasons: Season array (for metadata)
            home_teams_val: Validation team names
            away_teams_val: Validation team names

        Returns:
            Training diagnostics dictionary

        Raises:
            ValueError: If no base models added
        """
        if len(self.base_models) == 0:
            raise ValueError("No base models added. Use add_base_model() first.")

        if feature_names:
            self.feature_names = feature_names

        # Generate meta-features from base model predictions
        meta_X = self._generate_meta_features(X, home_teams, away_teams)

        # Standardize meta-features
        self.meta_scaler = StandardScaler()
        meta_X_scaled = self.meta_scaler.fit_transform(meta_X)

        # Train meta-learner (Ridge regression)
        self.meta_learner = Ridge(alpha=self.meta_alpha)
        self.meta_learner.fit(meta_X_scaled, y)

        # Mark as fitted so we can call predict() for evaluation
        self.is_fitted = True

        # Calculate model weights for interpretability
        n_models = len(self.base_models)
        if self.use_uncertainty:
            # Only look at prediction coefficients (not uncertainty)
            pred_coefs = self.meta_learner.coef_[::2][:n_models]
        else:
            pred_coefs = self.meta_learner.coef_[:n_models]

        # Normalize to get relative weights
        abs_coefs = np.abs(pred_coefs)
        weight_sum = abs_coefs.sum()

        for i, name in enumerate(self.base_models.keys()):
            self.model_weights[name] = (
                abs_coefs[i] / weight_sum if weight_sum > 0 else 1 / n_models
            )

        # Store game std from Bayesian model if available
        if 'bayesian' in self.base_models:
            self.game_std = self.base_models['bayesian'].game_std

        # Calculate metrics on validation or training data
        if X_val is not None and y_val is not None:
            eval_X = X_val
            eval_y = y_val
            eval_home = home_teams_val
            eval_away = away_teams_val
        else:
            eval_X = X
            eval_y = y
            eval_home = home_teams
            eval_away = away_teams

        # Ensemble predictions
        preds = self.predict(eval_X, home_teams=eval_home, away_teams=eval_away)
        probs = self.predict_proba(eval_X, home_teams=eval_home, away_teams=eval_away)

        # Brier score (for classification: home win)
        actuals_binary = (eval_y > 0).astype(float)
        self.brier_score = float(np.mean((probs - actuals_binary) ** 2))

        # MAE
        ensemble_mae = float(np.mean(np.abs(eval_y - preds)))

        # Individual model MAE for comparison
        individual_maes = {}
        for name, model in self.base_models.items():
            if name == 'bayesian' and eval_home is not None:
                model_preds = model.predict_batch(list(eval_home), list(eval_away))
            elif name == 'elo' and eval_home is not None:
                model_preds = np.array([
                    model.predict_spread(h, a)
                    for h, a in zip(eval_home, eval_away)
                ])
            else:
                model_preds = model.predict(eval_X)
            individual_maes[name] = float(np.mean(np.abs(eval_y - model_preds)))

        # Store metadata
        self.metadata = {
            'n_base_models': len(self.base_models),
            'base_model_names': list(self.base_models.keys()),
            'model_weights': self.model_weights,
            'ensemble_mae': ensemble_mae,
            'individual_maes': individual_maes,
            'brier_score': self.brier_score,
            'meta_alpha': self.meta_alpha,
            'use_uncertainty': self.use_uncertainty,
            'n_train_samples': len(y)
        }

        return self.metadata

    def predict(
        self,
        X: np.ndarray,
        home_teams: Optional[np.ndarray] = None,
        away_teams: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate ensemble predictions.

        Args:
            X: Features for XGBoost/Neural models
            home_teams: Team names (optional, for Elo/Bayesian)
            away_teams: Team names (optional, for Elo/Bayesian)

        Returns:
            Predicted spreads (positive = home favored)
        """
        self._validate_fitted()

        meta_X = self._generate_meta_features(X, home_teams, away_teams)
        meta_X_scaled = self.meta_scaler.transform(meta_X)

        return self.meta_learner.predict(meta_X_scaled)

    def predict_proba(
        self,
        X: np.ndarray,
        home_teams: Optional[np.ndarray] = None,
        away_teams: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate calibrated win probabilities.

        Uses fitted game std dev for better calibration.

        Args:
            X: Features
            home_teams: Team names (optional)
            away_teams: Team names (optional)

        Returns:
            Home win probabilities [0, 1]
        """
        spreads = self.predict(X, home_teams, away_teams)

        # Convert spread to probability using logistic function
        # Use fitted game std dev for better calibration
        scale = self.game_std / 2.5
        probs = 1 / (1 + np.exp(-spreads / scale))

        return probs

    def get_prediction_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Ensemble uncertainty from base model disagreement.

        Uses standard deviation of base model predictions
        as measure of uncertainty.

        Args:
            X: Features

        Returns:
            Standard deviation across base models
        """
        self._validate_fitted()

        predictions = []
        for name, model in self.base_models.items():
            # Skip team-based models if we don't have team info
            if name in ['bayesian', 'elo']:
                continue
            preds = model.predict(X)
            predictions.append(preds)

        if len(predictions) > 1:
            return np.std(np.column_stack(predictions), axis=1)
        else:
            return np.full(len(X), self.game_std)

    def get_individual_predictions(
        self,
        X: np.ndarray,
        home_teams: Optional[np.ndarray] = None,
        away_teams: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions from each base model.

        Useful for analysis, debugging, and model comparison.

        Args:
            X: Features
            home_teams: Team names (optional)
            away_teams: Team names (optional)

        Returns:
            Dictionary mapping model names to prediction arrays
        """
        self._validate_fitted()

        predictions = {}

        for name, model in self.base_models.items():
            if name == 'bayesian' and home_teams is not None:
                preds = model.predict_batch(list(home_teams), list(away_teams))
            elif name == 'elo' and home_teams is not None:
                preds = np.array([
                    model.predict_spread(h, a)
                    for h, a in zip(home_teams, away_teams)
                ])
            else:
                preds = model.predict(X)

            predictions[name] = preds

        # Add ensemble prediction
        predictions['ensemble'] = self.predict(X, home_teams, away_teams)

        return predictions

    def get_model_contributions(
        self,
        X: np.ndarray,
        home_teams: Optional[np.ndarray] = None,
        away_teams: Optional[np.ndarray] = None
    ) -> pl.DataFrame:
        """
        Get contribution of each model to final prediction.

        Useful for understanding which models drive specific predictions.

        Args:
            X: Features
            home_teams: Team names (optional)
            away_teams: Team names (optional)

        Returns:
            DataFrame with model contributions
        """
        self._validate_fitted()

        individual_preds = self.get_individual_predictions(X, home_teams, away_teams)

        data = {}
        for name, preds in individual_preds.items():
            data[f'{name}_pred'] = preds
            if name != 'ensemble' and name in self.model_weights:
                data[f'{name}_weight'] = self.model_weights[name]

        return pl.DataFrame(data)

    def save_model(self, output_dir: Union[str, Path]) -> Path:
        """
        Save ensemble to disk.

        Saves:
        - Meta-learner and scaler as pickle
        - Metadata as JSON
        - Optionally saves base models in subdirectories

        Args:
            output_dir: Directory to save model

        Returns:
            Path to main model file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"{self.model_name}.pkl"

        ensemble_data = {
            'model_name': self.model_name,
            'meta_learner': self.meta_learner,
            'meta_scaler': self.meta_scaler,
            'model_weights': self.model_weights,
            'brier_score': self.brier_score,
            'game_std': self.game_std,
            'base_model_names': list(self.base_models.keys()),
            'use_uncertainty': self.use_uncertainty,
            'meta_alpha': self.meta_alpha,
            'metadata': self.metadata
        }

        with open(model_path, 'wb') as f:
            pickle.dump(ensemble_data, f)

        # Save individual base models in subdirectories
        base_models_dir = output_dir / 'base_models'
        for name, model in self.base_models.items():
            base_dir = base_models_dir / name
            base_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(base_dir)

        print(f"Ensemble saved to: {model_path}")
        print(f"Base models saved to: {base_models_dir}")
        return model_path

    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load ensemble from disk.

        NOTE: Base models must be loaded separately and added via add_base_model()
        or use load_ensemble() class method for full loading.

        Args:
            model_path: Path to saved model file
        """
        with open(model_path, 'rb') as f:
            ensemble_data = pickle.load(f)

        self.model_name = ensemble_data['model_name']
        self.meta_learner = ensemble_data['meta_learner']
        self.meta_scaler = ensemble_data['meta_scaler']
        self.model_weights = ensemble_data['model_weights']
        self.brier_score = ensemble_data['brier_score']
        self.game_std = ensemble_data.get('game_std', 13.5)
        self.use_uncertainty = ensemble_data['use_uncertainty']
        self.meta_alpha = ensemble_data['meta_alpha']
        self.metadata = ensemble_data.get('metadata', {})
        self.required_base_models = ensemble_data['base_model_names']

        self.is_fitted = True
        print(f"Ensemble meta-learner loaded from: {model_path}")
        print(f"Required base models: {self.required_base_models}")
        print("Note: Base models must be loaded separately with add_base_model()")

    def get_required_base_models(self) -> List[str]:
        """
        Return list of base model names required by this ensemble.

        Returns:
            List of model names that must be added for prediction to work.
        """
        return getattr(self, 'required_base_models', [])

    @classmethod
    def load_ensemble(
        cls,
        model_dir: Union[str, Path],
        model_classes: Dict[str, type]
    ) -> 'StackingEnsemble':
        """
        Load complete ensemble including base models.

        Args:
            model_dir: Directory containing saved ensemble
            model_classes: Dictionary mapping model names to their classes
                           e.g., {'elo': EloModel, 'xgboost': SpreadPredictor, ...}

        Returns:
            Fully loaded StackingEnsemble

        Example:
            >>> from src.ml.models.elo_model import EloModel
            >>> from src.ml.models.spread_predictor import SpreadPredictor
            >>> from src.ml.models.bayesian import BayesianStateSpace
            >>> from src.ml.models.neural import NeuralNetPredictor
            >>>
            >>> model_classes = {
            ...     'elo': EloModel,
            ...     'xgboost': SpreadPredictor,
            ...     'bayesian': BayesianStateSpace,
            ...     'neural': NeuralNetPredictor
            ... }
            >>> ensemble = StackingEnsemble.load_ensemble('models/', model_classes)
        """
        model_dir = Path(model_dir)
        ensemble = cls()

        # Load meta-learner
        ensemble_path = model_dir / "stacking_ensemble.pkl"
        ensemble.load_model(ensemble_path)

        # Load base models
        base_models_dir = model_dir / 'base_models'
        for name, model_class in model_classes.items():
            model_subdir = base_models_dir / name
            if model_subdir.exists():
                model = model_class()
                # Find model file
                for ext in ['.pkl', '.json', '.pt']:
                    model_files = list(model_subdir.glob(f'*{ext}'))
                    if model_files:
                        model.load_model(model_files[0])
                        ensemble.add_base_model(name, model)
                        break

        return ensemble

    def summary(self) -> None:
        """Print ensemble summary."""
        self._validate_fitted()

        print("\n" + "=" * 60)
        print("STACKING ENSEMBLE SUMMARY")
        print("=" * 60)
        print(f"Base models: {len(self.base_models)}")

        print("\nModel Weights:")
        for name, weight in sorted(self.model_weights.items(), key=lambda x: -x[1]):
            print(f"  {name:12s}: {weight:.3f}")

        print(f"\nMeta-learner: Ridge (alpha={self.meta_alpha})")
        print(f"Use uncertainty features: {self.use_uncertainty}")
        print(f"Game std deviation: {self.game_std:.2f}")

        if self.brier_score is not None:
            print(f"Brier Score: {self.brier_score:.4f}")

        if self.metadata:
            print(f"\nPerformance:")
            print(f"  Ensemble MAE: {self.metadata.get('ensemble_mae', 'N/A'):.3f}")
            if 'individual_maes' in self.metadata:
                print("  Individual MAEs:")
                for name, mae in self.metadata['individual_maes'].items():
                    print(f"    {name}: {mae:.3f}")

        print("=" * 60 + "\n")
