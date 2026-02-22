"""
Abstract base class for NFL prediction models.

All models in the ensemble must implement this interface to ensure
consistent behavior for training, prediction, and serialization.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
import numpy as np


class BasePredictor(ABC):
    """
    Abstract base class for all NFL prediction models.

    All models in the ensemble must implement this interface.
    Provides common functionality for validation, probability conversion,
    and uncertainty estimation.

    Attributes:
        model_name: Unique identifier for model serialization
        is_fitted: Whether the model has been trained
        feature_names: Names of input features
        metadata: Additional model metadata
    """

    def __init__(self, model_name: str = "base_model"):
        """
        Initialize base predictor.

        Args:
            model_name: Unique identifier for model serialization
        """
        self.model_name = model_name
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model on training data.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,) - spread or total
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: Names of features (for interpretability)
            **kwargs: Model-specific training parameters

        Returns:
            Dictionary containing training history/metrics

        Raises:
            ValueError: If X and y have incompatible shapes
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate point spread predictions.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predicted spreads (n_samples,)
            Positive = home favored, Negative = away favored

        Raises:
            ValueError: If model not fitted
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate win probability predictions.

        Default implementation converts spread to probability using
        logistic function. Models can override for calibrated probabilities.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Home win probabilities (n_samples,) in [0, 1]
        """
        spreads = self.predict(X)
        # Convert spread to probability using logistic function
        # NFL typical std dev is ~13.5 points, using 5.5 as scaling factor
        # gives reasonable probability mapping
        return 1 / (1 + np.exp(-spreads / 5.5))

    @abstractmethod
    def save_model(self, output_dir: Union[str, Path]) -> Path:
        """
        Serialize model to disk.

        Args:
            output_dir: Directory to save model artifacts

        Returns:
            Path to main model file
        """
        pass

    @abstractmethod
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load model from disk.

        Args:
            model_path: Path to saved model file
        """
        pass

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores,
            or None if model doesn't support feature importance
        """
        return None

    def get_prediction_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction uncertainty/std dev.

        Default returns constant estimate based on typical NFL variance.
        Bayesian models should override with posterior std dev.
        Neural networks can use MC Dropout.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Standard deviation of predictions (n_samples,)
        """
        return np.full(len(X), 13.5)  # NFL typical game std dev

    def _validate_fitted(self) -> None:
        """
        Raise error if model not fitted.

        Raises:
            ValueError: If model has not been trained
        """
        if not self.is_fitted:
            raise ValueError(
                f"{self.__class__.__name__} not fitted. "
                "Call fit() before predict()."
            )

    def _validate_input(self, X: np.ndarray) -> None:
        """
        Validate input array shape and type.

        Args:
            X: Input feature array

        Raises:
            ValueError: If X is not 2D or has wrong number of features
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if self.feature_names and X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"expected {len(self.feature_names)}"
            )

    def _validate_training_input(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        """
        Validate training input arrays.

        Args:
            X: Feature array
            y: Target array

        Raises:
            ValueError: If shapes are incompatible
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples, y has {y.shape[0]}"
            )
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y.ndim}D")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"is_fitted={self.is_fitted})"
        )
