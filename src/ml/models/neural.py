"""
Shallow Neural Network for NFL spread prediction.

Implements per CLAUDE.md constraints:
- Maximum 3 hidden layers
- Dropout: 0.3-0.5
- Early stopping with patience=10
- L2 regularization (weight_decay)
- NO recurrent architectures (LSTM/transformers forbidden)

Uses PyTorch for implementation.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import numpy as np
import json

# PyTorch imports with availability check
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.ml.base import BasePredictor


# Placeholder class when PyTorch is not available
class _SpreadNNPlaceholder:
    """Placeholder when PyTorch is not installed."""
    pass


if TORCH_AVAILABLE:
    class SpreadNN(nn.Module):
        """
        Shallow neural network for spread prediction.

        Architecture:
        - Input: n_features
        - Hidden 1: 64 units, BatchNorm, ReLU, Dropout
        - Hidden 2: 32 units, BatchNorm, ReLU, Dropout
        - Hidden 3: 16 units, BatchNorm, ReLU, Dropout (optional)
        - Output: 1 (spread prediction)

        Respects CLAUDE.md constraint: Maximum 3 hidden layers.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: Tuple[int, ...] = (64, 32, 16),
            dropout_rate: float = 0.4,
            use_batch_norm: bool = True
        ):
            """
            Initialize SpreadNN.

            Args:
                input_dim: Number of input features
                hidden_dims: Sizes of hidden layers (max 3)
                dropout_rate: Dropout probability (0.3-0.5 recommended)
                use_batch_norm: Whether to use batch normalization

            Raises:
                ValueError: If more than 3 hidden layers specified
            """
            super().__init__()

            if len(hidden_dims) > 3:
                raise ValueError(
                    f"CLAUDE.md constraint: Maximum 3 hidden layers, "
                    f"got {len(hidden_dims)}"
                )

            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim

            # Output layer (single value for spread)
            layers.append(nn.Linear(prev_dim, 1))

            self.network = nn.Sequential(*layers)

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            """Forward pass."""
            return self.network(x).squeeze(-1)
else:
    SpreadNN = _SpreadNNPlaceholder


class NeuralNetPredictor(BasePredictor):
    """
    Neural Network predictor for NFL spreads.

    Implements shallow NN per CLAUDE.md constraints:
    - Max 3 hidden layers
    - Dropout 0.3-0.5
    - Early stopping patience=10
    - L2 regularization via weight_decay
    - NO recurrent architectures

    Attributes:
        hidden_dims: Tuple of hidden layer sizes
        dropout_rate: Dropout probability
        learning_rate: Optimizer learning rate
        weight_decay: L2 regularization strength
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        patience: Early stopping patience
    """

    def __init__(
        self,
        model_name: str = "neural_net",
        hidden_dims: Tuple[int, ...] = (64, 32, 16),
        dropout_rate: float = 0.4,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        batch_size: int = 32,
        max_epochs: int = 200,
        patience: int = 10,
        use_batch_norm: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize Neural Network predictor.

        Args:
            model_name: Model identifier
            hidden_dims: Sizes of hidden layers (max 3)
            dropout_rate: Dropout probability (0.3-0.5 recommended)
            learning_rate: Adam optimizer learning rate
            weight_decay: L2 regularization (lambda)
            batch_size: Mini-batch size
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            use_batch_norm: Whether to use batch normalization
            device: 'cuda', 'cpu', or None (auto-detect)

        Raises:
            ImportError: If PyTorch not available
            ValueError: If constraints violated
        """
        super().__init__(model_name=model_name)

        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch not installed. Install with: pip install torch"
            )

        # Validate CLAUDE.md constraints
        if len(hidden_dims) > 3:
            raise ValueError("CLAUDE.md constraint: Maximum 3 hidden layers")
        if not 0.3 <= dropout_rate <= 0.5:
            print(f"Warning: dropout_rate {dropout_rate} outside recommended 0.3-0.5")

        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.use_batch_norm = use_batch_norm

        # Device selection
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)

        self.model: Optional[SpreadNN] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.training_history: List[Dict] = []

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Standardize features to zero mean, unit variance.

        Args:
            X: Input features
            fit: Whether to fit scaler parameters

        Returns:
            Standardized features
        """
        if fit:
            self.scaler_mean = np.nanmean(X, axis=0)
            self.scaler_std = np.nanstd(X, axis=0)
            # Avoid division by zero for constant features
            self.scaler_std[self.scaler_std < 1e-8] = 1.0

        return (X - self.scaler_mean) / self.scaler_std

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train neural network with early stopping.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (spreads)
            X_val: Validation features (recommended for early stopping)
            y_val: Validation targets
            feature_names: Feature names for interpretability
            verbose: Print training progress

        Returns:
            Training history dictionary
        """
        self._validate_training_input(X, y)

        if feature_names:
            self.feature_names = feature_names

        # Standardize features
        X_scaled = self._standardize(X, fit=True)

        # Handle NaN/Inf values
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Validation data setup
        has_validation = X_val is not None and y_val is not None
        if has_validation:
            X_val_scaled = self._standardize(X_val, fit=False)
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Initialize model
        self.model = SpreadNN(
            input_dim=X.shape[1],
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm
        ).to(self.device)

        # Optimizer with L2 regularization (weight_decay)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Loss function
        criterion = nn.MSELoss()

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Training loop with early stopping
        best_val_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0
        self.training_history = []

        if verbose:
            print(f"Training on {self.device}")
            print(f"Architecture: {self.hidden_dims}")
            print(f"Dropout: {self.dropout_rate}, Weight decay: {self.weight_decay}")

        for epoch in range(self.max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches

            # Validation phase
            val_loss = None
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_val_tensor)
                    val_loss = criterion(val_predictions, y_val_tensor).item()

                scheduler.step(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {
                        k: v.cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Record history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

            if verbose and (epoch + 1) % 20 == 0:
                val_str = f", val_loss={val_loss:.4f}" if val_loss else ""
                print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}{val_str}")

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)

        self.is_fitted = True

        # Store metadata
        self.metadata = {
            'train_samples': X.shape[0],
            'val_samples': X_val.shape[0] if has_validation else 0,
            'n_features': X.shape[1],
            'final_train_loss': train_loss,
            'best_val_loss': best_val_loss if has_validation else None,
            'epochs_trained': len(self.training_history),
            'early_stopped': epochs_without_improvement >= self.patience
        }

        if verbose:
            print(f"\nTraining complete!")
            print(f"  Epochs: {self.metadata['epochs_trained']}")
            print(f"  Final train loss: {train_loss:.4f}")
            if has_validation:
                print(f"  Best val loss: {best_val_loss:.4f}")

        return self.metadata

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate point predictions.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predicted spreads
        """
        self._validate_fitted()
        self._validate_input(X)

        # Standardize
        X_scaled = self._standardize(X, fit=False)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

        return predictions

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo dropout for uncertainty estimation.

        Runs multiple forward passes with dropout enabled
        to estimate prediction uncertainty.

        Args:
            X: Features
            n_samples: Number of forward passes with dropout

        Returns:
            (mean_predictions, std_predictions)
        """
        self._validate_fitted()

        X_scaled = self._standardize(X, fit=False)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Enable dropout during inference (MC Dropout)
        self.model.train()  # This enables dropout

        predictions_samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds = self.model(X_tensor).cpu().numpy()
                predictions_samples.append(preds)

        # Return to eval mode
        self.model.eval()

        predictions_samples = np.array(predictions_samples)

        return (
            np.mean(predictions_samples, axis=0),
            np.std(predictions_samples, axis=0)
        )

    def get_prediction_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction uncertainty using MC Dropout.

        Args:
            X: Features

        Returns:
            Standard deviation of predictions
        """
        _, uncertainty = self.predict_with_uncertainty(X, n_samples=50)
        return uncertainty

    def save_model(self, output_dir: Union[str, Path]) -> Path:
        """
        Save model to disk.

        Saves:
        - PyTorch model state dict
        - Metadata and scaler parameters as JSON

        Args:
            output_dir: Directory to save model

        Returns:
            Path to main model file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save PyTorch model
        model_path = output_dir / f"{self.model_name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.network[0].in_features,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm
        }, model_path)

        # Save metadata and scaler
        metadata_path = output_dir / f"{self.model_name}_metadata.json"
        metadata = {
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'hyperparameters': {
                'hidden_dims': list(self.hidden_dims),
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'batch_size': self.batch_size,
                'max_epochs': self.max_epochs,
                'patience': self.patience,
                'use_batch_norm': self.use_batch_norm
            },
            'scaler_mean': self.scaler_mean.tolist() if self.scaler_mean is not None else None,
            'scaler_std': self.scaler_std.tolist() if self.scaler_std is not None else None,
            'training_history': self.training_history,
            'metadata': self.metadata
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Neural network saved to: {model_path}")
        return model_path

    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load model from disk.

        Args:
            model_path: Path to saved model file
        """
        model_path = Path(model_path)

        # Load PyTorch model
        checkpoint = torch.load(model_path, map_location=self.device)

        self.hidden_dims = tuple(checkpoint['hidden_dims'])
        self.dropout_rate = checkpoint['dropout_rate']
        self.use_batch_norm = checkpoint['use_batch_norm']

        self.model = SpreadNN(
            input_dim=checkpoint['input_dim'],
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load metadata
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.feature_names = metadata.get('feature_names', [])
            if metadata.get('scaler_mean'):
                self.scaler_mean = np.array(metadata['scaler_mean'])
            if metadata.get('scaler_std'):
                self.scaler_std = np.array(metadata['scaler_std'])
            self.training_history = metadata.get('training_history', [])
            self.metadata = metadata.get('metadata', {})

        self.is_fitted = True
        print(f"Neural network loaded from: {model_path}")
        print(f"  Architecture: {self.hidden_dims}")
        print(f"  Features: {len(self.feature_names)}")

    def summary(self) -> None:
        """Print model summary."""
        self._validate_fitted()

        print("\n" + "=" * 60)
        print("NEURAL NETWORK MODEL SUMMARY")
        print("=" * 60)
        print(f"Architecture: Input -> {' -> '.join(map(str, self.hidden_dims))} -> 1")
        print(f"Dropout rate: {self.dropout_rate}")
        print(f"Batch normalization: {self.use_batch_norm}")
        print(f"L2 regularization: {self.weight_decay}")
        print(f"Device: {self.device}")

        if self.metadata:
            print(f"\nTraining Info:")
            print(f"  Samples: {self.metadata.get('train_samples', 'N/A')}")
            print(f"  Epochs: {self.metadata.get('epochs_trained', 'N/A')}")
            print(f"  Final loss: {self.metadata.get('final_train_loss', 'N/A'):.4f}")

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nParameters: {total_params:,} (trainable: {trainable_params:,})")
        print("=" * 60 + "\n")
