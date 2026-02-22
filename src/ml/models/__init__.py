"""ML model classes for NFL predictions."""

from src.ml.models.elo_model import EloModel
from src.ml.models.spread_predictor import SpreadPredictor

# Optional imports for advanced models (require additional dependencies)
try:
    from src.ml.models.bayesian import BayesianStateSpace
except ImportError:
    BayesianStateSpace = None

try:
    from src.ml.models.neural import NeuralNetPredictor
except ImportError:
    NeuralNetPredictor = None

try:
    from src.ml.models.ensemble import StackingEnsemble
except ImportError:
    StackingEnsemble = None

__all__ = [
    'EloModel',
    'SpreadPredictor',
    'BayesianStateSpace',
    'NeuralNetPredictor',
    'StackingEnsemble',
]
