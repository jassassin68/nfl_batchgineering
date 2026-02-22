"""ML utility functions for feature engineering, evaluation, and validation."""

from src.ml.utils.feature_engineering import (
    select_spread_features,
    create_derived_features,
    create_target_variable,
    prepare_training_data,
)

from src.ml.utils.evaluation import (
    evaluate_spread_model,
    evaluate_by_confidence,
    create_performance_report,
)

from src.ml.utils.validation import (
    walk_forward_cv,
    walk_forward_cv_arrays,
    calculate_cv_metrics,
    calculate_ats_accuracy,
    calculate_roi,
    calculate_brier_score,
    train_test_split_temporal,
)

__all__ = [
    # Feature engineering
    'select_spread_features',
    'create_derived_features',
    'create_target_variable',
    'prepare_training_data',
    # Evaluation
    'evaluate_spread_model',
    'evaluate_by_confidence',
    'create_performance_report',
    # Validation
    'walk_forward_cv',
    'walk_forward_cv_arrays',
    'calculate_cv_metrics',
    'calculate_ats_accuracy',
    'calculate_roi',
    'calculate_brier_score',
    'train_test_split_temporal',
]
