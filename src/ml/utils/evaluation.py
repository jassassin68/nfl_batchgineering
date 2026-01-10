"""
Model evaluation utilities for NFL prediction models.
Provides metrics for regression (MAE, RMSE) and betting performance (ROI, directional accuracy).
"""

import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Optional

# Lazy imports for visualization (optional dependencies)
# These will only be imported when create_performance_report is called
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def evaluate_spread_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for spread predictions.

    Args:
        y_true: Actual spreads (home_score - away_score)
        y_pred: Predicted spreads
        verbose: Whether to print metrics

    Returns:
        Dictionary of evaluation metrics
    """

    # Basic regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Residuals
    residuals = y_true - y_pred
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)

    # Directional accuracy (did we predict the right winner?)
    # Positive spread = home team wins, negative = away team wins
    correct_direction = (np.sign(y_true) == np.sign(y_pred))
    directional_accuracy = np.mean(correct_direction)

    # Betting performance (assuming -110 odds)
    # A bet "wins" if the prediction is within 3 points of the actual spread
    # This is a conservative estimate - in reality, you'd compare to the Vegas line
    spread_margin = 3.0
    correct_beats_spread = np.abs(residuals) < spread_margin

    # ROI calculation: win = +0.91 units (bet $1.10 to win $1), loss = -1 unit
    wins = np.sum(correct_beats_spread)
    losses = len(correct_beats_spread) - wins
    total_return = (wins * 0.91) - losses
    roi = total_return / len(correct_beats_spread)

    # Against the spread (ATS) accuracy
    ats_accuracy = np.mean(correct_beats_spread)

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r_squared': r_squared,
        'mean_residual': mean_residual,
        'std_residual': std_residual,
        'directional_accuracy': directional_accuracy,
        'ats_accuracy': ats_accuracy,
        'betting_roi': roi,
        'wins': int(wins),
        'losses': int(losses),
        'n_samples': len(y_true)
    }

    if verbose:
        print("\n" + "="*60)
        print("SPREAD PREDICTION EVALUATION METRICS")
        print("="*60)
        print(f"Sample Size:              {metrics['n_samples']:,}")
        print(f"\nRegression Metrics:")
        print(f"  Mean Absolute Error:    {metrics['mae']:.2f} points")
        print(f"  Root Mean Squared Error:{metrics['rmse']:.2f} points")
        print(f"  R-Squared:              {metrics['r_squared']:.3f}")
        print(f"  Mean Residual (Bias):   {metrics['mean_residual']:.2f} points")
        print(f"  Std Residual:           {metrics['std_residual']:.2f} points")
        print(f"\nPrediction Accuracy:")
        print(f"  Directional Accuracy:   {metrics['directional_accuracy']:.1%}")
        print(f"  ATS Accuracy (±3pts):   {metrics['ats_accuracy']:.1%}")
        print(f"\nBetting Performance (Hypothetical):")
        print(f"  Wins:                   {metrics['wins']}")
        print(f"  Losses:                 {metrics['losses']}")
        print(f"  ROI:                    {metrics['betting_roi']:.2%}")
        print("="*60 + "\n")

    return metrics


def evaluate_by_confidence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence_threshold: float = 7.0
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance by confidence level.
    High confidence = larger predicted spread (confident in one team winning big).

    Args:
        y_true: Actual spreads
        y_pred: Predicted spreads
        confidence_threshold: Threshold for high confidence (e.g., 7 points)

    Returns:
        Dictionary with 'high_confidence' and 'low_confidence' metrics
    """

    # High confidence predictions (large predicted spreads)
    high_conf_mask = np.abs(y_pred) >= confidence_threshold
    low_conf_mask = ~high_conf_mask

    results = {}

    if np.sum(high_conf_mask) > 0:
        results['high_confidence'] = evaluate_spread_model(
            y_true[high_conf_mask],
            y_pred[high_conf_mask],
            verbose=False
        )
        results['high_confidence']['threshold'] = confidence_threshold
    else:
        results['high_confidence'] = {'n_samples': 0}

    if np.sum(low_conf_mask) > 0:
        results['low_confidence'] = evaluate_spread_model(
            y_true[low_conf_mask],
            y_pred[low_conf_mask],
            verbose=False
        )
    else:
        results['low_confidence'] = {'n_samples': 0}

    print("\n" + "="*60)
    print("PERFORMANCE BY CONFIDENCE LEVEL")
    print("="*60)
    print(f"High Confidence (|spread| >= {confidence_threshold}):")
    if results['high_confidence']['n_samples'] > 0:
        print(f"  Games: {results['high_confidence']['n_samples']}")
        print(f"  MAE: {results['high_confidence']['mae']:.2f}")
        print(f"  Directional Accuracy: {results['high_confidence']['directional_accuracy']:.1%}")
        print(f"  ROI: {results['high_confidence']['betting_roi']:.2%}")
    else:
        print("  No high confidence predictions")

    print(f"\nLow Confidence (|spread| < {confidence_threshold}):")
    if results['low_confidence']['n_samples'] > 0:
        print(f"  Games: {results['low_confidence']['n_samples']}")
        print(f"  MAE: {results['low_confidence']['mae']:.2f}")
        print(f"  Directional Accuracy: {results['low_confidence']['directional_accuracy']:.1%}")
        print(f"  ROI: {results['low_confidence']['betting_roi']:.2%}")
    else:
        print("  No low confidence predictions")
    print("="*60 + "\n")

    return results


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Spread Predictions vs Actual Results"
) -> None:
    """
    Create scatter plot of predictions vs actual values.

    Args:
        y_true: Actual spreads
        y_pred: Predicted spreads
        save_path: Path to save the plot (if None, displays instead)
        title: Plot title
    """

    plt.figure(figsize=(10, 8))

    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Predictions')

    # Add ±3 point bands (typical spread margin)
    plt.plot([min_val, max_val], [min_val + 3, max_val + 3], 'g--', alpha=0.5, linewidth=1)
    plt.plot([min_val, max_val], [min_val - 3, max_val - 3], 'g--', alpha=0.5, linewidth=1)

    plt.xlabel('Actual Spread (Home - Away)', fontsize=12)
    plt.ylabel('Predicted Spread', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add metrics text
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

    metrics_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nDir. Acc: {dir_acc:.1%}'
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_residuals_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of prediction residuals.

    Args:
        y_true: Actual spreads
        y_pred: Predicted spreads
        save_path: Path to save the plot
    """

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].axvline(np.mean(residuals), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(residuals):.2f}')
    axes[0].set_xlabel('Residual (Actual - Predicted)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residuals plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_feature_importance(
    feature_names: list,
    importance_values: np.ndarray,
    top_n: int = 20,
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance from trained model.

    Args:
        feature_names: List of feature names
        importance_values: Importance scores from model
        top_n: Number of top features to show
        save_path: Path to save the plot
    """

    # Create DataFrame and sort
    importance_df = pl.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort('importance', descending=True).head(top_n)

    plt.figure(figsize=(10, 8))

    # Horizontal bar plot
    plt.barh(range(len(importance_df)), importance_df['importance'].to_list())
    plt.yticks(range(len(importance_df)), importance_df['feature'].to_list())
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def create_performance_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: Optional[list] = None,
    feature_importance: Optional[np.ndarray] = None,
    output_dir: str = "ml_models",
    model_name: str = "model",
    metadata: Optional[Dict] = None
) -> str:
    """
    Generate comprehensive performance report with plots and metrics.

    Args:
        y_true: Actual spreads
        y_pred: Predicted spreads
        feature_names: List of feature names (optional, for tree-based models)
        feature_importance: Feature importance scores (optional, for tree-based models)
        output_dir: Directory to save plots
        model_name: Name of the model (for report naming)
        metadata: Additional metadata to include in report

    Returns:
        Path to the generated report
    """

    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib/seaborn not available. Skipping visualization.")
        print("Install with: pip install matplotlib seaborn")
        # Just return metrics without plots
        metrics = evaluate_spread_model(y_true, y_pred, verbose=True)
        return None

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Calculate metrics
    metrics = evaluate_spread_model(y_true, y_pred, verbose=True)

    # Evaluate by confidence
    confidence_metrics = evaluate_by_confidence(y_true, y_pred)

    # Generate plots
    plot_predictions_vs_actual(
        y_true, y_pred,
        save_path=f"{output_dir}/predictions_vs_actual.png"
    )

    plot_residuals_distribution(
        y_true, y_pred,
        save_path=f"{output_dir}/residuals_distribution.png"
    )

    # Only plot feature importance if provided (tree-based models)
    if feature_names is not None and feature_importance is not None:
        plot_feature_importance(
            feature_names, feature_importance,
            save_path=f"{output_dir}/feature_importance.png"
        )

    print(f"\n✅ Performance report generated in {output_dir}/")

    return f"{output_dir}/predictions_vs_actual.png"
