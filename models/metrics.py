"""
Evaluation metrics for the workout recommendation system.
Includes accuracy, precision, recall, F1, and recall@k.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score
    """
    return accuracy_score(y_true, y_pred)


def calculate_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> float:
    """
    Calculate precision.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy
        
    Returns:
        Precision score
    """
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def calculate_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> float:
    """
    Calculate recall.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy
        
    Returns:
        Recall score
    """
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def calculate_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> float:
    """
    Calculate F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy
        
    Returns:
        F1 score
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def calculate_recall_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: int = 3
) -> float:
    """
    Calculate Recall@K for recommendation systems.
    Measures if the true class is in the top-k predictions.
    
    Args:
        y_true: True labels (1D array)
        y_prob: Predicted probabilities (2D array, shape [n_samples, n_classes])
        k: Number of top predictions to consider
        
    Returns:
        Recall@K score
    """
    # Get top-k predictions for each sample
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    
    # Check if true label is in top-k
    hits = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            hits += 1
    
    return hits / len(y_true)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    k_values: list = [1, 3, 5]
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for recall@k)
        k_values: List of k values for recall@k
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'accuracy': calculate_accuracy(y_true, y_pred),
        'precision': calculate_precision(y_true, y_pred),
        'recall': calculate_recall(y_true, y_pred),
        'f1_score': calculate_f1(y_true, y_pred)
    }
    
    # Add recall@k metrics if probabilities provided
    if y_prob is not None:
        for k in k_values:
            metrics[f'recall@{k}'] = calculate_recall_at_k(y_true, y_prob, k)
    
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\nEvaluation Metrics:")
    print("=" * 50)
    for metric_name, value in metrics.items():
        print(f"  {metric_name:15s}: {value:.4f}")


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None
) -> None:
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names
    """
    print("\nClassification Report:")
    print("=" * 50)
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    ))


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names
        save_path: Path to save figure
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def compare_metrics(
    metrics1: Dict[str, float],
    metrics2: Dict[str, float],
    label1: str = "Model 1",
    label2: str = "Model 2"
) -> None:
    """
    Compare metrics between two models.
    
    Args:
        metrics1: First model's metrics
        metrics2: Second model's metrics
        label1: Label for first model
        label2: Label for second model
    """
    print(f"\nMetrics Comparison: {label1} vs {label2}")
    print("=" * 70)
    print(f"{'Metric':<20} {label1:>15} {label2:>15} {'Difference':>15}")
    print("-" * 70)
    
    for metric_name in metrics1.keys():
        if metric_name in metrics2:
            val1 = metrics1[metric_name]
            val2 = metrics2[metric_name]
            diff = val2 - val1
            diff_str = f"{diff:+.4f}"
            print(f"{metric_name:<20} {val1:>15.4f} {val2:>15.4f} {diff_str:>15}")


def plot_metric_comparison(
    metrics1: Dict[str, float],
    metrics2: Dict[str, float],
    label1: str = "Centralized",
    label2: str = "Federated",
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot bar chart comparing metrics.
    
    Args:
        metrics1: First model's metrics
        metrics2: Second model's metrics
        label1: Label for first model
        label2: Label for second model
        save_path: Path to save figure
        figsize: Figure size
    """
    # Get common metrics
    common_metrics = set(metrics1.keys()) & set(metrics2.keys())
    metric_names = sorted(common_metrics)
    
    values1 = [metrics1[m] for m in metric_names]
    values2 = [metrics2[m] for m in metric_names]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - width/2, values1, width, label=label1, alpha=0.8)
    bars2 = ax.bar(x + width/2, values2, width, label=label2, alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.close()


class MetricsTracker:
    """Track metrics across training rounds."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.history = {
            'round': [],
            'accuracy': [],
            'loss': []
        }
    
    def add_round(self, round_num: int, accuracy: float, loss: float = None) -> None:
        """
        Add metrics for a round.
        
        Args:
            round_num: Round number
            accuracy: Accuracy for this round
            loss: Optional loss value
        """
        self.history['round'].append(round_num)
        self.history['accuracy'].append(accuracy)
        if loss is not None:
            self.history['loss'].append(loss)
    
    def plot_history(
        self,
        save_path: str = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot training history.
        
        Args:
            save_path: Path to save figure
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot accuracy
        axes[0].plot(self.history['round'], self.history['accuracy'], marker='o')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy over Rounds')
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss if available
        if self.history['loss']:
            axes[1].plot(self.history['round'], self.history['loss'], marker='o', color='orange')
            axes[1].set_xlabel('Round')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Loss over Rounds')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.close()


def main():
    """Test metrics functions."""
    print("Testing metrics calculations...")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 100
    n_classes = 6
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.rand(n_samples, n_classes)
    
    # Normalize probabilities
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    # Calculate metrics
    metrics = calculate_all_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics)
    
    # Test classification report
    class_names = ['cardio', 'strength', 'hiit', 'yoga', 'pilates', 'cycling']
    print_classification_report(y_true, y_pred, class_names)
    
    print("\nMetrics test complete!")


if __name__ == '__main__':
    main()