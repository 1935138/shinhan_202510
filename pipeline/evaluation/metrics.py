"""Evaluation Metrics

This module contains functions for model evaluation and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate various classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)

    return metrics


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = 'ROC Curve',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        ax: Matplotlib axes (optional)

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random classifier', linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    return ax


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = 'Precision-Recall Curve',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot Precision-Recall curve.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        ax: Matplotlib axes (optional)

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)

    ax.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.4f})', linewidth=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    return ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = 'Confusion Matrix',
    ax: Optional[plt.Axes] = None,
    normalize: bool = False
) -> plt.Axes:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        ax: Matplotlib axes (optional)
        normalize: Whether to normalize the confusion matrix

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)

    return ax


class ModelEvaluator:
    """
    Model evaluator class for comprehensive model evaluation.
    """

    def __init__(self, model: Any, model_name: str = 'Model'):
        """
        Initialize ModelEvaluator.

        Args:
            model: Trained model
            model_name: Name of the model
        """
        self.model = model
        self.model_name = model_name

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = 'Dataset'
    ) -> Dict[str, Any]:
        """
        Evaluate model on given dataset.

        Args:
            X: Features
            y: True labels
            dataset_name: Name of the dataset (e.g., 'Train', 'Validation', 'Test')

        Returns:
            Dictionary containing evaluation results
        """
        # Predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        # Metrics
        metrics = calculate_metrics(y, y_pred, y_pred_proba)

        # Classification report
        report = classification_report(y, y_pred, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)

        results = {
            'dataset': dataset_name,
            'model': self.model_name,
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        return results

    def print_evaluation(self, results: Dict[str, Any]):
        """
        Print evaluation results.

        Args:
            results: Results from evaluate() method
        """
        print("=" * 80)
        print(f"{results['model']} - {results['dataset']} Evaluation")
        print("=" * 80)

        print("\nMetrics:")
        for metric, value in results['metrics'].items():
            print(f"  {metric}: {value:.4f}")

        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])

        print("\nClassification Report:")
        report_df = pd.DataFrame(results['classification_report']).transpose()
        print(report_df.round(4))

    def plot_evaluation(
        self,
        results: Dict[str, Any],
        y_true: pd.Series
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot evaluation results (ROC, PR curve, Confusion matrix).

        Args:
            results: Results from evaluate() method
            y_true: True labels

        Returns:
            Matplotlib figure and axes
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # ROC Curve
        plot_roc_curve(
            y_true,
            results['y_pred_proba'],
            title=f"{results['model']} - ROC Curve ({results['dataset']})",
            ax=axes[0]
        )

        # PR Curve
        plot_precision_recall_curve(
            y_true,
            results['y_pred_proba'],
            title=f"{results['model']} - PR Curve ({results['dataset']})",
            ax=axes[1]
        )

        # Confusion Matrix
        plot_confusion_matrix(
            y_true,
            results['y_pred'],
            title=f"{results['model']} - Confusion Matrix ({results['dataset']})",
            ax=axes[2]
        )

        plt.tight_layout()

        return fig, axes

    def compare_models(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = 'Dataset'
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same dataset.

        Args:
            models: Dictionary of models {name: model}
            X: Features
            y: True labels
            dataset_name: Name of the dataset

        Returns:
            DataFrame with comparison results
        """
        results = []

        for name, model in models.items():
            evaluator = ModelEvaluator(model, name)
            eval_results = evaluator.evaluate(X, y, dataset_name)

            result_row = {
                'Model': name,
                'Dataset': dataset_name,
                **eval_results['metrics']
            }
            results.append(result_row)

        comparison_df = pd.DataFrame(results)

        return comparison_df
