"""Evaluation Module

This module contains model evaluation utilities and metrics.
"""

from .metrics import (
    calculate_metrics,
    ModelEvaluator,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix
)

__all__ = [
    'calculate_metrics',
    'ModelEvaluator',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_confusion_matrix',
]
