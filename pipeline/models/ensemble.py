"""Ensemble Model

This module contains ensemble model implementations.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any


class EnsembleModel:
    """
    Ensemble model that combines predictions from multiple models.
    """

    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        """
        Initialize ensemble model.

        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal weights)
        """
        self.models = models
        self.weights = weights if weights is not None else [1.0 / len(models)] * len(models)

        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")

        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using weighted average of models.

        Args:
            X: Features

        Returns:
            Weighted average of predicted probabilities
        """
        predictions = []

        for model in self.models:
            pred_proba = model.predict_proba(X)
            predictions.append(pred_proba)

        # Weighted average
        ensemble_proba = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_proba += pred * weight

        return ensemble_proba

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels using threshold.

        Args:
            X: Features
            threshold: Classification threshold (default: 0.5)

        Returns:
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    def optimize_weights(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric_func: callable
    ) -> List[float]:
        """
        Optimize ensemble weights using validation data.

        Args:
            X_val: Validation features
            y_val: Validation target
            metric_func: Metric function to optimize (higher is better)

        Returns:
            Optimized weights
        """
        from scipy.optimize import minimize

        def objective(weights):
            # Normalize weights to sum to 1
            weights = weights / weights.sum()

            # Calculate ensemble predictions
            ensemble_proba = np.zeros((len(X_val), 2))
            for model, weight in zip(self.models, weights):
                pred_proba = model.predict_proba(X_val)
                ensemble_proba += pred_proba * weight

            # Calculate metric (negative because we minimize)
            score = metric_func(y_val, ensemble_proba[:, 1])
            return -score

        # Initial weights
        initial_weights = np.array(self.weights)

        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}

        # Bounds: each weight between 0 and 1
        bounds = [(0, 1) for _ in range(len(self.models))]

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        # Update weights
        optimized_weights = result.x / result.x.sum()
        self.weights = optimized_weights.tolist()

        return self.weights


class VotingEnsemble:
    """
    Voting ensemble that uses majority voting or soft voting.
    """

    def __init__(self, models: List[Any], voting: str = 'soft'):
        """
        Initialize voting ensemble.

        Args:
            models: List of trained models
            voting: 'hard' for majority voting, 'soft' for probability averaging
        """
        self.models = models
        self.voting = voting

        if voting not in ['hard', 'soft']:
            raise ValueError("voting must be 'hard' or 'soft'")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels using voting.

        Args:
            X: Features

        Returns:
            Predicted class labels
        """
        if self.voting == 'hard':
            # Hard voting: majority vote
            predictions = []
            for model in self.models:
                pred = model.predict(X)
                predictions.append(pred)

            predictions = np.array(predictions)
            # Majority vote
            return np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=0,
                arr=predictions
            )
        else:
            # Soft voting: average probabilities
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using average.

        Args:
            X: Features

        Returns:
            Average of predicted probabilities
        """
        predictions = []

        for model in self.models:
            pred_proba = model.predict_proba(X)
            predictions.append(pred_proba)

        # Average
        ensemble_proba = np.mean(predictions, axis=0)

        return ensemble_proba
