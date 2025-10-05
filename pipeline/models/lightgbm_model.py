"""LightGBM Model Wrapper

This module contains a wrapper class for LightGBM models.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


class LightGBMModel:
    """
    LightGBM model wrapper with custom functionality.
    """

    def __init__(self, **params):
        """
        Initialize LightGBM model.

        Args:
            **params: LightGBM parameters
        """
        self.params = params
        self.model = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True
    ):
        """
        Train LightGBM model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            verbose: Whether to print training progress
        """
        self.model = lgb.LGBMClassifier(**self.params)

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        callbacks = None if verbose else [lgb.early_stopping(50, verbose=False)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features

        Returns:
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features

        Returns:
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        importance_df = pd.DataFrame({
            'feature': self.model.feature_name_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df

    def save(self, filepath: str):
        """
        Save model to file.

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        self.model.booster_.save_model(filepath)

    def load(self, filepath: str):
        """
        Load model from file.

        Args:
            filepath: Path to load model from
        """
        self.model = lgb.Booster(model_file=filepath)
