"""
Loss functions for classification models
"""

import numpy as np
from .base import Loss

class BinaryCrossEntropy(Loss):
    """Any binary classifier (logistic regression, neural network) can use this loss"""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Clip predictions to avoid log(0)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Gradient of BCE with respect to y_pred."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)