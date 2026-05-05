"""
Loss functions for classification models
"""

import numpy as np
from .base import Loss

class BinaryCrossEntropy(Loss):
    """Any binary classifier (logistic regression, neural network) can use this loss"""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return binary cross-entropy loss between y_true and y_pred."""
        # Clip predictions to avoid log(0)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Gradient of BCE with respect to y_pred."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)


class CategoricalCrossEntropy(Loss):
    """Softmax + cross-entropy fused loss for multi-class classification."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return categorical cross-entropy loss (softmax fused)."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Fused softmax+CE gradient: dL/dz = softmax(z) - y_onehot.
        # Dense.backward divides by batch_size, so return the unnormalized diff.
        return y_pred - y_true