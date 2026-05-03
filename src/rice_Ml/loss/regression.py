import numpy as np
from .base import Loss

class MeanSquaredError(Loss):
    """Mean Squared Error loss for regression."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Gradient of MSE with respect to y_pred.
        Returns array of same shape as y_pred.
        """
        return 2 * (y_pred - y_true) / y_true.size