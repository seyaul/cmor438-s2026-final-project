import numpy as np
from .base import Metric


class R2Score(Metric):
    """R² (coefficient of determination) regression score."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return the coefficient of determination R²."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


class RMSE(Metric):
    """Root Mean Squared Error."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return root mean squared error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


class MAE(Metric):
    """Mean Absolute Error."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return mean absolute error."""
        return np.mean(np.abs(y_true - y_pred))


# Convenience function aliases (backward compatible)
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    return R2Score()(y_true, y_pred)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return RMSE()(y_true, y_pred)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return MAE()(y_true, y_pred)
