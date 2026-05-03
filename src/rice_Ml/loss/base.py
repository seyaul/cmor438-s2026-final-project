from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    """Abstract base class for all loss functions."""

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Gradient of the loss with respect to y_pred."""
        pass