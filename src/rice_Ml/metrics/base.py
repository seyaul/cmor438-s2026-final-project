from abc import ABC, abstractmethod
import numpy as np

class Metric(ABC):
    """Abstract base class for all evaluation metrics."""

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the metric."""
        pass