from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """
    An abstract base class that declares three methods every model must have:

        fit(X, y): learn from data.

        predict(X): generate outputs for new data.

        score(X, y): evaluate performance (with a sensible default).
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model according to the given training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for X."""
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Default scoring metric. Override in subclasses if needed.
        For regression, defaults to R²; for classification, this would be overridden.
        """
        y_pred = self.predict(X)
        # Default: R² (coefficient of determination)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0