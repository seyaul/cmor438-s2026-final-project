from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    """
    Defines the abstract interface that all activation functions must implement. 
    This ensures consistency and allows models to swap activations easily.
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the activation function element‑wise."""
        pass

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the activation function with respect to its input.

        Parameters
        ----------
        x : np.ndarray
            Input array (pre‑activation values).

        Returns
        -------
        np.ndarray
            Derivative evaluated at x.
        """
        pass