import numpy as np
from .base import Activation

class Sigmoid(Activation):
    """Sigmoid activation: σ(x) = 1 / (1 + e^{-x})."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Clip extreme values to avoid overflow in exp
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        s = self(x)
        return s * (1 - s)


class ReLU(Activation):
    """Rectified Linear Unit: max(0, x)."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class Tanh(Activation):
    """Hyperbolic tangent: tanh(x)."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class Linear(Activation):
    """Identity activation: f(x) = x (useful for regression output)."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class Softmax(Activation):
    """
    Softmax activation for multi‑class classification.
    Returns probabilities that sum to 1 along the last axis.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability
        x_max = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        # Fused with CategoricalCrossEntropy: the combined gradient dL/dz = softmax(z) - y
        # is passed as dA into Dense.backward. Returning ones lets it pass straight through.
        return np.ones_like(x)