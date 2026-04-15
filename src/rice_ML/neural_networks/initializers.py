import numpy as np
from abc import ABC, abstractmethod

class Initializer(ABC):
    """Abstract base class for weight initializers."""

    @abstractmethod
    def __call__(self, shape: tuple) -> np.ndarray:
        """Return an initialized array of the given shape."""
        pass


class Zeros(Initializer):
    def __call__(self, shape: tuple) -> np.ndarray:
        return np.zeros(shape)


class Ones(Initializer):
    def __call__(self, shape: tuple) -> np.ndarray:
        return np.ones(shape)


class RandomNormal(Initializer):
    def __init__(self, mean: float = 0.0, stddev: float = 0.05):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape: tuple) -> np.ndarray:
        return np.random.normal(self.mean, self.stddev, shape)


class RandomUniform(Initializer):
    def __init__(self, minval: float = -0.05, maxval: float = 0.05):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, shape: tuple) -> np.ndarray:
        return np.random.uniform(self.minval, self.maxval, shape)


class XavierUniform(Initializer):
    """Glorot uniform initializer (also called Xavier uniform)."""
    def __call__(self, shape: tuple) -> np.ndarray:
        if len(shape) != 2:
            raise ValueError(
                f"XavierUniform expects a 2D shape (fan_in, fan_out), got {shape}"
            )
        fan_in, fan_out = shape
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)


class XavierNormal(Initializer):
    """Glorot normal initializer (also called Xavier normal)."""
    def __call__(self, shape: tuple) -> np.ndarray:
        if len(shape) != 2:
            raise ValueError(
                f"XavierNormal expects a 2D shape (fan_in, fan_out), got {shape}"
            )
        fan_in, fan_out = shape
        stddev = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0.0, stddev, shape)


class HeUniform(Initializer):
    """He uniform initializer (good for ReLU)."""
    def __call__(self, shape: tuple) -> np.ndarray:
        if len(shape) != 2:
            raise ValueError(
                f"HeUniform expects a 2D shape (fan_in, fan_out), got {shape}"
            )
        fan_in = shape[0]
        limit = np.sqrt(6.0 / fan_in)
        return np.random.uniform(-limit, limit, shape)


class HeNormal(Initializer):
    """He normal initializer (good for ReLU)."""
    def __call__(self, shape: tuple) -> np.ndarray:
        if len(shape) != 2:
            raise ValueError(
                f"HeNormal expects a 2D shape (fan_in, fan_out), got {shape}"
            )
        fan_in = shape[0]
        stddev = np.sqrt(2.0 / fan_in)
        return np.random.normal(0.0, stddev, shape)