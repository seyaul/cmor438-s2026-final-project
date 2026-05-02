"""
CNN layer primitives: Conv1D, MaxPool1D, Flatten, Dense.

Each layer exposes:
    forward(X)     -> output array (caches inputs needed for backward)
    backward(dout) -> gradient w.r.t. input
    params         -> dict of learnable parameter arrays
    grads          -> dict of parameter gradients (populated after backward)
"""
from __future__ import annotations

import numpy as np


class Conv1D:
    """1-D convolutional layer.

    Input shape:  (batch, in_channels, length)
    Output shape: (batch, out_channels, length - kernel_size + 1)

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # He initialisation for ReLU activations
        scale = np.sqrt(2.0 / (in_channels * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size) * scale
        self.b = np.zeros(out_channels)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._X: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        batch, in_ch, length = X.shape
        out_length = length - self.kernel_size + 1
        out = np.zeros((batch, self.out_channels, out_length), dtype=np.float64)
        for k in range(self.kernel_size):
            # X slice: (batch, in_ch, out_length)
            # W[:, :, k]:  (out_ch, in_ch)
            out += np.einsum("bik,oi->bok", X[:, :, k : k + out_length], self.W[:, :, k])
        out += self.b[np.newaxis, :, np.newaxis]
        self._X = X
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        X = self._X
        batch, in_ch, length = X.shape
        out_length = dout.shape[2]

        dX = np.zeros_like(X)
        self.dW = np.zeros_like(self.W)
        self.db = dout.sum(axis=(0, 2))

        for k in range(self.kernel_size):
            self.dW[:, :, k] = np.einsum(
                "bok,bik->oi", dout, X[:, :, k : k + out_length]
            )
            dX[:, :, k : k + out_length] += np.einsum(
                "bok,oi->bik", dout, self.W[:, :, k]
            )
        return dX

    @property
    def params(self) -> dict:
        return {"W": self.W, "b": self.b}

    @property
    def grads(self) -> dict:
        return {"W": self.dW, "b": self.db}


class MaxPool1D:
    """1-D max-pooling layer.

    Input shape:  (batch, channels, length)
    Output shape: (batch, channels, length // pool_size)

    Trailing samples that don't fill a full window are dropped.

    Parameters
    ----------
    pool_size : int, default 2
    """

    def __init__(self, pool_size: int = 2) -> None:
        self.pool_size = pool_size
        self._cache: tuple | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        batch, channels, length = X.shape
        out_length = length // self.pool_size
        X_trunc = X[:, :, : out_length * self.pool_size]
        X_reshaped = X_trunc.reshape(batch, channels, out_length, self.pool_size)
        out = X_reshaped.max(axis=3)
        self._cache = (X, X_reshaped, out)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        X, X_reshaped, out = self._cache
        batch, channels, length = X.shape
        out_length = length // self.pool_size

        # Gradient flows only to the position that held the maximum value.
        # Ties are split evenly.
        mask = X_reshaped == out[:, :, :, np.newaxis]
        mask = mask / mask.sum(axis=3, keepdims=True)

        dX = np.zeros_like(X)
        dX[:, :, : out_length * self.pool_size] = (
            (mask * dout[:, :, :, np.newaxis])
            .reshape(batch, channels, out_length * self.pool_size)
        )
        return dX

    @property
    def params(self) -> dict:
        return {}

    @property
    def grads(self) -> dict:
        return {}


class Flatten:
    """Flatten all dimensions after the batch dimension into a single vector.

    Input shape:  (batch, ...)
    Output shape: (batch, product_of_remaining_dims)
    """

    def __init__(self) -> None:
        self._input_shape: tuple | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self._input_shape)

    @property
    def params(self) -> dict:
        return {}

    @property
    def grads(self) -> dict:
        return {}


class Dense:
    """Fully-connected linear layer.

    Input shape:  (batch, in_features)
    Output shape: (batch, out_features)

    Parameters
    ----------
    in_features : int
    out_features : int
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._X: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._X = X
        return X @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW = self._X.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    @property
    def params(self) -> dict:
        return {"W": self.W, "b": self.b}

    @property
    def grads(self) -> dict:
        return {"W": self.dW, "b": self.db}
