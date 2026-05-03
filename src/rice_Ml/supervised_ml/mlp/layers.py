import numpy as np
from typing import Union, Optional
from abc import ABC, abstractmethod

from ...activations import Activation, Linear
from .initializers import (
    Initializer,
    XavierUniform, XavierNormal,
    HeUniform, HeNormal,
    RandomNormal, RandomUniform,
    Zeros, Ones,
)


class Layer(ABC):
    """Abstract base class for all neural network layers."""

    def __init__(self):
        self._params = {}
        self._grads = {}

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> dict:
        return self._params

    def gradients(self) -> dict:
        return self._grads


class Dense(Layer):
    _INITIALIZERS = {
        'xavier_uniform': XavierUniform,
        'xavier_normal': XavierNormal,
        'he_uniform': HeUniform,
        'he_normal': HeNormal,
        'random_normal': RandomNormal,
        'random_uniform': RandomUniform,
        'zeros': Zeros,
        'ones': Ones,
    }

    def __init__(
        self,
        units: int,
        activation: Optional[Activation] = None,
        kernel_initializer: Union[Initializer, str] = 'xavier_uniform',
    ):
        super().__init__()
        self.units = units
        self.activation = activation or Linear()

        if isinstance(kernel_initializer, str):
            if kernel_initializer not in self._INITIALIZERS:
                raise ValueError(f"Unknown initializer: {kernel_initializer}")
            self.kernel_initializer = self._INITIALIZERS[kernel_initializer]()
        else:
            self.kernel_initializer = kernel_initializer or XavierUniform()

        self._built = False

    def _build(self, input_dim: int) -> None:
        self._params['W'] = self.kernel_initializer((input_dim, self.units))
        self._params['b'] = np.zeros((1, self.units))
        self._built = True

    def forward(self, X: np.ndarray) -> np.ndarray:
        if not self._built:
            self._build(X.shape[1])
        self.X = X
        self.Z = X @ self._params['W'] + self._params['b']
        return self.activation(self.Z)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        dZ = dA * self.activation.gradient(self.Z)
        batch_size = self.X.shape[0]
        self._grads['W'] = self.X.T @ dZ / batch_size
        self._grads['b'] = np.sum(dZ, axis=0, keepdims=True) / batch_size
        dX = dZ @ self._params['W'].T
        return dX
