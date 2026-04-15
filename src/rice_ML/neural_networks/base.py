from abc import abstractmethod
import numpy as np
from ..base.base_model import BaseModel

class BaseNeuralNetwork(BaseModel):
    def __init__(self):
        self.layers = []
        self._is_built = False

    @abstractmethod
    def _build(self, input_shape: tuple) -> None:
        """Construct layers based on input shape."""
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self) -> dict:
        """Return all trainable parameters as a dict {layer_name: {param_name: array}}."""
        params = {}
        for i, layer in enumerate(self.layers):
            layer_params = layer.parameters()
            if layer_params:
                params[f"layer_{i}"] = layer_params
        return params

    def gradients(self) -> dict:
        """Return gradients for all trainable parameters (same structure as parameters)."""
        grads = {}
        for i, layer in enumerate(self.layers):
            layer_grads = layer.gradients()
            if layer_grads:
                grads[f"layer_{i}"] = layer_grads
        return grads