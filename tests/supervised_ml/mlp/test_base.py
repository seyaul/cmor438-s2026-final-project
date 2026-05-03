import numpy as np
import pytest
from rice_Ml.supervised_ml.mlp.base import BaseNeuralNetwork
from rice_Ml.supervised_ml.mlp.layers import Dense
from rice_Ml.activations import Linear

class MinimalNN(BaseNeuralNetwork):
    def _build(self, input_shape):
        self.layers.append(Dense(2, Linear()))
        self.layers.append(Dense(1, Linear()))

    def fit(self, X, y):
        self._build(X.shape)
        self._is_built = True
        _ = self.forward(X)
        return self

    def predict(self, X):
        return self.forward(X)

class TestBaseNeuralNetwork:
    def test_forward_pass(self):
        X = np.random.randn(10, 3)
        model = MinimalNN()
        model.fit(X, None)
        out = model.forward(X)
        assert out.shape == (10, 1)

    def test_parameters_dict(self):
        X = np.random.randn(10, 3)
        model = MinimalNN()
        model.fit(X, np.zeros((10, 1)))
        params = model.parameters()
        assert 'layer_0' in params
        assert 'W' in params['layer_0']

    def test_gradients_dict_after_backward(self):
        X = np.random.randn(10, 3)
        model = MinimalNN()
        model.fit(X, None)
        out = model.forward(X)
        grad = np.ones_like(out)
        model.backward(grad)
        grads = model.gradients()
        assert 'layer_0' in grads
        assert 'layer_1' in grads
        assert grads['layer_0']['W'].shape == (3, 2)
