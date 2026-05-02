import numpy as np
import pytest
from rice_ML.neural_networks.layers import Dense
from rice_ML.activations import Sigmoid, ReLU, Linear

class TestDenseLayer:
    def test_forward_shape(self):
        X = np.random.randn(10, 5)
        layer = Dense(units=3)
        out = layer.forward(X)
        assert out.shape == (10, 3)

    def test_forward_with_activation(self):
        X = np.random.randn(10, 5)
        layer = Dense(units=3, activation=Sigmoid())
        out = layer.forward(X)
        assert out.shape == (10, 3)
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_parameter_shapes(self):
        X = np.random.randn(10, 5)
        layer = Dense(units=3)
        layer.forward(X)  # builds
        params = layer.parameters()
        assert params['W'].shape == (5, 3)
        assert params['b'].shape == (1, 3)

    def test_backward_gradient_shapes(self):
        X = np.random.randn(10, 5)
        layer = Dense(units=3)
        layer.forward(X)
        dA = np.random.randn(10, 3)
        dX = layer.backward(dA)
        assert dX.shape == (10, 5)
        grads = layer.gradients()
        assert grads['W'].shape == (5, 3)
        assert grads['b'].shape == (1, 3)

    def test_numerical_gradient_W(self):
        """Verify gradient of loss w.r.t. W using finite differences."""
        X = np.random.randn(5, 4)
        layer = Dense(units=2, activation=Linear())
        layer.forward(X)

        # Use sum of outputs as loss (no scaling)
        def loss_fn():
            out = layer.forward(X)
            return np.sum(out)

        # For sum loss, dA = np.ones_like(out)
        out = layer.forward(X)
        dA = np.ones_like(out)
        layer.backward(dA)
        grad_W_analytic = layer.gradients()['W']

        # Numerical gradient
        eps = 1e-5
        grad_W_numeric = np.zeros_like(grad_W_analytic)
        W = layer.parameters()['W']
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] += eps
                loss_plus = loss_fn()
                W[i, j] -= 2 * eps
                loss_minus = loss_fn()
                W[i, j] += eps
                grad_W_numeric[i, j] = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(grad_W_analytic, grad_W_numeric, rtol=1e-4, atol=1e-6)

    def test_numerical_gradient_b(self):
        X = np.random.randn(5, 4)
        layer = Dense(units=2, activation=Linear())
        layer.forward(X)

        def loss_fn():
            out = layer.forward(X)
            return np.sum(out)

        out = layer.forward(X)
        dA = np.ones_like(out)
        layer.backward(dA)
        grad_b_analytic = layer.gradients()['b']

        eps = 1e-5
        grad_b_numeric = np.zeros_like(grad_b_analytic)
        b = layer.parameters()['b']
        for j in range(b.shape[1]):
            b[0, j] += eps
            loss_plus = loss_fn()
            b[0, j] -= 2 * eps
            loss_minus = loss_fn()
            b[0, j] += eps
            grad_b_numeric[0, j] = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(grad_b_analytic, grad_b_numeric, rtol=1e-4, atol=1e-6)

    def test_backward_with_activation(self):
        X = np.random.randn(10, 5)
        layer = Dense(units=3, activation=Sigmoid())
        out = layer.forward(X)
        dA = np.random.randn(10, 3)
        dX = layer.backward(dA)
        assert dX.shape == (10, 5)
        # Gradients should exist and be finite
        grads = layer.gradients()
        assert np.all(np.isfinite(grads['W']))
        assert np.all(np.isfinite(grads['b']))

    def test_custom_initializer(self):
        from rice_ML.neural_networks.initializers import Ones
        X = np.random.randn(10, 5)
        layer = Dense(units=3, kernel_initializer=Ones())
        layer.forward(X)
        W = layer.parameters()['W']
        assert np.all(W == 1.0)

    def test_string_initializer_alias(self):
        X = np.random.randn(10, 5)
        layer = Dense(units=3, kernel_initializer='zeros')
        layer.forward(X)
        W = layer.parameters()['W']
        assert np.all(W == 0.0)

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Unknown initializer"):
            Dense(units=3, kernel_initializer='not_an_initializer')