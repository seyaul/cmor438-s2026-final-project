import numpy as np
import pytest
from rice_Ml.activations import Sigmoid, ReLU, Tanh, Linear

def test_sigmoid_output():
    sig = Sigmoid()
    assert sig(0) == 0.5
    assert sig(100) == pytest.approx(1.0)
    assert sig(-100) == pytest.approx(0.0)

def test_sigmoid_gradient():
    sig = Sigmoid()
    x = np.array([-2, 0, 2])
    grad = sig.gradient(x)
    expected = sig(x) * (1 - sig(x))
    np.testing.assert_array_almost_equal(grad, expected)

def test_relu():
    relu = ReLU()
    x = np.array([-2, -1, 0, 1, 2])
    np.testing.assert_array_equal(relu(x), [0, 0, 0, 1, 2])
    np.testing.assert_array_equal(relu.gradient(x), [0, 0, 0, 1, 1])

def test_tanh():
    tanh = Tanh()
    x = np.array([0])
    assert tanh(x) == 0.0
    np.testing.assert_almost_equal(tanh.gradient(x), 1.0)

def test_linear():
    lin = Linear()
    x = np.array([1, 2, 3])
    np.testing.assert_array_equal(lin(x), x)
    np.testing.assert_array_equal(lin.gradient(x), np.ones_like(x))