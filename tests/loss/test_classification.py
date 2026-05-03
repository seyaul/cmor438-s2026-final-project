import numpy as np
import pytest
from rice_ML.loss.classification import BinaryCrossEntropy

def test_bce_perfect_prediction():
    bce = BinaryCrossEntropy()
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0, 1])
    assert bce(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)

def test_bce_gradient():
    bce = BinaryCrossEntropy()
    y_true = np.array([1, 0])
    y_pred = np.array([0.9, 0.1])
    grad = bce.gradient(y_true, y_pred)
    # Manual computation
    expected = -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)
    np.testing.assert_array_almost_equal(grad, expected)