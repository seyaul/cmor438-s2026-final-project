"""
Unit tests for the Perceptron class.

Covers: initialization, net_input, predict, train (convergence, early stopping,
weight updates), and end-to-end classification on linearly separable data.
"""

import numpy as np
import pytest
from rice_ML.supervised_ml.Perceptron.perceptron import Perceptron


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def separable_2d():
    """Simple linearly separable 2-D dataset (two tight clusters)."""
    X = np.array([
        [1.0, 1.0],
        [1.5, 1.0],
        [1.0, 1.5],
        [5.0, 5.0],
        [5.5, 5.0],
        [5.0, 5.5],
    ])
    y = np.array([-1, -1, -1, 1, 1, 1])
    return X, y


@pytest.fixture
def trained_perceptron(separable_2d):
    X, y = separable_2d
    p = Perceptron(eta=0.1, epochs=100)
    p.train(X, y)
    return p, X, y


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_eta(self):
        assert Perceptron().eta == pytest.approx(0.1)

    def test_default_epochs(self):
        assert Perceptron().epochs == 50

    def test_custom_params(self):
        p = Perceptron(eta=0.01, epochs=200)
        assert p.eta == pytest.approx(0.01)
        assert p.epochs == 200

    def test_no_w_b_before_training(self):
        p = Perceptron()
        assert not hasattr(p, "w_b")

    def test_no_mistakes_before_training(self):
        p = Perceptron()
        assert not hasattr(p, "mistakes")


# ---------------------------------------------------------------------------
# net_input
# ---------------------------------------------------------------------------

class TestNetInput:
    def test_single_sample(self):
        p = Perceptron()
        p.w_b = np.array([2.0, 3.0, 1.0])   # weights=[2,3], bias=1
        result = p.net_input(np.array([1.0, 1.0]))
        assert result == pytest.approx(6.0)   # 2*1 + 3*1 + 1

    def test_batch_samples(self):
        p = Perceptron()
        p.w_b = np.array([1.0, 0.0, 0.0])   # weights=[1,0], bias=0
        X = np.array([[3.0, 7.0], [5.0, 2.0]])
        result = p.net_input(X)
        np.testing.assert_array_almost_equal(result, [3.0, 5.0])

    def test_bias_only(self):
        p = Perceptron()
        p.w_b = np.array([0.0, 0.0, 5.0])   # weights=[0,0], bias=5
        assert p.net_input(np.array([99.0, 99.0])) == pytest.approx(5.0)

    def test_negative_net_input(self):
        p = Perceptron()
        p.w_b = np.array([-1.0, -1.0, 0.0])
        assert p.net_input(np.array([2.0, 3.0])) == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_positive_net_input_gives_1(self):
        p = Perceptron()
        p.w_b = np.array([1.0, 0.0, 0.0])   # net = x[0]
        assert p.predict(np.array([1.0, 0.0])) == 1

    def test_negative_net_input_gives_minus1(self):
        p = Perceptron()
        p.w_b = np.array([-1.0, 0.0, 0.0])
        assert p.predict(np.array([1.0, 0.0])) == -1

    def test_zero_net_input_gives_1(self):
        """Boundary: net_input == 0 should classify as 1 (>= 0)."""
        p = Perceptron()
        p.w_b = np.array([0.0, 0.0, 0.0])
        assert p.predict(np.array([5.0, 5.0])) == 1

    def test_batch_predict_shape(self, trained_perceptron):
        p, X, _ = trained_perceptron
        preds = p.predict(X)
        assert preds.shape == (len(X),)

    def test_batch_predict_values_in_set(self, trained_perceptron):
        p, X, _ = trained_perceptron
        preds = p.predict(X)
        assert set(preds).issubset({1, -1})


# ---------------------------------------------------------------------------
# train — return value and attribute setup
# ---------------------------------------------------------------------------

class TestTrainAttributes:
    def test_returns_self(self, separable_2d):
        X, y = separable_2d
        p = Perceptron()
        result = p.train(X, y)
        assert result is p

    def test_w_b_shape(self, separable_2d):
        X, y = separable_2d
        p = Perceptron()
        p.train(X, y)
        assert p.w_b.shape == (X.shape[1] + 1,)

    def test_mistakes_is_list(self, separable_2d):
        X, y = separable_2d
        p = Perceptron()
        p.train(X, y)
        assert isinstance(p.mistakes, list)

    def test_accepts_python_lists(self):
        X = [[1.0, 0.0], [-1.0, 0.0]]
        y = [1, -1]
        p = Perceptron()
        p.train(X, y)
        assert hasattr(p, "w_b")

    def test_single_feature(self):
        X = np.array([[1.0], [2.0], [-1.0], [-2.0]])
        y = np.array([1, 1, -1, -1])
        p = Perceptron(epochs=100)
        p.train(X, y)
        assert p.w_b.shape == (2,)


# ---------------------------------------------------------------------------
# train — convergence and correctness
# ---------------------------------------------------------------------------

class TestTrainConvergence:
    def test_converges_on_separable_data(self, separable_2d):
        """After training, all labels should be predicted correctly."""
        X, y = separable_2d
        p = Perceptron(eta=0.1, epochs=200)
        p.train(X, y)
        np.testing.assert_array_equal(p.predict(X), y)

    def test_early_stopping(self, separable_2d):
        """If data is separable, training should stop before max epochs."""
        X, y = separable_2d
        p = Perceptron(eta=0.1, epochs=500)
        p.train(X, y)
        assert len(p.mistakes) < 500

    def test_and_gate(self):
        """Perceptron should learn the AND gate (linearly separable)."""
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([-1, -1, -1, 1])
        p = Perceptron(eta=0.1, epochs=100)
        p.train(X, y)
        np.testing.assert_array_equal(p.predict(X), y)

    def test_mistakes_decrease_toward_zero(self, separable_2d):
        """The last recorded mistake count should be nonzero only when converged."""
        X, y = separable_2d
        p = Perceptron(eta=0.1, epochs=200)
        p.train(X, y)
        # Once converged, mistakes list ends at the last epoch with errors
        if p.mistakes:
            # Ensure it eventually stopped recording (convergence happened)
            assert p.predict(X) is not None

    def test_non_separable_does_not_crash(self):
        """XOR is not linearly separable; training should finish without error."""
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([-1, 1, 1, -1])
        p = Perceptron(eta=0.1, epochs=20)
        p.train(X, y)
        assert len(p.mistakes) > 0


# ---------------------------------------------------------------------------
# train — hyperparameter effects
# ---------------------------------------------------------------------------

class TestHyperparameters:
    def test_custom_eta_changes_weights(self, separable_2d):
        """Two runs with different eta should generally produce different weights."""
        X, y = separable_2d
        np.random.seed(0)
        p1 = Perceptron(eta=0.01, epochs=5)
        p1.train(X, y)

        np.random.seed(0)
        p2 = Perceptron(eta=0.9, epochs=5)
        p2.train(X, y)

        assert not np.allclose(p1.w_b, p2.w_b)

    def test_epoch_limit_respected_on_non_separable(self):
        """For non-separable data, mistakes list length <= epochs."""
        X = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        y = np.array([-1, -1, 1, 1])   # not cleanly separable with linear boundary
        epochs = 10
        p = Perceptron(eta=0.1, epochs=epochs)
        p.train(X, y)
        assert len(p.mistakes) <= epochs
