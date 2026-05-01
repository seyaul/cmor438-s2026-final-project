import numpy as np
import pytest
from rice_ML.supervised_ml.Perceptron.perceptron import Perceptron
from sklearn.datasets import make_blobs


@pytest.fixture
def _make_test_data():
    # tightly separated blobs guarantee linear separability
    X, y = make_blobs(n_samples=100, centers=2, cluster_std=0.5, random_state=0)
    y = np.where(y == 0, -1, y)
    return X, y


class TestInit:
    # checks all post-training attributes exist and have the right types/shapes
    def test_train_perceptron(self, _make_test_data):
        X, y = _make_test_data
        clf = Perceptron(eta=0.1, epochs=50).train(X, y)

        assert hasattr(clf, "w_b_")
        assert hasattr(clf, "eta")
        assert hasattr(clf, "epochs")
        assert hasattr(clf, "mistakes_")

        assert isinstance(clf.w_b_, np.ndarray)
        assert clf.w_b_.shape == (1 + X.shape[1],)

        assert isinstance(clf.mistakes_, list)
        assert all(isinstance(e, (int, np.integer)) for e in clf.mistakes_)

    # tests if accuracy is above 0.95 and if our model converges on blob data
    def test_accuracy(self, _make_test_data):
        X, y = _make_test_data
        clf = Perceptron(eta=0.1, epochs=200, random_state=0).train(X, y)

        pred = clf.predict(X)
        acc = np.mean(pred == y)
        assert acc >= 0.95
        assert clf.converged_

    # linearly separable data should converge before the epoch limit
    def test_early_stopping(self, _make_test_data):
        X, y = _make_test_data
        p = Perceptron(eta=0.1, epochs=500, random_state=0)
        p.train(X, y)
        assert len(p.mistakes_) < 500

    def test_invalid_eta_raises(self):
        with pytest.raises(ValueError):
            Perceptron(eta=0)

    def test_negative_eta_raises(self):
        with pytest.raises(ValueError):
            Perceptron(eta=-1)

    def test_invalid_epoch_raises(self):
        with pytest.raises(ValueError):
            Perceptron(epochs=0)

    # perceptron should accept plain Python lists, not just numpy arrays
    def test_accepts_python_lists(self):
        X = [[1.0, 0.0], [-1.0, 0.0]]
        y = [1, -1]
        p = Perceptron()
        p.train(X, y)
        assert hasattr(p, "w_b_")

    # score() should return near-perfect accuracy on separable blob data
    def test_score_perfect_on_blobs(self, _make_test_data):
        X, y = _make_test_data
        clf = Perceptron(eta=0.1, epochs=200, random_state=0).train(X, y)
        assert clf.score(X, y) >= 0.99

    # score() must return a Python float, not an ndarray or np.float64
    def test_score_returns_float(self, _make_test_data):
        X, y = _make_test_data
        clf = Perceptron(eta=0.1, epochs=200, random_state=0).train(X, y)
        assert isinstance(clf.score(X, y), float)

    # early stopping means mistakes_ should be shorter than the epoch limit
    def test_mistakes_length_less_than_epochs(self, _make_test_data):
        X, y = _make_test_data
        clf = Perceptron(eta=0.1, epochs=200, random_state=0).train(X, y)
        assert len(clf.mistakes_) < 200

    # mistake count should not increase over training on separable data
    def test_mistakes_decreasing(self, _make_test_data):
        X, y = _make_test_data
        clf = Perceptron(eta=0.1, epochs=200, random_state=0).train(X, y)
        assert clf.mistakes_[0] >= clf.mistakes_[-1]


class TestValidation:
    # X and y must have the same number of samples
    def test_mismatched_xy_lengths(self):
        X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        y = np.array([1, -1])
        with pytest.raises(ValueError):
            Perceptron().train(X, y)

    # labels must be 1 or -1 — passing 0/1 should raise
    def test_invalid_labels_raises(self):
        X = np.array([[1.0, 1.0], [2.0, 2.0]])
        y = np.array([0, 1])
        with pytest.raises(ValueError):
            Perceptron().train(X, y)

    # X must be 2-D; a flat array should raise
    def test_1d_X_raises(self):
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([1, -1, 1])
        with pytest.raises(ValueError):
            Perceptron().train(X, y)


class TestPredict:
    # all predicted values must be exactly 1 or -1
    def test_predict_output_binary(self, _make_test_data):
        X, y = _make_test_data
        clf = Perceptron(eta=0.1, epochs=200, random_state=0).train(X, y)
        preds = clf.predict(X)
        assert set(preds).issubset({1, -1})

    # batch predict should return one label per sample
    def test_predict_batch_shape(self, _make_test_data):
        X, y = _make_test_data
        clf = Perceptron(eta=0.1, epochs=200, random_state=0).train(X, y)
        assert clf.predict(X).shape == (len(X),)

    # single-sample predict should return a scalar 1 or -1
    def test_predict_single_sample(self, _make_test_data):
        X, y = _make_test_data
        clf = Perceptron(eta=0.1, epochs=200, random_state=0).train(X, y)
        result = clf.predict(X[0])
        assert result in (1, -1)


class TestReproducibility:
    # same random_state must produce identical weights across runs
    def test_same_random_state_same_weights(self, _make_test_data):
        X, y = _make_test_data
        p1 = Perceptron(eta=0.1, epochs=200, random_state=7).train(X, y)
        p2 = Perceptron(eta=0.1, epochs=200, random_state=7).train(X, y)
        np.testing.assert_array_equal(p1.w_b_, p2.w_b_)

    # different seeds should produce different starting weights and final weights
    def test_different_random_state_different_weights(self, _make_test_data):
        X, y = _make_test_data
        p1 = Perceptron(eta=0.1, epochs=5, random_state=0).train(X, y)
        p2 = Perceptron(eta=0.1, epochs=5, random_state=99).train(X, y)
        assert not np.allclose(p1.w_b_, p2.w_b_)


class TestEpochsRun:
    # n_epochs_run_ must be set after training
    def test_n_epochs_run_set(self, _make_test_data):
        X, y = _make_test_data
        clf = Perceptron(eta=0.1, epochs=200, random_state=0).train(X, y)
        assert hasattr(clf, "n_epochs_run_")

    # n_epochs_run_ must be between 1 and the epoch limit (inclusive)
    def test_n_epochs_run_in_range(self, _make_test_data):
        X, y = _make_test_data
        clf = Perceptron(eta=0.1, epochs=200, random_state=0).train(X, y)
        assert 1 <= clf.n_epochs_run_ <= 200
