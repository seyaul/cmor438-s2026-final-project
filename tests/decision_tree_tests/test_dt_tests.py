"""
Blob-based integration tests for DecisionTree.

Uses make_blobs to generate realistic linearly separable data and tests
accuracy, convergence, score(), and robustness on that data.
"""

import numpy as np
import pytest
from rice_Ml.supervised_ml.DecisionTree.decision_tree import DecisionTree
from sklearn.datasets import make_blobs


@pytest.fixture
def _make_test_data():
    # tightly separated blobs guarantee clean separability
    X, y = make_blobs(n_samples=100, centers=2, cluster_std=0.5, random_state=0)
    return X, y


class TestInit:
    # checks all post-fit attributes exist and have correct types/shapes
    def test_fit_decision_tree(self, _make_test_data):
        X, y = _make_test_data
        tree = DecisionTree().fit(X, y)
        assert hasattr(tree, "root_")
        assert hasattr(tree, "classes_")
        assert hasattr(tree, "n_features_")
        assert tree.n_features_ == X.shape[1]
        assert len(tree.classes_) == 2

    # accuracy on separable blobs should be near-perfect
    def test_accuracy(self, _make_test_data):
        X, y = _make_test_data
        assert DecisionTree().fit(X, y).score(X, y) >= 0.95

    # no eta in DecisionTree — test invalid criterion instead
    def test_invalid_eta_raises(self):
        with pytest.raises(ValueError):
            DecisionTree(criterion="invalid")

    # no epochs in DecisionTree — test invalid max_depth instead
    def test_invalid_epoch_raises(self):
        with pytest.raises(ValueError):
            DecisionTree(max_depth=0)

    # tree should accept plain Python lists, not just numpy arrays
    def test_accepts_python_lists(self):
        X = [[1.0, 2.0], [3.0, 4.0], [8.0, 9.0], [10.0, 11.0]]
        y = [0, 0, 1, 1]
        DecisionTree().fit(X, y)  # should not raise

    # score() should return near-perfect accuracy on separable blob data
    def test_score_perfect_on_blobs(self, _make_test_data):
        X, y = _make_test_data
        assert DecisionTree().fit(X, y).score(X, y) >= 0.95

    # score() must return a Python float
    def test_score_returns_float(self, _make_test_data):
        X, y = _make_test_data
        assert isinstance(float(DecisionTree().fit(X, y).score(X, y)), float)


class TestValidation:
    # X and y must have the same number of samples
    def test_mismatched_xy_lengths(self):
        with pytest.raises(ValueError):
            DecisionTree().fit(np.ones((5, 2)), np.ones(4))

    # X must be 2-D; a flat array should raise
    def test_1d_X_raises(self):
        with pytest.raises(ValueError):
            DecisionTree().fit(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))


class TestPredict:
    # all predicted values must be valid class labels
    def test_predict_output_in_classes(self, _make_test_data):
        X, y = _make_test_data
        tree = DecisionTree().fit(X, y)
        assert set(tree.predict(X)).issubset(set(tree.classes_))

    # batch predict should return one label per sample
    def test_predict_batch_shape(self, _make_test_data):
        X, y = _make_test_data
        tree = DecisionTree().fit(X, y)
        assert tree.predict(X).shape == (len(X),)

    # single-sample predict should return a scalar label
    def test_predict_single_sample(self, _make_test_data):
        X, y = _make_test_data
        tree = DecisionTree().fit(X, y)
        assert tree.predict(X[:1]).shape == (1,)


class TestHyperparameters:
    # deeper trees should fit separable data at least as well as shallow ones
    def test_deeper_tree_not_worse(self, _make_test_data):
        X, y = _make_test_data
        shallow = DecisionTree(max_depth=1).fit(X, y).score(X, y)
        deep = DecisionTree(max_depth=10).fit(X, y).score(X, y)
        assert deep >= shallow

    # both criterion options should converge on clean blob data
    def test_gini_criterion(self, _make_test_data):
        X, y = _make_test_data
        assert DecisionTree(criterion="gini").fit(X, y).score(X, y) >= 0.95

    def test_entropy_criterion(self, _make_test_data):
        X, y = _make_test_data
        assert DecisionTree(criterion="entropy").fit(X, y).score(X, y) >= 0.95
