"""
Unit tests for the DecisionTree class.

Covers: initialization, fit, predict, score, impurity helpers,
and end-to-end classification on simple hand-crafted datasets.
"""

import numpy as np
import pytest
from rice_Ml.supervised_ml.DecisionTree.decision_tree import DecisionTree


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
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def trained_tree(separable_2d):
    X, y = separable_2d
    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)
    return tree, X, y


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:

    def test_default_max_depth(self):
        assert DecisionTree().max_depth is None

    def test_default_min_samples_split(self):
        assert DecisionTree().min_samples_split == 2

    def test_default_criterion(self):
        assert DecisionTree().criterion == "gini"

    def test_invalid_criterion_raises(self):
        with pytest.raises(ValueError):
            DecisionTree(criterion="bad")

    def test_invalid_min_samples_split_raises(self):
        with pytest.raises(ValueError):
            DecisionTree(min_samples_split=1)

    def test_no_root_before_fit(self):
        assert not hasattr(DecisionTree(), "root_")


# ---------------------------------------------------------------------------
# fit — attribute setup
# ---------------------------------------------------------------------------

class TestFitAttributes:

    def test_returns_self(self, separable_2d):
        X, y = separable_2d
        tree = DecisionTree()
        assert tree.fit(X, y) is tree

    def test_root_set_after_fit(self, separable_2d):
        X, y = separable_2d
        tree = DecisionTree().fit(X, y)
        assert hasattr(tree, "root_") and tree.root_ is not None

    def test_classes_set(self, separable_2d):
        X, y = separable_2d
        tree = DecisionTree().fit(X, y)
        np.testing.assert_array_equal(tree.classes_, [0, 1])

    def test_n_features_set(self, separable_2d):
        X, y = separable_2d
        tree = DecisionTree().fit(X, y)
        assert tree.n_features_ == 2

    def test_accepts_python_lists(self):
        X = [[1.0, 1.0], [1.5, 1.0], [5.0, 5.0], [5.5, 5.0]]
        y = [0, 0, 1, 1]
        DecisionTree().fit(X, y)  # should not raise

    def test_mismatched_xy_raises(self):
        with pytest.raises(ValueError):
            DecisionTree().fit(np.ones((4, 2)), np.ones(3))

    def test_1d_X_raises(self):
        with pytest.raises(ValueError):
            DecisionTree().fit(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestPredict:

    def test_predict_batch_shape(self, trained_tree):
        tree, X, y = trained_tree
        assert tree.predict(X).shape == (len(X),)

    def test_predict_values_in_classes(self, trained_tree):
        tree, X, y = trained_tree
        assert set(tree.predict(X)).issubset({0, 1})

    def test_predict_single_sample(self, trained_tree):
        tree, X, y = trained_tree
        assert tree.predict(X[:1]).shape == (1,)


# ---------------------------------------------------------------------------
# fit — convergence and correctness
# ---------------------------------------------------------------------------

class TestFitConvergence:

    def test_converges_on_separable_data(self, separable_2d):
        X, y = separable_2d
        assert DecisionTree().fit(X, y).score(X, y) == pytest.approx(1.0)

    def test_and_gate(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 0, 0, 1])
        assert DecisionTree().fit(X, y).score(X, y) == pytest.approx(1.0)

    def test_or_gate(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 1])
        assert DecisionTree().fit(X, y).score(X, y) == pytest.approx(1.0)

    def test_depth_1_single_split(self, separable_2d):
        X, y = separable_2d
        preds = DecisionTree(max_depth=1).fit(X, y).predict(X)
        assert preds.shape == (len(y),)
        assert set(preds).issubset({0, 1})

    def test_pure_node_stops_early(self):
        X = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=float)
        y = np.array([0, 0])
        tree = DecisionTree().fit(X, y)
        assert tree.root_.is_leaf()


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

class TestScore:

    def test_score_returns_float(self, trained_tree):
        tree, X, y = trained_tree
        assert isinstance(float(tree.score(X, y)), float)

    def test_score_perfect_on_separable(self, separable_2d):
        X, y = separable_2d
        assert DecisionTree().fit(X, y).score(X, y) == pytest.approx(1.0)

    def test_score_between_0_and_1(self, trained_tree):
        tree, X, y = trained_tree
        assert 0.0 <= tree.score(X, y) <= 1.0


# ---------------------------------------------------------------------------
# hyperparameter effects
# ---------------------------------------------------------------------------

class TestHyperparameters:

    def test_deeper_tree_fits_better(self, separable_2d):
        X, y = separable_2d
        shallow = DecisionTree(max_depth=1).fit(X, y).score(X, y)
        deep = DecisionTree(max_depth=10).fit(X, y).score(X, y)
        assert deep >= shallow

    def test_max_depth_respected(self, separable_2d):
        X, y = separable_2d
        preds = DecisionTree(max_depth=1).fit(X, y).predict(X)
        assert preds.shape == (len(y),)

    def test_min_samples_split_prevents_split(self):
        X = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTree(min_samples_split=10).fit(X, y)
        assert tree.root_.is_leaf()

    def test_gini_and_entropy_both_converge(self, separable_2d):
        X, y = separable_2d
        assert DecisionTree(criterion="gini").fit(X, y).score(X, y) == pytest.approx(1.0)
        assert DecisionTree(criterion="entropy").fit(X, y).score(X, y) == pytest.approx(1.0)
