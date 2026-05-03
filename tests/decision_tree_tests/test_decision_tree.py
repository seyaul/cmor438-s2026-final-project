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
    # TODO: fill in once __init__ is implemented

    def test_default_max_depth(self):
        raise NotImplementedError

    def test_default_min_samples_split(self):
        raise NotImplementedError

    def test_default_criterion(self):
        raise NotImplementedError

    def test_invalid_criterion_raises(self):
        raise NotImplementedError

    def test_invalid_min_samples_split_raises(self):
        raise NotImplementedError

    def test_no_root_before_fit(self):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# fit — attribute setup
# ---------------------------------------------------------------------------

class TestFitAttributes:
    # TODO: fill in once fit() is implemented

    def test_returns_self(self, separable_2d):
        raise NotImplementedError

    def test_root_set_after_fit(self, separable_2d):
        raise NotImplementedError

    def test_classes_set(self, separable_2d):
        raise NotImplementedError

    def test_n_features_set(self, separable_2d):
        raise NotImplementedError

    def test_accepts_python_lists(self):
        raise NotImplementedError

    def test_mismatched_xy_raises(self):
        raise NotImplementedError

    def test_1d_X_raises(self):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestPredict:
    # TODO: fill in once predict() is implemented

    def test_predict_batch_shape(self, trained_tree):
        raise NotImplementedError

    def test_predict_values_in_classes(self, trained_tree):
        raise NotImplementedError

    def test_predict_single_sample(self, trained_tree):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# fit — convergence and correctness
# ---------------------------------------------------------------------------

class TestFitConvergence:
    # TODO: fill in once fit() and predict() are implemented

    def test_converges_on_separable_data(self, separable_2d):
        raise NotImplementedError

    def test_and_gate(self):
        raise NotImplementedError

    def test_or_gate(self):
        raise NotImplementedError

    def test_depth_1_single_split(self, separable_2d):
        raise NotImplementedError

    def test_pure_node_stops_early(self):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

class TestScore:
    # TODO: fill in once score() is implemented

    def test_score_returns_float(self, trained_tree):
        raise NotImplementedError

    def test_score_perfect_on_separable(self, separable_2d):
        raise NotImplementedError

    def test_score_between_0_and_1(self, trained_tree):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# hyperparameter effects
# ---------------------------------------------------------------------------

class TestHyperparameters:
    # TODO: fill in once fit() is implemented

    def test_deeper_tree_fits_better(self, separable_2d):
        raise NotImplementedError

    def test_max_depth_respected(self, separable_2d):
        raise NotImplementedError

    def test_min_samples_split_prevents_split(self):
        raise NotImplementedError

    def test_gini_and_entropy_both_converge(self, separable_2d):
        raise NotImplementedError
