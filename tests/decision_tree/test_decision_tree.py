"""
Unit tests for DecisionTree — binary classifier with gini/entropy splitting.
"""

import numpy as np
import pytest
from rice_Ml.supervised_ml.DecisionTree.decision_tree import DecisionTree


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_blobs():
    """Two clearly separated 2-D clusters."""
    X = np.array([
        [1.0, 1.0], [1.5, 1.2], [0.8, 1.1],
        [8.0, 8.0], [8.5, 7.8], [7.9, 8.2],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def three_class_data():
    """Three separable Gaussian clusters."""
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal([0, 0], 0.3, (20, 2)),
        rng.normal([5, 0], 0.3, (20, 2)),
        rng.normal([2.5, 5], 0.3, (20, 2)),
    ])
    y = np.repeat([0, 1, 2], 20)
    return X, y


# ===========================================================================
# TestInit
# ===========================================================================

class TestInit:
    def test_defaults(self):
        dt = DecisionTree()
        assert dt.max_depth is None
        assert dt.min_samples_split == 2
        assert dt.criterion == "gini"

    def test_custom_params(self):
        dt = DecisionTree(max_depth=5, min_samples_split=4, criterion="entropy")
        assert dt.max_depth == 5
        assert dt.min_samples_split == 4
        assert dt.criterion == "entropy"

    def test_invalid_max_depth_zero(self):
        with pytest.raises(ValueError, match="max depth"):
            DecisionTree(max_depth=0)

    def test_invalid_max_depth_negative(self):
        with pytest.raises(ValueError, match="max depth"):
            DecisionTree(max_depth=-3)

    def test_invalid_max_depth_float(self):
        with pytest.raises(ValueError, match="max depth"):
            DecisionTree(max_depth=2.5)

    def test_invalid_min_samples_split_too_small(self):
        with pytest.raises(ValueError, match="minimum samples"):
            DecisionTree(min_samples_split=1)

    def test_invalid_min_samples_split_negative(self):
        with pytest.raises(ValueError, match="minimum samples"):
            DecisionTree(min_samples_split=-1)

    def test_invalid_criterion(self):
        with pytest.raises(ValueError, match="criterion"):
            DecisionTree(criterion="mse")

    def test_no_root_before_fit(self):
        assert not hasattr(DecisionTree(), "root_")


# ===========================================================================
# TestFit
# ===========================================================================

class TestFit:
    def test_sets_classes_binary(self, binary_blobs):
        X, y = binary_blobs
        dt = DecisionTree().fit(X, y)
        np.testing.assert_array_equal(dt.classes_, [0, 1])

    def test_sets_classes_three_class(self, three_class_data):
        X, y = three_class_data
        dt = DecisionTree().fit(X, y)
        np.testing.assert_array_equal(dt.classes_, [0, 1, 2])

    def test_sets_n_features(self, binary_blobs):
        X, y = binary_blobs
        dt = DecisionTree().fit(X, y)
        assert dt.n_features_ == 2

    def test_sets_root(self, binary_blobs):
        X, y = binary_blobs
        dt = DecisionTree().fit(X, y)
        assert dt.root_ is not None

    def test_1d_X_raises(self, binary_blobs):
        _, y = binary_blobs
        with pytest.raises(ValueError, match="2-D"):
            DecisionTree().fit(np.array([1, 2, 3, 4, 5, 6]), y)

    def test_length_mismatch_raises(self, binary_blobs):
        X, y = binary_blobs
        with pytest.raises(ValueError):
            DecisionTree().fit(X, y[:3])

    def test_returns_self(self, binary_blobs):
        X, y = binary_blobs
        dt = DecisionTree()
        assert dt.fit(X, y) is dt

    def test_accepts_python_lists(self):
        X = [[1.0, 1.0], [1.5, 1.0], [8.0, 8.0], [8.5, 8.0]]
        y = [0, 0, 1, 1]
        DecisionTree().fit(X, y)  # should not raise

    def test_pure_node_stops_early(self):
        X = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=float)
        y = np.array([0, 0])
        tree = DecisionTree().fit(X, y)
        assert tree.root_.is_leaf()


# ===========================================================================
# TestPredict
# ===========================================================================

class TestPredict:
    def test_perfect_separation_gini(self, binary_blobs):
        X, y = binary_blobs
        dt = DecisionTree(criterion="gini").fit(X, y)
        np.testing.assert_array_equal(dt.predict(X), y)

    def test_perfect_separation_entropy(self, binary_blobs):
        X, y = binary_blobs
        dt = DecisionTree(criterion="entropy").fit(X, y)
        np.testing.assert_array_equal(dt.predict(X), y)

    def test_output_shape(self, binary_blobs):
        X, y = binary_blobs
        dt = DecisionTree().fit(X, y)
        assert dt.predict(X).shape == (len(y),)

    def test_labels_within_training_set(self, binary_blobs):
        X, y = binary_blobs
        dt = DecisionTree().fit(X, y)
        assert set(dt.predict(X)).issubset({0, 1})

    def test_three_class_perfect(self, three_class_data):
        X, y = three_class_data
        dt = DecisionTree().fit(X, y)
        np.testing.assert_array_equal(dt.predict(X), y)

    def test_max_depth_respected(self, binary_blobs):
        X, y = binary_blobs
        dt = DecisionTree(max_depth=1).fit(X, y)
        preds = dt.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_single_sample(self, binary_blobs):
        X, y = binary_blobs
        dt = DecisionTree().fit(X, y)
        assert dt.predict(X[:1]).shape == (1,)

    def test_and_gate(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 0, 0, 1])
        assert DecisionTree().fit(X, y).score(X, y) == pytest.approx(1.0)

    def test_or_gate(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 1])
        assert DecisionTree().fit(X, y).score(X, y) == pytest.approx(1.0)


# ===========================================================================
# TestScore
# ===========================================================================

class TestScore:
    def test_perfect_score_on_training_data(self, binary_blobs):
        X, y = binary_blobs
        dt = DecisionTree().fit(X, y)
        assert dt.score(X, y) == pytest.approx(1.0)

    def test_score_in_unit_interval(self, binary_blobs):
        X, y = binary_blobs
        dt = DecisionTree(max_depth=1).fit(X, y)
        assert 0.0 <= dt.score(X, y) <= 1.0

    def test_score_matches_manual_accuracy(self, binary_blobs):
        X, y = binary_blobs
        dt = DecisionTree().fit(X, y)
        preds = dt.predict(X)
        expected = np.mean(preds == y)
        assert dt.score(X, y) == pytest.approx(expected)

    def test_score_returns_float(self, binary_blobs):
        X, y = binary_blobs
        dt = DecisionTree().fit(X, y)
        assert isinstance(float(dt.score(X, y)), float)


# ===========================================================================
# TestImpurity
# ===========================================================================

class TestImpurity:
    def test_pure_node_gini_zero(self):
        dt = DecisionTree(criterion="gini")
        assert dt._impurity(np.array([1, 1, 1])) == pytest.approx(0.0)

    def test_pure_node_entropy_zero(self):
        dt = DecisionTree(criterion="entropy")
        assert dt._impurity(np.array([0, 0, 0])) == pytest.approx(0.0)

    def test_balanced_binary_gini(self):
        # gini = 1 - (0.5^2 + 0.5^2) = 0.5
        dt = DecisionTree(criterion="gini")
        assert dt._impurity(np.array([0, 0, 1, 1])) == pytest.approx(0.5)

    def test_balanced_binary_entropy(self):
        # entropy = -2 * (0.5 * log2(0.5)) = 1.0
        dt = DecisionTree(criterion="entropy")
        assert dt._impurity(np.array([0, 0, 1, 1])) == pytest.approx(1.0)

    def test_gini_max_at_uniform_binary(self):
        dt = DecisionTree(criterion="gini")
        assert dt._impurity(np.array([0, 1])) == pytest.approx(0.5)

    def test_gini_three_class_uniform(self):
        # gini = 1 - 3*(1/3)^2 = 1 - 1/3 = 2/3
        dt = DecisionTree(criterion="gini")
        assert dt._impurity(np.array([0, 1, 2])) == pytest.approx(2 / 3)


# ===========================================================================
# TestMajorityClass
# ===========================================================================

class TestMajorityClass:
    def test_clear_majority(self):
        dt = DecisionTree()
        assert dt._majority_class(np.array([0, 0, 0, 1])) == 0

    def test_all_same_label(self):
        dt = DecisionTree()
        assert dt._majority_class(np.array([2, 2, 2])) == 2

    def test_three_class_majority(self):
        dt = DecisionTree()
        assert dt._majority_class(np.array([1, 1, 2, 2, 2, 0])) == 2


# ===========================================================================
# TestBestSplit
# ===========================================================================

class TestBestSplit:
    def test_finds_correct_feature(self):
        # Feature 0 perfectly separates; feature 1 is noise
        X = np.array([[1.0, 5.0], [2.0, 3.0], [8.0, 4.0], [9.0, 6.0]])
        y = np.array([0, 0, 1, 1])
        feature, _ = DecisionTree()._best_split(X, y)
        assert feature == 0

    def test_threshold_partitions_classes(self):
        X = np.array([[1.0], [2.0], [8.0], [9.0]])
        y = np.array([0, 0, 1, 1])
        feature, thresh = DecisionTree()._best_split(X, y)
        assert feature == 0
        assert 2.0 <= thresh < 8.0

    def test_returns_none_when_no_valid_split(self):
        # All feature values identical — every threshold produces an empty child
        X = np.ones((4, 2))
        y = np.array([0, 0, 1, 1])
        feature, thresh = DecisionTree()._best_split(X, y)
        assert feature is None
        assert thresh is None

    def test_gain_favours_purer_split(self):
        # Two candidate splits: one perfect, one imperfect
        X = np.array([[1.0], [2.0], [3.0], [10.0]])
        y = np.array([0, 0, 0, 1])
        _, thresh = DecisionTree()._best_split(X, y)
        # Best split should put 10.0 alone on the right
        assert thresh < 10.0


# ===========================================================================
# TestHyperparameters
# ===========================================================================

class TestHyperparameters:
    def test_deeper_tree_fits_better(self, binary_blobs):
        X, y = binary_blobs
        shallow = DecisionTree(max_depth=1).fit(X, y).score(X, y)
        deep = DecisionTree(max_depth=10).fit(X, y).score(X, y)
        assert deep >= shallow

    def test_min_samples_split_prevents_split(self):
        X = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTree(min_samples_split=10).fit(X, y)
        assert tree.root_.is_leaf()

    def test_gini_and_entropy_both_converge(self, binary_blobs):
        X, y = binary_blobs
        assert DecisionTree(criterion="gini").fit(X, y).score(X, y) == pytest.approx(1.0)
        assert DecisionTree(criterion="entropy").fit(X, y).score(X, y) == pytest.approx(1.0)
