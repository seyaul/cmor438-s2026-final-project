"""Unit tests for RandomForest."""

import numpy as np
import pytest
from rice_Ml.supervised_ml.ensembles.random_forest import RandomForest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_blobs():
    X = np.array([
        [1.0, 1.0], [1.5, 1.2], [0.8, 1.1],
        [8.0, 8.0], [8.5, 7.8], [7.9, 8.2],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def three_class_data():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal([0, 0], 0.3, (30, 2)),
        rng.normal([5, 0], 0.3, (30, 2)),
        rng.normal([2.5, 5], 0.3, (30, 2)),
    ])
    y = np.repeat([0, 1, 2], 30)
    return X, y


# ===========================================================================
# TestInit
# ===========================================================================

class TestInit:
    def test_defaults(self):
        rf = RandomForest()
        assert rf.n_estimators == 100
        assert rf.max_depth is None
        assert rf.min_samples_split == 2
        assert rf.max_features == "sqrt"
        assert rf.criterion == "gini"
        assert rf.random_state is None

    def test_custom_params(self):
        rf = RandomForest(n_estimators=10, max_depth=3, criterion="entropy", random_state=42)
        assert rf.n_estimators == 10
        assert rf.max_depth == 3
        assert rf.criterion == "entropy"
        assert rf.random_state == 42

    def test_invalid_n_estimators_zero(self):
        with pytest.raises(ValueError, match="n_estimators"):
            RandomForest(n_estimators=0)

    def test_invalid_n_estimators_negative(self):
        with pytest.raises(ValueError, match="n_estimators"):
            RandomForest(n_estimators=-5)


# ===========================================================================
# TestFit
# ===========================================================================

class TestFit:
    def test_returns_self(self, binary_blobs):
        X, y = binary_blobs
        rf = RandomForest(n_estimators=5, random_state=0)
        assert rf.fit(X, y) is rf

    def test_sets_estimators(self, binary_blobs):
        X, y = binary_blobs
        rf = RandomForest(n_estimators=7, random_state=0).fit(X, y)
        assert len(rf.estimators_) == 7

    def test_sets_classes(self, binary_blobs):
        X, y = binary_blobs
        rf = RandomForest(n_estimators=5, random_state=0).fit(X, y)
        np.testing.assert_array_equal(rf.classes_, [0, 1])

    def test_sets_n_features(self, binary_blobs):
        X, y = binary_blobs
        rf = RandomForest(n_estimators=5, random_state=0).fit(X, y)
        assert rf.n_features_ == 2

    def test_1d_X_raises(self, binary_blobs):
        _, y = binary_blobs
        with pytest.raises(ValueError, match="2-D"):
            RandomForest(n_estimators=3).fit(np.array([1, 2, 3, 4, 5, 6]), y)

    def test_length_mismatch_raises(self, binary_blobs):
        X, y = binary_blobs
        with pytest.raises(ValueError):
            RandomForest(n_estimators=3).fit(X, y[:3])


# ===========================================================================
# TestPredict
# ===========================================================================

class TestPredict:
    def test_output_shape(self, binary_blobs):
        X, y = binary_blobs
        rf = RandomForest(n_estimators=10, random_state=0).fit(X, y)
        assert rf.predict(X).shape == (len(y),)

    def test_labels_in_training_set(self, binary_blobs):
        X, y = binary_blobs
        rf = RandomForest(n_estimators=10, random_state=0).fit(X, y)
        assert set(rf.predict(X)).issubset({0, 1})

    def test_high_accuracy_on_separable_data(self, binary_blobs):
        X, y = binary_blobs
        rf = RandomForest(n_estimators=20, random_state=0).fit(X, y)
        assert rf.score(X, y) == pytest.approx(1.0)

    def test_three_class_high_accuracy(self, three_class_data):
        X, y = three_class_data
        rf = RandomForest(n_estimators=20, random_state=0).fit(X, y)
        assert rf.score(X, y) >= 0.9

    def test_reproducible_with_same_seed(self, binary_blobs):
        X, y = binary_blobs
        preds1 = RandomForest(n_estimators=10, random_state=7).fit(X, y).predict(X)
        preds2 = RandomForest(n_estimators=10, random_state=7).fit(X, y).predict(X)
        np.testing.assert_array_equal(preds1, preds2)

    def test_different_seeds_may_differ(self, three_class_data):
        X, y = three_class_data
        preds1 = RandomForest(n_estimators=5, random_state=1).fit(X, y).predict(X)
        preds2 = RandomForest(n_estimators=5, random_state=99).fit(X, y).predict(X)
        # Not guaranteed to differ, but with enough data they almost always will
        # Just check shapes match
        assert preds1.shape == preds2.shape


# ===========================================================================
# TestPredictProba
# ===========================================================================

class TestPredictProba:
    def test_output_shape_binary(self, binary_blobs):
        X, y = binary_blobs
        rf = RandomForest(n_estimators=10, random_state=0).fit(X, y)
        proba = rf.predict_proba(X)
        assert proba.shape == (len(y), 2)

    def test_output_shape_three_class(self, three_class_data):
        X, y = three_class_data
        rf = RandomForest(n_estimators=10, random_state=0).fit(X, y)
        proba = rf.predict_proba(X)
        assert proba.shape == (len(y), 3)

    def test_rows_sum_to_one(self, binary_blobs):
        X, y = binary_blobs
        rf = RandomForest(n_estimators=10, random_state=0).fit(X, y)
        proba = rf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(y)))

    def test_values_in_unit_interval(self, binary_blobs):
        X, y = binary_blobs
        rf = RandomForest(n_estimators=10, random_state=0).fit(X, y)
        proba = rf.predict_proba(X)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_argmax_matches_predict(self, three_class_data):
        X, y = three_class_data
        rf = RandomForest(n_estimators=20, random_state=0).fit(X, y)
        proba = rf.predict_proba(X)
        proba_preds = rf.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(proba_preds, rf.predict(X))


# ===========================================================================
# TestScore
# ===========================================================================

class TestScore:
    def test_score_in_unit_interval(self, binary_blobs):
        X, y = binary_blobs
        rf = RandomForest(n_estimators=10, random_state=0).fit(X, y)
        assert 0.0 <= rf.score(X, y) <= 1.0

    def test_perfect_score_separable(self, binary_blobs):
        X, y = binary_blobs
        rf = RandomForest(n_estimators=20, random_state=0).fit(X, y)
        assert rf.score(X, y) == pytest.approx(1.0)


# ===========================================================================
# TestMaxFeatures
# ===========================================================================

class TestMaxFeatures:
    def test_sqrt_max_features(self, three_class_data):
        X, y = three_class_data
        rf = RandomForest(n_estimators=10, max_features="sqrt", random_state=0).fit(X, y)
        assert rf.score(X, y) >= 0.5

    def test_log2_max_features(self, three_class_data):
        X, y = three_class_data
        rf = RandomForest(n_estimators=10, max_features="log2", random_state=0).fit(X, y)
        assert rf.score(X, y) >= 0.5

    def test_int_max_features(self, three_class_data):
        X, y = three_class_data
        rf = RandomForest(n_estimators=10, max_features=1, random_state=0).fit(X, y)
        assert rf.score(X, y) >= 0.5

    def test_none_max_features_uses_all(self, binary_blobs):
        X, y = binary_blobs
        rf = RandomForest(n_estimators=10, max_features=None, random_state=0).fit(X, y)
        assert rf.score(X, y) == pytest.approx(1.0)
