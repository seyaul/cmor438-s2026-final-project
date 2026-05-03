"""Unit tests for AdaBoost."""

import numpy as np
import pytest
from rice_ml.supervised_ml.ensembles.adaboost import AdaBoost


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
        ab = AdaBoost()
        assert ab.n_estimators == 50
        assert ab.learning_rate == 1.0
        assert ab.random_state is None

    def test_custom_params(self):
        ab = AdaBoost(n_estimators=10, learning_rate=0.5, random_state=42)
        assert ab.n_estimators == 10
        assert ab.learning_rate == 0.5

    def test_invalid_n_estimators(self):
        with pytest.raises(ValueError, match="n_estimators"):
            AdaBoost(n_estimators=0)

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate"):
            AdaBoost(learning_rate=-0.1)


# ===========================================================================
# TestFit
# ===========================================================================

class TestFit:
    def test_returns_self(self, binary_blobs):
        X, y = binary_blobs
        ab = AdaBoost(n_estimators=5, random_state=0)
        assert ab.fit(X, y) is ab

    def test_sets_estimators(self, binary_blobs):
        X, y = binary_blobs
        ab = AdaBoost(n_estimators=7, random_state=0).fit(X, y)
        assert len(ab.estimators_) == 7
        assert len(ab.alphas_) == 7

    def test_sets_classes(self, binary_blobs):
        X, y = binary_blobs
        ab = AdaBoost(n_estimators=5, random_state=0).fit(X, y)
        np.testing.assert_array_equal(ab.classes_, [0, 1])

    def test_1d_X_raises(self, binary_blobs):
        _, y = binary_blobs
        with pytest.raises(ValueError, match="2-D"):
            AdaBoost(n_estimators=3).fit(np.array([1, 2, 3, 4, 5, 6]), y)

    def test_length_mismatch_raises(self, binary_blobs):
        X, y = binary_blobs
        with pytest.raises(ValueError):
            AdaBoost(n_estimators=3).fit(X, y[:3])


# ===========================================================================
# TestPredict
# ===========================================================================

class TestPredict:
    def test_output_shape(self, binary_blobs):
        X, y = binary_blobs
        ab = AdaBoost(n_estimators=10, random_state=0).fit(X, y)
        assert ab.predict(X).shape == (len(y),)

    def test_labels_in_training_set(self, binary_blobs):
        X, y = binary_blobs
        ab = AdaBoost(n_estimators=10, random_state=0).fit(X, y)
        assert set(ab.predict(X)).issubset({0, 1})

    def test_high_accuracy_separable(self, binary_blobs):
        X, y = binary_blobs
        ab = AdaBoost(n_estimators=20, random_state=0).fit(X, y)
        assert ab.score(X, y) == pytest.approx(1.0)

    def test_three_class(self, three_class_data):
        X, y = three_class_data
        ab = AdaBoost(n_estimators=30, random_state=0).fit(X, y)
        assert ab.score(X, y) >= 0.9

    def test_reproducible(self, binary_blobs):
        X, y = binary_blobs
        p1 = AdaBoost(n_estimators=10, random_state=7).fit(X, y).predict(X)
        p2 = AdaBoost(n_estimators=10, random_state=7).fit(X, y).predict(X)
        np.testing.assert_array_equal(p1, p2)


# ===========================================================================
# TestPredictProba
# ===========================================================================

class TestPredictProba:
    def test_shape_binary(self, binary_blobs):
        X, y = binary_blobs
        ab = AdaBoost(n_estimators=10, random_state=0).fit(X, y)
        assert ab.predict_proba(X).shape == (len(y), 2)

    def test_rows_sum_to_one(self, binary_blobs):
        X, y = binary_blobs
        ab = AdaBoost(n_estimators=10, random_state=0).fit(X, y)
        np.testing.assert_allclose(ab.predict_proba(X).sum(axis=1), np.ones(len(y)))

    def test_values_in_unit_interval(self, binary_blobs):
        X, y = binary_blobs
        proba = AdaBoost(n_estimators=10, random_state=0).fit(X, y).predict_proba(X)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


# ===========================================================================
# TestScore
# ===========================================================================

class TestScore:
    def test_score_in_unit_interval(self, binary_blobs):
        X, y = binary_blobs
        ab = AdaBoost(n_estimators=10, random_state=0).fit(X, y)
        assert 0.0 <= ab.score(X, y) <= 1.0
