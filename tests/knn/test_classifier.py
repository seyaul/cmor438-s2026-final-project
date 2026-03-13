"""Unit tests for KNNClassifier."""

import numpy as np
import pytest

from rice_Ml.supervised_ml.knn.classifier import KNNClassifier

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    """Two clearly separated clusters in 2D."""
    X = np.array([
        [0.0, 0.0], [0.5, 0.0], [0.0, 0.5],   # class 0
        [5.0, 5.0], [5.5, 5.0], [5.0, 5.5],   # class 1
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def multiclass_data():
    """Three clusters — classes 0, 1, 2."""
    X = np.array([
        [0.0, 0.0], [0.1, 0.1],    # class 0
        [5.0, 0.0], [5.1, 0.1],    # class 1
        [0.0, 5.0], [0.1, 5.1],    # class 2
    ])
    y = np.array([0, 0, 1, 1, 2, 2])
    return X, y


@pytest.fixture
def fitted_binary(binary_data):
    X, y = binary_data
    return KNNClassifier(k=3).fit(X, y)


# ===========================================================================
# Initialisation
# ===========================================================================

class TestInit:
    def test_default_params(self):
        clf = KNNClassifier()
        assert clf.k == 5
        assert clf.metric == "euclidean"

    def test_custom_k_and_metric(self):
        clf = KNNClassifier(k=3, metric="taxicab")
        assert clf.k == 3
        assert clf.metric == "taxicab"

    def test_callable_metric(self):
        fn = lambda u, v: float(np.sum(np.abs(np.asarray(u) - np.asarray(v))))
        clf = KNNClassifier(k=1, metric=fn)
        assert clf._distance_fn is fn

    def test_invalid_k_zero(self):
        with pytest.raises(ValueError):
            KNNClassifier(k=0)

    def test_invalid_k_negative(self):
        with pytest.raises(ValueError):
            KNNClassifier(k=-1)

    def test_invalid_k_float(self):
        with pytest.raises(ValueError):
            KNNClassifier(k=2.5)

    def test_invalid_metric_string(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            KNNClassifier(metric="cosine")

    def test_invalid_metric_type(self):
        with pytest.raises(TypeError):
            KNNClassifier(metric=42)


# ===========================================================================
# fit
# ===========================================================================

class TestFit:
    def test_returns_self(self, binary_data):
        X, y = binary_data
        clf = KNNClassifier(k=3)
        assert clf.fit(X, y) is clf

    def test_sets_attributes(self, binary_data):
        X, y = binary_data
        clf = KNNClassifier(k=3).fit(X, y)
        assert clf.n_features_in_ == 2
        np.testing.assert_array_equal(clf.classes_, [0, 1])
        assert clf._is_fitted

    def test_string_labels(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array(["cat", "dog", "cat"])
        clf = KNNClassifier(k=1).fit(X, y)
        assert set(clf.classes_) == {"cat", "dog"}

    def test_k_larger_than_samples(self, binary_data):
        X, y = binary_data
        with pytest.raises(ValueError, match="exceeds"):
            KNNClassifier(k=100).fit(X, y)

    def test_X_not_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            KNNClassifier(k=1).fit([1, 2, 3], [0, 1, 0])

    def test_X_contains_nan(self):
        X = np.array([[1.0, float("nan")], [2.0, 3.0]])
        with pytest.raises(ValueError, match="NaN"):
            KNNClassifier(k=1).fit(X, [0, 1])

    def test_y_wrong_length(self, binary_data):
        X, _ = binary_data
        with pytest.raises(ValueError, match="same number of samples"):
            KNNClassifier(k=3).fit(X, [0, 1])

    def test_y_not_1d(self, binary_data):
        X, _ = binary_data
        with pytest.raises(ValueError, match="1-D"):
            KNNClassifier(k=1).fit(X, np.zeros((6, 2)))

    def test_X_none(self):
        with pytest.raises(TypeError):
            KNNClassifier(k=1).fit(None, [0])


# ===========================================================================
# predict
# ===========================================================================

class TestPredict:
    def test_correct_classification_cluster_0(self, fitted_binary):
        pred = fitted_binary.predict([[0.1, 0.1]])
        assert pred[0] == 0

    def test_correct_classification_cluster_1(self, fitted_binary):
        pred = fitted_binary.predict([[5.1, 5.1]])
        assert pred[0] == 1

    def test_batch_prediction(self, fitted_binary):
        X_test = np.array([[0.0, 0.0], [5.0, 5.0]])
        pred = fitted_binary.predict(X_test)
        np.testing.assert_array_equal(pred, [0, 1])

    def test_returns_ndarray(self, fitted_binary):
        assert isinstance(fitted_binary.predict([[0.0, 0.0]]), np.ndarray)

    def test_output_shape(self, fitted_binary):
        X_test = np.ones((7, 2)) * 0.1
        assert fitted_binary.predict(X_test).shape == (7,)

    def test_multiclass(self, multiclass_data):
        X, y = multiclass_data
        clf = KNNClassifier(k=1).fit(X, y)
        # Each training point should predict its own class
        np.testing.assert_array_equal(clf.predict(X), y)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            KNNClassifier().predict([[1.0, 2.0]])

    def test_wrong_feature_count(self, fitted_binary):
        with pytest.raises(ValueError, match="feature"):
            fitted_binary.predict([[1.0, 2.0, 3.0]])

    def test_taxicab_metric(self, binary_data):
        X, y = binary_data
        clf = KNNClassifier(k=3, metric="taxicab").fit(X, y)
        assert clf.predict([[0.1, 0.1]])[0] == 0


# ===========================================================================
# predict_proba
# ===========================================================================

class TestPredictProba:
    def test_shape(self, fitted_binary):
        proba = fitted_binary.predict_proba([[0.0, 0.0], [5.0, 5.0]])
        assert proba.shape == (2, 2)

    def test_rows_sum_to_one(self, fitted_binary):
        proba = fitted_binary.predict_proba(np.random.default_rng(0).random((10, 2)) * 6)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(10))

    def test_pure_cluster_probability(self, fitted_binary):
        # A point deep in cluster 0 should have proba ≈ [1, 0]
        proba = fitted_binary.predict_proba([[0.0, 0.0]])
        assert proba[0, 0] == pytest.approx(1.0)
        assert proba[0, 1] == pytest.approx(0.0)

    def test_column_order_matches_classes(self, binary_data):
        X, y = binary_data
        clf = KNNClassifier(k=3).fit(X, y)
        proba = clf.predict_proba([[0.0, 0.0]])
        # classes_ = [0, 1]; col 0 is P(class=0)
        assert proba[0, 0] > proba[0, 1]


# ===========================================================================
# score
# ===========================================================================

class TestScore:
    def test_perfect_score(self, binary_data):
        X, y = binary_data
        clf = KNNClassifier(k=3).fit(X, y)
        assert clf.score(X, y) == pytest.approx(1.0)

    def test_score_range(self, binary_data):
        X, y = binary_data
        clf = KNNClassifier(k=3).fit(X, y)
        s = clf.score(X, y)
        assert 0.0 <= s <= 1.0

    def test_score_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            KNNClassifier().score([[1.0, 2.0]], [0])


# ===========================================================================
# repr
# ===========================================================================

class TestRepr:
    def test_repr(self):
        r = repr(KNNClassifier(k=3, metric="taxicab"))
        assert "KNNClassifier" in r
        assert "k=3" in r
