"""Unit tests for KNNRegressor."""

import numpy as np
import pytest

from rice_Ml.supervised_ml.knn.regressor import KNNRegressor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_data():
    """y = x, so neighbours give the correct answer for uniform weighting."""
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = np.arange(10, dtype=float)
    return X, y


@pytest.fixture
def fitted_uniform(linear_data):
    X, y = linear_data
    return KNNRegressor(k=3, weights="uniform").fit(X, y)


@pytest.fixture
def fitted_distance(linear_data):
    X, y = linear_data
    return KNNRegressor(k=3, weights="distance").fit(X, y)


# ===========================================================================
# Initialisation
# ===========================================================================

class TestInit:
    def test_defaults(self):
        reg = KNNRegressor()
        assert reg.k == 5
        assert reg.metric == "euclidean"
        assert reg.weights == "uniform"

    def test_distance_weights(self):
        reg = KNNRegressor(weights="distance")
        assert reg.weights == "distance"

    def test_invalid_weights(self):
        with pytest.raises(ValueError, match="weights"):
            KNNRegressor(weights="gaussian")

    def test_callable_metric(self):
        fn = lambda u, v: float(np.linalg.norm(np.asarray(u) - np.asarray(v)))
        reg = KNNRegressor(k=2, metric=fn)
        assert reg._distance_fn is fn

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            KNNRegressor(k=0)

    def test_invalid_metric_string(self):
        with pytest.raises(ValueError):
            KNNRegressor(metric="l3")


# ===========================================================================
# fit
# ===========================================================================

class TestFit:
    def test_returns_self(self, linear_data):
        X, y = linear_data
        reg = KNNRegressor(k=2)
        assert reg.fit(X, y) is reg

    def test_sets_attributes(self, linear_data):
        X, y = linear_data
        reg = KNNRegressor(k=2).fit(X, y)
        assert reg.n_features_in_ == 1
        assert reg._is_fitted

    def test_k_exceeds_samples(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="exceeds"):
            KNNRegressor(k=10).fit(X, y)

    def test_y_non_finite(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, float("nan")])
        with pytest.raises(ValueError, match="NaN"):
            KNNRegressor(k=1).fit(X, y)

    def test_y_wrong_length(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="same number"):
            KNNRegressor(k=1).fit(X, y)

    def test_X_not_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            KNNRegressor(k=1).fit([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    def test_X_contains_inf(self):
        X = np.array([[float("inf")], [2.0]])
        with pytest.raises(ValueError, match="NaN or infinite"):
            KNNRegressor(k=1).fit(X, [1.0, 2.0])


# ===========================================================================
# predict — uniform weights
# ===========================================================================

class TestPredictUniform:
    def test_exact_training_point(self, fitted_uniform):
        # Query point coincides with training point x=5.0, y=5.0
        # k=3 → neighbours at 4, 5, 6 → mean = 5.0
        pred = fitted_uniform.predict([[5.0]])
        assert pred[0] == pytest.approx(5.0)

    def test_output_shape(self, fitted_uniform):
        assert fitted_uniform.predict(np.ones((7, 1))).shape == (7,)

    def test_returns_float64(self, fitted_uniform):
        pred = fitted_uniform.predict([[3.0]])
        assert pred.dtype == np.float64

    def test_non_negative_for_positive_targets(self, linear_data):
        X, y = linear_data
        reg = KNNRegressor(k=1).fit(X, y)
        pred = reg.predict([[0.0]])
        assert pred[0] >= 0.0

    def test_predict_before_fit(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            KNNRegressor().predict([[1.0]])

    def test_wrong_feature_count(self, fitted_uniform):
        with pytest.raises(ValueError, match="feature"):
            fitted_uniform.predict([[1.0, 2.0]])

    def test_k1_returns_nearest(self):
        X = np.array([[0.0], [1.0], [10.0]])
        y = np.array([0.0, 1.0, 100.0])
        reg = KNNRegressor(k=1).fit(X, y)
        assert reg.predict([[0.1]])[0] == pytest.approx(0.0)
        assert reg.predict([[9.9]])[0] == pytest.approx(100.0)

    def test_batch_prediction(self, linear_data):
        X, y = linear_data
        reg = KNNRegressor(k=1).fit(X, y)
        X_test = np.array([[0.0], [5.0], [9.0]])
        pred = reg.predict(X_test)
        np.testing.assert_allclose(pred, [0.0, 5.0, 9.0])

    def test_taxicab_metric(self, linear_data):
        X, y = linear_data
        reg = KNNRegressor(k=1, metric="taxicab").fit(X, y)
        pred = reg.predict([[3.0]])
        assert pred[0] == pytest.approx(3.0)


# ===========================================================================
# predict — distance weights
# ===========================================================================

class TestPredictDistance:
    def test_exact_match_returns_that_value(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([10.0, 20.0, 30.0])
        reg = KNNRegressor(k=3, weights="distance").fit(X, y)
        # Query exactly at training point x=1 → zero distance → return 20.0
        pred = reg.predict([[1.0]])
        assert pred[0] == pytest.approx(20.0)

    def test_closer_neighbour_has_more_influence(self):
        X = np.array([[0.0], [1.0], [100.0]])
        y = np.array([0.0, 1.0, 1000.0])
        reg = KNNRegressor(k=2, weights="distance").fit(X, y)
        # Query at 0.1 — much closer to x=0 (dist=0.1) than x=1 (dist=0.9)
        pred = reg.predict([[0.1]])
        # Weighted mean should be closer to 0.0 than to 1.0
        assert pred[0] < 0.5

    def test_output_shape(self):
        X = np.arange(10, dtype=float).reshape(-1, 1)
        y = np.arange(10, dtype=float)
        reg = KNNRegressor(k=2, weights="distance").fit(X, y)
        assert reg.predict(np.ones((5, 1))).shape == (5,)

    def test_multiple_exact_matches(self):
        X = np.array([[0.0], [0.0], [5.0]])   # duplicated training point
        y = np.array([10.0, 20.0, 30.0])
        reg = KNNRegressor(k=2, weights="distance").fit(X, y)
        pred = reg.predict([[0.0]])
        assert pred[0] == pytest.approx(15.0)   # mean of 10 and 20


# ===========================================================================
# score (R²)
# ===========================================================================

class TestScore:
    def test_perfect_r2_k1(self, linear_data):
        X, y = linear_data
        reg = KNNRegressor(k=1).fit(X, y)
        # k=1 on training set → each point is its own neighbour → R² = 1
        assert reg.score(X, y) == pytest.approx(1.0)

    def test_r2_range(self, linear_data):
        X, y = linear_data
        reg = KNNRegressor(k=3).fit(X, y)
        s = reg.score(X, y)
        assert -1e9 < s <= 1.0

    def test_constant_target_r2(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([5.0, 5.0, 5.0])
        reg = KNNRegressor(k=1).fit(X, y)
        # Constant target → SS_tot = 0 → R² = 1 by convention
        assert reg.score(X, y) == pytest.approx(1.0)

    def test_score_before_fit(self):
        with pytest.raises(RuntimeError):
            KNNRegressor().score([[1.0]], [1.0])


# ===========================================================================
# repr
# ===========================================================================

class TestRepr:
    def test_repr_contains_key_info(self):
        r = repr(KNNRegressor(k=7, metric="taxicab", weights="distance"))
        assert "KNNRegressor" in r
        assert "k=7" in r
        assert "taxicab" in r
        assert "distance" in r
