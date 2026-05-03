"""
Unit tests for KMeans clustering.
"""

import numpy as np
import pytest
from rice_Ml.unsupervised_ml.KMeans.kmeans import KMeans


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_blobs():
    """Two clearly separated 2-D clusters of equal size."""
    X = np.array([
        [1.0, 1.0], [1.2, 0.9], [0.9, 1.1],
        [8.0, 8.0], [8.1, 7.9], [7.9, 8.1],
    ])
    return X


@pytest.fixture
def three_blobs():
    """Three separable Gaussian clusters."""
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal([0, 0], 0.3, (30, 2)),
        rng.normal([6, 0], 0.3, (30, 2)),
        rng.normal([3, 6], 0.3, (30, 2)),
    ])
    return X


# ===========================================================================
# TestInit
# ===========================================================================

class TestInit:
    def test_defaults(self):
        km = KMeans()
        assert km.k == 8
        assert km.max_iter == 300
        assert km.tol == pytest.approx(1e-4)
        assert km.init == "kmeans++"

    def test_custom_params(self):
        km = KMeans(k=3, max_iter=100, tol=1e-3, init="random")
        assert km.k == 3
        assert km.max_iter == 100
        assert km.tol == pytest.approx(1e-3)
        assert km.init == "random"

    def test_invalid_k_zero(self):
        with pytest.raises(ValueError):
            KMeans(k=0)

    def test_invalid_k_negative(self):
        with pytest.raises(ValueError):
            KMeans(k=-1)

    def test_invalid_max_iter(self):
        with pytest.raises(ValueError):
            KMeans(max_iter=0)

    def test_invalid_init_strategy(self):
        with pytest.raises(ValueError):
            KMeans(init="bad_strategy")


# ===========================================================================
# TestFit
# ===========================================================================

class TestFit:
    def test_sets_centroids(self, two_blobs):
        km = KMeans(k=2).fit(two_blobs)
        assert km.centroids_.shape == (2, 2)

    def test_sets_labels(self, two_blobs):
        km = KMeans(k=2).fit(two_blobs)
        assert km.labels_.shape == (len(two_blobs),)

    def test_sets_inertia(self, two_blobs):
        km = KMeans(k=2).fit(two_blobs)
        assert isinstance(km.inertia_, float)
        assert km.inertia_ >= 0.0

    def test_sets_n_iter(self, two_blobs):
        km = KMeans(k=2).fit(two_blobs)
        assert isinstance(km.n_iter_, int)
        assert km.n_iter_ >= 1

    def test_returns_self(self, two_blobs):
        km = KMeans(k=2)
        result = km.fit(two_blobs)
        assert result is km

    def test_1d_X_raises(self):
        with pytest.raises(ValueError):
            KMeans(k=2).fit(np.array([1.0, 2.0, 3.0]))

    def test_k_exceeds_samples_raises(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError):
            KMeans(k=5).fit(X)

    def test_label_count_matches_samples(self, three_blobs):
        km = KMeans(k=3).fit(three_blobs)
        assert len(km.labels_) == len(three_blobs)

    def test_labels_in_valid_range(self, three_blobs):
        km = KMeans(k=3).fit(three_blobs)
        assert set(km.labels_).issubset(set(range(3)))


# ===========================================================================
# TestPredict
# ===========================================================================

class TestPredict:
    def test_output_shape(self, two_blobs):
        km = KMeans(k=2).fit(two_blobs)
        preds = km.predict(two_blobs)
        assert preds.shape == (len(two_blobs),)

    def test_labels_in_valid_range(self, two_blobs):
        km = KMeans(k=2).fit(two_blobs)
        preds = km.predict(two_blobs)
        assert set(preds).issubset({0, 1})

    def test_consistent_with_fit_labels(self, two_blobs):
        km = KMeans(k=2).fit(two_blobs)
        preds = km.predict(two_blobs)
        np.testing.assert_array_equal(preds, km.labels_)

    def test_separates_two_clusters(self, two_blobs):
        km = KMeans(k=2).fit(two_blobs)
        preds = km.predict(two_blobs)
        # First three points should share one label, last three another
        assert preds[0] == preds[1] == preds[2]
        assert preds[3] == preds[4] == preds[5]
        assert preds[0] != preds[3]

    def test_new_point_nearest_centroid(self, two_blobs):
        km = KMeans(k=2).fit(two_blobs)
        # A point near cluster 1 (around [1,1]) should get the same label
        near_cluster_0 = np.array([[1.05, 1.05]])
        pred = km.predict(near_cluster_0)
        assert pred[0] == km.predict(np.array([[1.0, 1.0]]))[0]


# ===========================================================================
# TestFitPredict
# ===========================================================================

class TestFitPredict:
    def test_output_shape(self, two_blobs):
        labels = KMeans(k=2).fit_predict(two_blobs)
        assert labels.shape == (len(two_blobs),)

    def test_matches_fit_then_predict(self, two_blobs):
        km1 = KMeans(k=2)
        labels_fp = km1.fit_predict(two_blobs)

        km2 = KMeans(k=2)
        km2.fit(two_blobs)
        labels_p = km2.predict(two_blobs)

        # Cluster indices may be permuted — compare partition structure
        assert (labels_fp[0] == labels_fp[1]) == (labels_p[0] == labels_p[1])


# ===========================================================================
# TestScore
# ===========================================================================

class TestScore:
    def test_score_is_negative_or_zero(self, two_blobs):
        km = KMeans(k=2).fit(two_blobs)
        assert km.score(two_blobs) <= 0.0

    def test_more_clusters_higher_score(self, three_blobs):
        score_2 = KMeans(k=2).fit(three_blobs).score(three_blobs)
        score_3 = KMeans(k=3).fit(three_blobs).score(three_blobs)
        assert score_3 >= score_2


# ===========================================================================
# TestInitCentroids
# ===========================================================================

class TestInitCentroids:
    def test_random_shape(self, two_blobs):
        km = KMeans(k=2, init="random")
        centroids = km._init_centroids(two_blobs)
        assert centroids.shape == (2, two_blobs.shape[1])

    def test_kmeans_plus_plus_shape(self, two_blobs):
        km = KMeans(k=2, init="kmeans++")
        centroids = km._init_centroids(two_blobs)
        assert centroids.shape == (2, two_blobs.shape[1])

    def test_random_centroids_from_data(self, two_blobs):
        km = KMeans(k=2, init="random")
        centroids = km._init_centroids(two_blobs)
        for c in centroids:
            assert any(np.allclose(c, row) for row in two_blobs)

    def test_kmeans_plus_plus_centroids_from_data(self, two_blobs):
        km = KMeans(k=2, init="kmeans++")
        centroids = km._init_centroids(two_blobs)
        for c in centroids:
            assert any(np.allclose(c, row) for row in two_blobs)


# ===========================================================================
# TestAssignClusters
# ===========================================================================

class TestAssignClusters:
    def test_output_shape(self, two_blobs):
        km = KMeans(k=2).fit(two_blobs)
        labels = km._assign_clusters(two_blobs)
        assert labels.shape == (len(two_blobs),)

    def test_labels_in_range(self, two_blobs):
        km = KMeans(k=2).fit(two_blobs)
        labels = km._assign_clusters(two_blobs)
        assert set(labels).issubset({0, 1})

    def test_assigns_to_nearest(self):
        km = KMeans(k=2)
        km.centroids_ = np.array([[0.0, 0.0], [10.0, 10.0]])
        X = np.array([[0.1, 0.1], [9.9, 9.9]])
        labels = km._assign_clusters(X)
        assert labels[0] == 0
        assert labels[1] == 1


# ===========================================================================
# TestUpdateCentroids
# ===========================================================================

class TestUpdateCentroids:
    def test_output_shape(self, two_blobs):
        km = KMeans(k=2).fit(two_blobs)
        new_c = km._update_centroids(two_blobs, km.labels_)
        assert new_c.shape == (2, two_blobs.shape[1])

    def test_centroid_is_cluster_mean(self):
        km = KMeans(k=2)
        X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0]])
        labels = np.array([0, 0, 1, 1])
        new_c = km._update_centroids(X, labels)
        np.testing.assert_allclose(new_c[0], [1.0, 0.0])
        np.testing.assert_allclose(new_c[1], [11.0, 0.0])


# ===========================================================================
# TestHasConverged
# ===========================================================================

class TestHasConverged:
    def test_identical_centroids_converged(self):
        km = KMeans(tol=1e-4)
        c = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert km._has_converged(c, c.copy()) is True

    def test_large_shift_not_converged(self):
        km = KMeans(tol=1e-4)
        old = np.array([[0.0, 0.0]])
        new = np.array([[10.0, 10.0]])
        assert km._has_converged(old, new) is False

    def test_small_shift_converged(self):
        km = KMeans(tol=1e-4)
        old = np.array([[0.0, 0.0]])
        new = np.array([[0.00001, 0.00001]])
        assert km._has_converged(old, new) is True


# ===========================================================================
# TestComputeInertia
# ===========================================================================

class TestComputeInertia:
    def test_zero_inertia_when_samples_equal_centroids(self):
        km = KMeans(k=2)
        X = np.array([[1.0, 0.0], [5.0, 0.0]])
        km.centroids_ = X.copy()
        labels = np.array([0, 1])
        assert km._compute_inertia(X, labels) == pytest.approx(0.0)

    def test_inertia_nonnegative(self, two_blobs):
        km = KMeans(k=2).fit(two_blobs)
        assert km._compute_inertia(two_blobs, km.labels_) >= 0.0

    def test_inertia_matches_manual(self):
        km = KMeans(k=2)
        X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0]])
        km.centroids_ = np.array([[1.0, 0.0], [11.0, 0.0]])
        labels = np.array([0, 0, 1, 1])
        expected = 1.0 + 1.0 + 1.0 + 1.0  # each point 1 unit from centroid
        assert km._compute_inertia(X, labels) == pytest.approx(expected)
