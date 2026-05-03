"""Unit tests for KNNRecommender."""

import numpy as np
import pytest

from rice_Ml.supervised_ml.knn.recommender import KNNRecommender

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rating_matrix():
    """
    4 users, 5 items.
    Users 0 & 1 like items 0-1 (group A).
    Users 2 & 3 like items 3-4 (group B).
    """
    return np.array([
        [5.0, 4.0, 0.0, 0.0, 0.0],   # user 0 — group A
        [4.0, 5.0, 0.0, 0.0, 0.0],   # user 1 — group A
        [0.0, 0.0, 0.0, 5.0, 4.0],   # user 2 — group B
        [0.0, 0.0, 0.0, 4.0, 5.0],   # user 3 — group B
    ])


@pytest.fixture
def rec(rating_matrix):
    return KNNRecommender(k=1).fit(rating_matrix)


# ===========================================================================
# Initialisation
# ===========================================================================

class TestInit:
    def test_defaults(self):
        r = KNNRecommender()
        assert r.k == 5
        assert r.metric == "euclidean"

    def test_custom_params(self):
        r = KNNRecommender(k=3, metric="taxicab")
        assert r.k == 3

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            KNNRecommender(k=0)

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            KNNRecommender(metric="cosine")

    def test_callable_metric(self):
        fn = lambda u, v: float(np.sum(np.abs(np.asarray(u) - np.asarray(v))))
        r = KNNRecommender(k=1, metric=fn)
        assert r._distance_fn is fn


# ===========================================================================
# fit
# ===========================================================================

class TestFit:
    def test_returns_self(self, rating_matrix):
        r = KNNRecommender(k=1)
        assert r.fit(rating_matrix) is r

    def test_attributes_set(self, rec):
        assert rec.n_users_ == 4
        assert rec.n_items_ == 5
        assert rec._is_fitted

    def test_nan_converted_to_zero(self):
        R = np.array([[5.0, np.nan], [np.nan, 5.0], [3.0, 3.0]])
        rec = KNNRecommender(k=1).fit(R)
        assert np.all(rec._R >= 0)

    def test_k_ge_n_users_raises(self, rating_matrix):
        with pytest.raises(ValueError, match="less than"):
            KNNRecommender(k=4).fit(rating_matrix)   # k must be < n_users

    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            KNNRecommender(k=1).fit([1.0, 2.0, 3.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            KNNRecommender(k=1).fit(np.zeros((0, 5)))

    def test_none_raises(self):
        with pytest.raises(TypeError):
            KNNRecommender(k=1).fit(None)

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError):
            KNNRecommender(k=1).fit([["a", "b"], ["c", "d"], ["e", "f"]])


# ===========================================================================
# similar_users
# ===========================================================================

class TestSimilarUsers:
    def test_group_a_user_finds_group_a_neighbour(self, rec):
        indices, dists = rec.similar_users(0, n=1)
        assert indices[0] == 1   # user 1 is in the same group

    def test_group_b_user_finds_group_b_neighbour(self, rec):
        indices, dists = rec.similar_users(2, n=1)
        assert indices[0] == 3

    def test_distances_non_negative(self, rec):
        _, dists = rec.similar_users(0, n=3)
        assert np.all(dists >= 0)

    def test_distances_ascending(self, rec):
        _, dists = rec.similar_users(0, n=3)
        assert np.all(np.diff(dists) >= 0)

    def test_self_not_returned(self, rec):
        indices, _ = rec.similar_users(0, n=3)
        assert 0 not in indices

    def test_output_lengths(self, rec):
        indices, dists = rec.similar_users(0, n=2)
        assert len(indices) == 2
        assert len(dists) == 2

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            KNNRecommender(k=1).similar_users(0)

    def test_out_of_range_user(self, rec):
        with pytest.raises(ValueError, match="out of range"):
            rec.similar_users(99)

    def test_non_integer_user_idx(self, rec):
        with pytest.raises(TypeError):
            rec.similar_users(0.5)


# ===========================================================================
# recommend
# ===========================================================================

class TestRecommend:
    def test_group_a_user_gets_group_a_items(self, rec):
        # user 0 has rated items 0-1; unseen items are 2, 3, 4
        # nearest neighbour (user 1) has rated items 0-1 → scores for 3-4 = 0
        items = rec.recommend(0, n=5)
        # returned items should not include already-seen items 0 or 1
        assert 0 not in items
        assert 1 not in items

    def test_exclude_seen_false_includes_rated_items(self, rating_matrix):
        rec = KNNRecommender(k=1).fit(rating_matrix)
        items = rec.recommend(0, n=5, exclude_seen=False)
        # With exclude_seen=False rated items may appear in results
        assert len(items) > 0

    def test_output_length_at_most_n(self, rec):
        items = rec.recommend(0, n=2)
        assert len(items) <= 2

    def test_returns_ndarray(self, rec):
        assert isinstance(rec.recommend(0, n=3), np.ndarray)

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            KNNRecommender(k=1).recommend(0)

    def test_out_of_range_user_raises(self, rec):
        with pytest.raises(ValueError):
            rec.recommend(100)

    def test_invalid_n_raises(self, rec):
        with pytest.raises(ValueError):
            rec.recommend(0, n=0)

    def test_cross_group_recommendation(self):
        """User from group A should not get items only seen by group B."""
        R = np.array([
            [5.0, 5.0, 0.0, 0.0],   # user 0 — group A
            [4.0, 4.0, 0.0, 0.0],   # user 1 — group A (nearest to user 0)
            [0.0, 0.0, 5.0, 5.0],   # user 2 — group B
        ])
        rec = KNNRecommender(k=1).fit(R)
        items = rec.recommend(0, n=4)
        # Nearest user is user 1 who only rated items 0 & 1 → nothing useful unseen
        # or very low scores for items 2-3 (rated 0 by neighbour)
        # either way items 0 & 1 should be excluded (seen)
        assert 0 not in items
        assert 1 not in items

    def test_no_unseen_items_returns_empty(self):
        """User who rated everything gets an empty recommendation list."""
        R = np.array([
            [5.0, 4.0, 3.0],   # user 0 — rated all items
            [4.0, 3.0, 2.0],   # user 1
            [3.0, 2.0, 1.0],   # user 2
        ])
        rec = KNNRecommender(k=1).fit(R)
        items = rec.recommend(0, exclude_seen=True)
        assert len(items) == 0


# ===========================================================================
# predict_rating
# ===========================================================================

class TestPredictRating:
    def test_predict_rating_returns_float(self, rec):
        rating = rec.predict_rating(0, 2)
        assert isinstance(rating, float)

    def test_predict_rating_non_negative(self, rec):
        assert rec.predict_rating(0, 3) >= 0.0

    def test_out_of_range_item(self, rec):
        with pytest.raises(ValueError, match="out of range"):
            rec.predict_rating(0, 99)

    def test_out_of_range_user(self, rec):
        with pytest.raises(ValueError):
            rec.predict_rating(99, 0)

    def test_known_rating(self):
        """Predicted rating equals the nearest neighbour's rating for that item."""
        R = np.array([
            [5.0, 0.0, 0.0],   # user 0 — only rated item 0
            [5.0, 4.0, 0.0],   # user 1 — close: dist([5,0,0],[5,4,0]) = 4
            [0.0, 0.0, 5.0],   # user 2 — far:  dist([5,0,0],[0,0,5]) ≈ 7.07
        ])
        rec = KNNRecommender(k=1).fit(R)
        # Nearest neighbour is user 1; they rated item 1 as 4.0
        rating = rec.predict_rating(0, 1)
        assert rating == pytest.approx(4.0)

    def test_non_integer_item_idx(self, rec):
        with pytest.raises(TypeError):
            rec.predict_rating(0, 1.5)


# ===========================================================================
# repr
# ===========================================================================

class TestRepr:
    def test_repr(self):
        r = repr(KNNRecommender(k=3, metric="taxicab"))
        assert "KNNRecommender" in r
        assert "k=3" in r
