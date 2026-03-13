"""
Smoke tests — fast, top-level sanity checks.

These do NOT test correctness in depth; they verify that the package imports
cleanly and that each public class can be instantiated and run end-to-end
without crashing.  They should complete in well under a second.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

def test_import_distances():
    from rice_Ml.measures_ml.distances import euclidean, taxicab
    assert callable(euclidean)
    assert callable(taxicab)


def test_import_knn_package():
    from rice_Ml.supervised_ml.knn import KNNClassifier, KNNRegressor, KNNRecommender
    assert KNNClassifier
    assert KNNRegressor
    assert KNNRecommender


# ---------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------

def test_smoke_euclidean():
    from rice_Ml.measures_ml.distances import euclidean
    assert euclidean([0, 0], [3, 4]) == pytest.approx(5.0)


def test_smoke_taxicab():
    from rice_Ml.measures_ml.distances import taxicab
    assert taxicab([0, 0], [3, 4]) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# KNNClassifier
# ---------------------------------------------------------------------------

def test_smoke_classifier():
    from rice_Ml.supervised_ml.knn import KNNClassifier

    X = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 5.0]])
    y = np.array([0, 0, 1, 1])

    clf = KNNClassifier(k=2).fit(X, y)
    preds = clf.predict([[0.1, 0.1], [5.5, 5.5]])
    proba = clf.predict_proba([[0.1, 0.1]])

    assert preds.shape == (2,)
    assert proba.shape == (1, 2)
    assert clf.score(X, y) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# KNNRegressor
# ---------------------------------------------------------------------------

def test_smoke_regressor_uniform():
    from rice_Ml.supervised_ml.knn import KNNRegressor

    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.arange(6, dtype=float)

    reg = KNNRegressor(k=2, weights="uniform").fit(X, y)
    preds = reg.predict([[2.0], [4.0]])
    assert preds.shape == (2,)


def test_smoke_regressor_distance():
    from rice_Ml.supervised_ml.knn import KNNRegressor

    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.arange(6, dtype=float)

    reg = KNNRegressor(k=2, weights="distance").fit(X, y)
    preds = reg.predict([[1.0]])
    assert preds.shape == (1,)


# ---------------------------------------------------------------------------
# KNNRecommender
# ---------------------------------------------------------------------------

def test_smoke_recommender():
    from rice_Ml.supervised_ml.knn import KNNRecommender

    R = np.array([
        [5.0, 4.0, 0.0, 0.0],
        [4.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 4.0],
        [0.0, 0.0, 4.0, 5.0],
    ])
    rec = KNNRecommender(k=1).fit(R)

    neighbours, dists = rec.similar_users(0, n=1)
    items = rec.recommend(0, n=2)
    rating = rec.predict_rating(0, 2)

    assert len(neighbours) == 1
    assert isinstance(rating, float)
