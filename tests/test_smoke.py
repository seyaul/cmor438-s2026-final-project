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


def test_import_cnn():
    from rice_Ml.supervised_ml.cnn import CNN
    assert CNN


def test_import_knn_package():
    from rice_Ml.supervised_ml.knn import KNNClassifier, KNNRegressor, KNNRecommender
    assert KNNClassifier
    assert KNNRegressor
    assert KNNRecommender

def test_import_perceptron():
    from rice_Ml.supervised_ml.Perceptron.perceptron import Perceptron
    assert Perceptron


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

def test_smoke_cnn():
    from rice_Ml.supervised_ml.cnn import CNN

    X = np.random.default_rng(0).standard_normal((60, 18))
    y = np.array([0] * 30 + [1] * 30)

    model = CNN(epochs=2, random_state=0).fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)

    assert preds.shape == (60,)
    assert proba.shape == (60, 2)
    assert 0.0 <= model.score(X, y) <= 1.0


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


# ---------------------------------------------------------------------------
# DecisionTree
# ---------------------------------------------------------------------------

def test_import_decision_tree():
    from rice_Ml.supervised_ml.DecisionTree.decision_tree import DecisionTree
    assert DecisionTree


def test_smoke_decision_tree_gini():
    from rice_Ml.supervised_ml.DecisionTree.decision_tree import DecisionTree

    X = np.array([[1.0, 1.0], [1.5, 1.2], [0.8, 1.1],
                  [8.0, 8.0], [8.5, 7.8], [7.9, 8.2]])
    y = np.array([0, 0, 0, 1, 1, 1])

    dt = DecisionTree(criterion="gini").fit(X, y)
    preds = dt.predict(X)

    assert preds.shape == (6,)
    assert dt.score(X, y) == pytest.approx(1.0)


def test_smoke_decision_tree_entropy():
    from rice_Ml.supervised_ml.DecisionTree.decision_tree import DecisionTree

    X = np.array([[1.0, 1.0], [1.5, 1.2], [0.8, 1.1],
                  [8.0, 8.0], [8.5, 7.8], [7.9, 8.2]])
    y = np.array([0, 0, 0, 1, 1, 1])

    dt = DecisionTree(criterion="entropy").fit(X, y)
    preds = dt.predict(X)

    assert preds.shape == (6,)
    assert dt.score(X, y) == pytest.approx(1.0)


def test_smoke_decision_tree_max_depth():
    from rice_Ml.supervised_ml.DecisionTree.decision_tree import DecisionTree

    X = np.array([[1.0, 1.0], [1.5, 1.2], [0.8, 1.1],
                  [8.0, 8.0], [8.5, 7.8], [7.9, 8.2]])
    y = np.array([0, 0, 0, 1, 1, 1])

    dt = DecisionTree(max_depth=1).fit(X, y)
    preds = dt.predict(X)

    assert preds.shape == (6,)
    assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------

def test_import_kmeans():
    from rice_Ml.unsupervised_ml.KMeans import KMeans
    assert KMeans


def test_smoke_kmeans_two_clusters():
    from rice_Ml.unsupervised_ml.KMeans import KMeans

    X = np.array([
        [1.0, 1.0], [1.2, 0.9], [0.9, 1.1],
        [8.0, 8.0], [8.1, 7.9], [7.9, 8.1],
    ])

    km = KMeans(k=2).fit(X)
    preds = km.predict(X)

    assert preds.shape == (6,)
    assert set(preds).issubset({0, 1})
    assert km.centroids_.shape == (2, 2)
    assert km.inertia_ >= 0.0


def test_smoke_kmeans_fit_predict():
    from rice_Ml.unsupervised_ml.KMeans import KMeans

    X = np.array([
        [1.0, 1.0], [1.2, 0.9], [0.9, 1.1],
        [8.0, 8.0], [8.1, 7.9], [7.9, 8.1],
    ])

    labels = KMeans(k=2).fit_predict(X)

    assert labels.shape == (6,)
    assert set(labels).issubset({0, 1})


def test_smoke_kmeans_random_init():
    from rice_Ml.unsupervised_ml.KMeans import KMeans

    X = np.array([
        [1.0, 1.0], [1.2, 0.9], [0.9, 1.1],
        [8.0, 8.0], [8.1, 7.9], [7.9, 8.1],
    ])

    km = KMeans(k=2, init="random").fit(X)
    preds = km.predict(X)

    assert preds.shape == (6,)
    assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# RandomForest
# ---------------------------------------------------------------------------

def test_import_random_forest():
    from rice_Ml.supervised_ml.ensembles import RandomForest
    assert RandomForest


def test_smoke_random_forest():
    from rice_Ml.supervised_ml.ensembles import RandomForest

    X = np.array([[1.0, 1.0], [1.5, 1.2], [0.8, 1.1],
                  [8.0, 8.0], [8.5, 7.8], [7.9, 8.2]])
    y = np.array([0, 0, 0, 1, 1, 1])

    rf = RandomForest(n_estimators=10, random_state=0).fit(X, y)
    preds = rf.predict(X)
    proba = rf.predict_proba(X)

    assert preds.shape == (6,)
    assert proba.shape == (6, 2)
    assert 0.0 <= rf.score(X, y) <= 1.0


# ---------------------------------------------------------------------------
# AdaBoost
# ---------------------------------------------------------------------------

def test_import_adaboost():
    from rice_Ml.supervised_ml.ensembles import AdaBoost
    assert AdaBoost


def test_smoke_adaboost():
    from rice_Ml.supervised_ml.ensembles import AdaBoost

    X = np.array([[1.0, 1.0], [1.5, 1.2], [0.8, 1.1],
                  [8.0, 8.0], [8.5, 7.8], [7.9, 8.2]])
    y = np.array([0, 0, 0, 1, 1, 1])

    ab = AdaBoost(n_estimators=10, random_state=0).fit(X, y)
    preds = ab.predict(X)
    proba = ab.predict_proba(X)

    assert preds.shape == (6,)
    assert proba.shape == (6, 2)
    assert 0.0 <= ab.score(X, y) <= 1.0


# ---------------------------------------------------------------------------
# StackingClassifier
# ---------------------------------------------------------------------------

def test_import_stacking():
    from rice_Ml.supervised_ml.ensembles import StackingClassifier
    assert StackingClassifier


def test_smoke_stacking():
    from rice_Ml.supervised_ml.ensembles import StackingClassifier
    from rice_Ml.supervised_ml.DecisionTree.decision_tree import DecisionTree
    from rice_Ml.supervised_ml.knn import KNNClassifier
    from rice_Ml.supervised_ml.linear_model import LogisticRegression

    X = np.array([[1.0, 1.0], [1.5, 1.2], [0.8, 1.1], [1.2, 0.9],
                  [8.0, 8.0], [8.5, 7.8], [7.9, 8.2], [8.2, 8.1]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    stacker = StackingClassifier(
        estimators=[("dt", DecisionTree(max_depth=2)), ("knn", KNNClassifier(k=2))],
        meta_estimator=LogisticRegression(n_epochs=200),
        cv=2,
    ).fit(X, y)

    preds = stacker.predict(X)
    assert preds.shape == (8,)
    assert 0.0 <= stacker.score(X, y) <= 1.0
