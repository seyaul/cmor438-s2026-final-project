"""Unit tests for StackingClassifier."""

import numpy as np
import pytest
from rice_ML.supervised_ml.ensembles.stacking import StackingClassifier
from rice_ML.supervised_ml.DecisionTree.decision_tree import DecisionTree
from rice_ML.supervised_ml.knn.classifier import KNNClassifier
from rice_ML.supervised_ml.linear_model import LogisticRegression


@pytest.fixture
def binary_blobs():
    X = np.array([
        [1.0, 1.0], [1.5, 1.2], [0.8, 1.1], [1.2, 0.9], [0.9, 1.3],
        [8.0, 8.0], [8.5, 7.8], [7.9, 8.2], [8.2, 8.1], [7.8, 7.9],
    ])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return X, y


def make_stacker(binary_blobs):
    X, y = binary_blobs
    estimators = [
        ("dt", DecisionTree(max_depth=2, random_state=0)),
        ("knn", KNNClassifier(k=2)),
    ]
    meta = LogisticRegression(n_epochs=200)
    return StackingClassifier(estimators=estimators, meta_estimator=meta, cv=2), X, y


# ===========================================================================
# TestInit
# ===========================================================================

class TestInit:
    def test_defaults(self, binary_blobs):
        stacker, _, _ = make_stacker(binary_blobs)
        assert stacker.cv == 2

    def test_empty_estimators_raises(self):
        with pytest.raises(ValueError, match="estimators"):
            StackingClassifier(estimators=[], meta_estimator=LogisticRegression())

    def test_invalid_cv_raises(self):
        with pytest.raises(ValueError, match="cv"):
            StackingClassifier(
                estimators=[("dt", DecisionTree())],
                meta_estimator=LogisticRegression(),
                cv=1,
            )


# ===========================================================================
# TestFit
# ===========================================================================

class TestFit:
    def test_returns_self(self, binary_blobs):
        stacker, X, y = make_stacker(binary_blobs)
        assert stacker.fit(X, y) is stacker

    def test_sets_estimators_(self, binary_blobs):
        stacker, X, y = make_stacker(binary_blobs)
        stacker.fit(X, y)
        assert len(stacker.estimators_) == 2

    def test_sets_classes(self, binary_blobs):
        stacker, X, y = make_stacker(binary_blobs)
        stacker.fit(X, y)
        np.testing.assert_array_equal(stacker.classes_, [0, 1])

    def test_1d_X_raises(self, binary_blobs):
        stacker, X, y = make_stacker(binary_blobs)
        with pytest.raises(ValueError, match="2-D"):
            stacker.fit(np.ones(10), y)

    def test_length_mismatch_raises(self, binary_blobs):
        stacker, X, y = make_stacker(binary_blobs)
        with pytest.raises(ValueError):
            stacker.fit(X, y[:5])


# ===========================================================================
# TestPredict
# ===========================================================================

class TestPredict:
    def test_output_shape(self, binary_blobs):
        stacker, X, y = make_stacker(binary_blobs)
        stacker.fit(X, y)
        assert stacker.predict(X).shape == (len(y),)

    def test_labels_in_training_set(self, binary_blobs):
        stacker, X, y = make_stacker(binary_blobs)
        stacker.fit(X, y)
        assert set(stacker.predict(X)).issubset({0, 1})

    def test_score_in_unit_interval(self, binary_blobs):
        stacker, X, y = make_stacker(binary_blobs)
        stacker.fit(X, y)
        assert 0.0 <= stacker.score(X, y) <= 1.0


# ===========================================================================
# TestScore
# ===========================================================================

class TestScore:
    def test_score_in_unit_interval(self, binary_blobs):
        stacker, X, y = make_stacker(binary_blobs)
        stacker.fit(X, y)
        assert 0.0 <= stacker.score(X, y) <= 1.0
