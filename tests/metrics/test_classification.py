import numpy as np
import pytest
from rice_Ml.metrics.classification import accuracy, precision, recall, f1_score
import warnings

class TestAccuracy:
    def test_perfect(self):
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0])
        assert accuracy(y_true, y_pred) == 1.0

    def test_all_wrong(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([0, 1, 0])
        assert accuracy(y_true, y_pred) == 0.0

    def test_mixed(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1])
        # Correct: index 0, 3, 4 → 3/5 = 0.6
        assert accuracy(y_true, y_pred) == 0.6

    def test_empty(self):
        y_true = np.array([])
        y_pred = np.array([])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = accuracy(y_true, y_pred)
        assert np.isnan(result)


class TestPrecision:
    def test_perfect(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert precision(y_true, y_pred) == 1.0

    def test_no_predicted_positives(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([0, 0, 0])
        # TP = 0, FP = 0 → should return 0.0 by your code
        assert precision(y_true, y_pred) == 0.0

    def test_some_false_positives(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 0])
        # TP = 2 (first and third), FP = 1 (second) → 2/3 ≈ 0.6667
        assert precision(y_true, y_pred) == pytest.approx(2/3)

    def test_all_false_positives(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        # TP = 0, FP = 3 → 0/3 = 0.0
        assert precision(y_true, y_pred) == 0.0


class TestRecall:
    def test_perfect(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert recall(y_true, y_pred) == 1.0

    def test_no_true_positives_in_ground_truth(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 0, 1])
        # TP = 0, FN = 0 → recall = 0.0 (by code)
        assert recall(y_true, y_pred) == 0.0

    def test_missed_positives(self):
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        # TP = 2, FN = 1 → 2/3 ≈ 0.6667
        assert recall(y_true, y_pred) == pytest.approx(2/3)

    def test_all_missed(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        assert recall(y_true, y_pred) == 0.0


class TestF1Score:
    def test_perfect(self):
        y_true = np.array([1, 0, 1, 1])
        y_pred = np.array([1, 0, 1, 1])
        assert f1_score(y_true, y_pred) == 1.0

    def test_worst(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([0, 1, 0])
        # precision=0, recall=0 → F1=0
        assert f1_score(y_true, y_pred) == 0.0

    def test_mixed(self):
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])
        # TP = 2 (first and last), FP = 1 (fourth), FN = 1 (second)
        # precision = 2/3, recall = 2/3 → F1 = 2/3 ≈ 0.6667
        assert f1_score(y_true, y_pred) == pytest.approx(2/3)

    def test_zero_division_handling(self):
        # When precision+recall=0, function returns 0.0
        y_true = np.array([1, 1])
        y_pred = np.array([0, 0])
        assert f1_score(y_true, y_pred) == 0.0


class TestConsistency:
    """Check that metrics behave consistently with each other."""
    
    def test_f1_harmonic_mean(self):
        y_true = np.array([1, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        expected_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        assert f1_score(y_true, y_pred) == pytest.approx(expected_f1)