import numpy as np
from .base import Metric


class Accuracy(Metric):
    """Fraction of correct predictions."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return the proportion of predictions matching the true labels."""
        return np.mean(y_true == y_pred)


class Precision(Metric):
    """Precision = TP / (TP + FP)."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return precision for binary predictions."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0


class Recall(Metric):
    """Recall = TP / (TP + FN)."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return recall for binary predictions."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0


class F1Score(Metric):
    """F1 score = 2 * precision * recall / (precision + recall)."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return the harmonic mean of precision and recall."""
        p = Precision()(y_true, y_pred)
        r = Recall()(y_true, y_pred)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# Convenience function aliases (backward compatible)
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions."""
    return Accuracy()(y_true, y_pred)

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision score for binary predictions."""
    return Precision()(y_true, y_pred)

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall score for binary predictions."""
    return Recall()(y_true, y_pred)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 score for binary predictions."""
    return F1Score()(y_true, y_pred)
