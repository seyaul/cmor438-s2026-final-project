import numpy as np
from .base import Metric


class Accuracy(Metric):
    """Fraction of correct predictions."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)


class Precision(Metric):
    """Precision = TP / (TP + FP)."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0


class Recall(Metric):
    """Recall = TP / (TP + FN)."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0


class F1Score(Metric):
    """F1 score = 2 * precision * recall / (precision + recall)."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = Precision()(y_true, y_pred)
        r = Recall()(y_true, y_pred)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# Convenience function aliases (backward compatible)
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return Accuracy()(y_true, y_pred)

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return Precision()(y_true, y_pred)

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return Recall()(y_true, y_pred)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return F1Score()(y_true, y_pred)

def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Per-class F1 scores for multi-class classification. Returns array aligned with classes."""
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return np.array(f1s)