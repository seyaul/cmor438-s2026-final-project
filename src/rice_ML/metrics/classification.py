import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions."""
    return np.mean(y_true == y_pred)

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision = TP / (TP + FP)."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall = TP / (TP + FN)."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0