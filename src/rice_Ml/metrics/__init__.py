from .base import Metric
from .classification import (
    Accuracy, Precision, Recall, F1Score,
    accuracy, precision, recall, f1_score, per_class_f1
)
from .regression import (
    R2Score, RMSE, MAE,
    r2_score, rmse, mae
)

__all__ = [
    # Base
    'Metric',
    # Classification classes
    'Accuracy', 'Precision', 'Recall', 'F1Score',
    # Classification functions
    'accuracy', 'precision', 'recall', 'f1_score', 'per_class_f1',
    # Regression classes
    'R2Score', 'RMSE', 'MAE',
    # Regression functions
    'r2_score', 'rmse', 'mae',
]