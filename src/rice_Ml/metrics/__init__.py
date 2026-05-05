"""Evaluation metrics for classification and regression."""

from .base import Metric
from .classification import (
    Accuracy, Precision, Recall, F1Score,
    accuracy, precision, recall, f1_score
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
    'accuracy', 'precision', 'recall', 'f1_score',
    # Regression classes
    'R2Score', 'RMSE', 'MAE',
    # Regression functions
    'r2_score', 'rmse', 'mae',
]