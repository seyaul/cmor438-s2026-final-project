"""Loss functions for classification and regression."""

from .base import Loss
from .regression import MeanSquaredError
from .classification import BinaryCrossEntropy, CategoricalCrossEntropy

__all__ = ['Loss', 'MeanSquaredError', 'BinaryCrossEntropy', 'CategoricalCrossEntropy']