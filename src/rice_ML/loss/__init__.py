from .base import Loss
from .regression import MeanSquaredError
from .classification import BinaryCrossEntropy

__all__ = ['Loss', 'MeanSquaredError', 'BinaryCrossEntropy']