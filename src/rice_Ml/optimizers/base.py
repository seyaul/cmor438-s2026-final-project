from abc import ABC, abstractmethod

class Optimizer(ABC):
    """Abstract base class for all optimizers."""

    @abstractmethod
    def update(self, params: dict, grads: dict) -> None:
        """Update params in-place using the provided gradients."""
        pass