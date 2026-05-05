import numpy as np
from .base import Optimizer

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with optional momentum and gradient clipping.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        clipnorm: float = None,
    ):
        """Initialise SGD with learning rate, optional momentum, and gradient clipping."""
        self.lr = learning_rate
        self.momentum = momentum
        self.clipnorm = clipnorm
        self.velocities = {}

    def update(self, params: dict, grads: dict) -> None:
        """
        Update parameters in-place.

        Parameters
        ----------
        params : dict
            Dictionary of parameter name -> numpy array.
        grads : dict
            Dictionary of parameter name -> gradient array.
        """
        for name, param in params.items():
            grad = grads[name].copy() 

            # ---- Gradient Clipping ----
            if self.clipnorm is not None and self.clipnorm > 0:
                grad_norm = np.linalg.norm(grad)
                if grad_norm > self.clipnorm:
                    grad = grad * (self.clipnorm / grad_norm)

            # ---- Momentum Update ----
            if self.momentum > 0:
                if name not in self.velocities:
                    self.velocities[name] = np.zeros_like(param)
                self.velocities[name] = self.momentum * self.velocities[name] - self.lr * grad
                param += self.velocities[name]
            else:
                param -= self.lr * grad