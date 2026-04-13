import numpy as np

class SGD:
    """
    Stochastic Gradient Descent optimizer.
    Supports optional momentum.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = {}   # For momentum updates

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
            grad = grads[name]

            if self.momentum > 0:
                if name not in self.velocities:
                    self.velocities[name] = np.zeros_like(param)
                self.velocities[name] = self.momentum * self.velocities[name] - self.lr * grad
                param += self.velocities[name]
            else:
                param -= self.lr * grad