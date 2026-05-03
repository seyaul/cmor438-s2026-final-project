import numpy as np
from typing import Optional, Literal

from .base import BaseLinearModel
from ...loss.classification import BinaryCrossEntropy
from ...optimizers.sgd import SGD
from ...activations.functions import Sigmoid

class LogisticRegression(BaseLinearModel):
    """
    Binary Logistic Regression classifier.

    Solver: 'sgd' only (for simplicity; can add 'irls' later).

    Parameters
    ----------
    fit_intercept : bool, default=True
    learning_rate : float, default=0.01
    n_epochs : int, default=100
    batch_size : int or None, default=None
    momentum : float, default=0.0
    clipnorm : float or None, default=None
    random_state : int or None, default=None

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
    intercept_ : float
    loss_history_ : list of float
    """
    def __init__(
        self,
        fit_intercept: bool = True,
        solver: Literal['sgd'] = 'sgd',
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        batch_size: Optional[int] = None,
        momentum: float = 0.0,
        clipnorm: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(fit_intercept)
        self.solver = solver
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.clipnorm = clipnorm
        self.random_state = random_state
        self.loss_history_ = []

        # Validation
        if self.solver not in ('sgd',):
            raise ValueError("Only 'sgd' solver is currently supported.")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be > 0, got {self.n_epochs}")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.momentum < 0 or self.momentum >= 1:
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}")
        if self.clipnorm is not None and self.clipnorm <= 0:
            raise ValueError(f"clipnorm must be > 0, got {self.clipnorm}")


    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """Fit the model using SGD."""
        y = np.asarray(y).flatten()
        X_design = self._add_intercept(X)
        n_samples, n_features = X_design.shape

        self._fit_sgd(X_design, y)
        self._separate_intercept(self.coef_)
        return self

    def _fit_sgd(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self._initialize_weights(n_features, initializer='zeros')
        loss_fn = BinaryCrossEntropy()
        sigmoid = Sigmoid()
        optimizer = SGD(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            clipnorm=self.clipnorm
        )

        batch_size = self.batch_size if self.batch_size is not None else n_samples
        self.loss_history_ = []

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass: linear combination -> sigmoid
                z = X_batch @ self.coef_
                y_pred = sigmoid(z)

                # Loss
                batch_loss = loss_fn(y_batch, y_pred)
                epoch_loss += batch_loss
                n_batches += 1

                # Gradient of loss w.r.t. predictions
                dloss = loss_fn.gradient(y_batch, y_pred)
                # Chain rule: dloss/dz = dloss/dy_pred * dy_pred/dz
                dz = dloss * sigmoid.gradient(z)
                # Gradient w.r.t. weights
                grad = X_batch.T @ dz / batch_size

                params = {'coef': self.coef_}
                grads = {'coef': grad}
                optimizer.update(params, grads)

            avg_epoch_loss = epoch_loss / n_batches
            self.loss_history_.append(avg_epoch_loss)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for the positive class."""
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before predict_proba().")
        X_design = self._add_intercept(X)
        full_coef = np.concatenate([[self.intercept_], self.coef_]) if self.fit_intercept else self.coef_
        z = X_design @ full_coef
        return Sigmoid()(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels (0 or 1)."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean accuracy on the given test data and labels."""
        from ...metrics.classification import accuracy
        y_pred = self.predict(X)
        return accuracy(y, y_pred)