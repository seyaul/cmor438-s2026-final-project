import numpy as np
from typing import Optional, Literal

from ...base.base_linear import BaseLinearModel
from ...loss.regression import MeanSquaredError
from ...optimizers.sgd import SGD

class LinearRegression(BaseLinearModel):
    """
    Ordinary Least Squares Linear Regression.

    Supports two solvers:
        - 'normal' : Closed-form solution using the normal equation.
        - 'sgd'    : Stochastic Gradient Descent (with optional mini‑batches).

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    solver : {'normal', 'sgd'}, default='normal'
        Algorithm used to find the coefficients.
    learning_rate : float, default=0.01
        Learning rate for SGD. Ignored if solver='normal'.
    n_epochs : int, default=100
        Number of epochs for SGD. Ignored if solver='normal'.
    batch_size : int or None, default=None
        Mini‑batch size for SGD. If None, uses full‑batch gradient descent.
    random_state : int or None, default=None
        Seed for reproducibility in SGD shuffling.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem.
    intercept_ : float
        Independent term in the linear model.
    loss_history_ : list
        Loss value after each epoch (only populated when solver='sgd').
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        solver: Literal['normal', 'sgd'] = 'normal',
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        batch_size: Optional[int] = None,
        random_state: Optional[int] = None,
        momentum: float = 0.0,          
        clipnorm: Optional[float] = None 
    ):
        super().__init__(fit_intercept)
        self.solver = solver
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.momentum = momentum        
        self.clipnorm = clipnorm
        self.loss_history_ = []

        if self.solver not in ('normal', 'sgd'):
            raise ValueError(f"solver must be 'normal' or 'sgd', got '{self.solver}'")
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the linear regression model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Ensure y is 1D
        y = np.asarray(y).flatten()

        # Add intercept column if requested
        X_design = self._add_intercept(X)
        n_samples, n_features = X_design.shape

        if self.solver == 'normal':
            self._fit_normal(X_design, y)
        else:  # 'sgd'
            self._fit_sgd(X_design, y)

        # Separate intercept from the full weight vector
        self._separate_intercept(self.coef_)

        return self

    def _fit_normal(self, X: np.ndarray, y: np.ndarray) -> None:
        """Closed-form solution: theta = (XᵀX)⁻¹ Xᵀy."""
        try:
            # Use more stable solve instead of explicit inverse
            theta = np.linalg.solve(X.T @ X, X.T @ y)
        except np.linalg.LinAlgError:
            # If singular, fall back to pseudo‑inverse
            theta = np.linalg.pinv(X.T @ X) @ X.T @ y

        self.coef_ = theta

    def _fit_sgd(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using Stochastic Gradient Descent."""
        n_samples, n_features = X.shape

        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize weights (can be zeros or small random)
        self._initialize_weights(n_features, initializer='zeros')

        # Create optimizer and loss
        loss_fn = MeanSquaredError()
        optimizer = SGD(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            clipnorm=self.clipnorm  
        )

        # Determine batch size
        batch_size = self.batch_size if self.batch_size is not None else n_samples

        self.loss_history_ = []

        for epoch in range(self.n_epochs):
            # Shuffle data each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass: predictions
                y_pred = X_batch @ self.coef_

                # Compute loss (for monitoring)
                batch_loss = loss_fn(y_batch, y_pred)
                epoch_loss += batch_loss
                n_batches += 1

                # Compute gradient of loss w.r.t predictions
                dloss = loss_fn.gradient(y_batch, y_pred)

                # Gradient w.r.t weights
                grad = X_batch.T @ dloss

                # Update weights using optimizer
                # We wrap parameters in a dict for the optimizer interface
                params = {'coef': self.coef_}
                grads = {'coef': grad}
                optimizer.update(params, grads)

                # self.coef_ has been updated in‑place

            # Track average epoch loss
            avg_epoch_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            self.loss_history_.append(avg_epoch_loss)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        X_design = self._add_intercept(X)
        # Combine intercept and coefficients
        full_coef = np.concatenate([[self.intercept_], self.coef_]) if self.fit_intercept else self.coef_
        return X_design @ full_coef

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the coefficient of determination R² of the prediction.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.
        y : np.ndarray of shape (n_samples,)
            True values for X.

        Returns
        -------
        score : float
            R² score.
        """
        from ...metrics.regression import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)