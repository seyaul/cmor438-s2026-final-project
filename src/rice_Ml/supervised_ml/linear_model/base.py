import numpy as np
from ...base.base_model import BaseModel

class BaseLinearModel(BaseModel):
    """
    Base class for all linear models (regression and classification).
    Manages weights, intercept, and design matrix creation.
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None   # Will hold weights (shape: n_features,)
        self.intercept_ = 0.0

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add a column of ones to X for the intercept term."""
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def _initialize_weights(self, n_features: int, initializer: str = 'zeros') -> None:
        """Initialize weight vector."""
        if initializer == 'zeros':
            self.coef_ = np.zeros(n_features)
        elif initializer == 'random_normal':
            self.coef_ = np.random.randn(n_features) * 0.01
        else:
            raise ValueError(f"Unknown initializer '{initializer}'")

    def _separate_intercept(self, full_weights: np.ndarray) -> None:
        """
        Extract intercept and coefficients from a combined weight vector.
        Assumes intercept is the first element if fit_intercept is True.
        """
        if self.fit_intercept:
            self.intercept_ = full_weights[0]
            self.coef_ = full_weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = full_weights