import numpy as np

class SimpleImputer:
    """Impute missing values with mean, median, or most frequent."""
    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, X: np.ndarray) -> 'SimpleImputer':
        """Compute per-column imputation statistics from X."""
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == 'most_frequent':
            # Compute mode for each column
            pass
        elif self.strategy == 'constant':
            self.statistics_ = self.fill_value
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Replace NaN values with the fitted per-column statistics."""
        # Replace NaNs with statistics_
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to X, then impute and return it."""
        return self.fit(X).transform(X)
