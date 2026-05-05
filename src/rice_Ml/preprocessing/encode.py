import numpy as np

class OneHotEncoder:
    """Encode categorical features as one-hot numeric arrays."""
    def __init__(self, sparse=False, handle_unknown='ignore'):
        self.categories_ = None
        self.sparse = sparse
        self.handle_unknown = handle_unknown

    def fit(self, X: np.ndarray) -> 'OneHotEncoder':
        """Compute the unique categories for each feature column in X."""
        self.categories_ = [np.unique(col) for col in X.T]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode X as a one-hot numeric array."""
        # Implementation returning one-hot matrix
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to X, then return its one-hot encoding."""
        return self.fit(X).transform(X)

class LabelEncoder:
    """Encode target labels with values between 0 and n_classes-1."""
    def fit(self, y):
        """Learn the label-to-integer mapping from y."""
        ...
    def transform(self, y):
        """Map labels in y to integers."""
        ...
    def inverse_transform(self, y):
        """Map integer-encoded labels back to their original values."""
        ...
