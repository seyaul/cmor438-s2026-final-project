import numpy as np

class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:
        z = (x - u) / s
    where `u` is the mean of the training samples and `s` is the standard deviation.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        The mean value for each feature in the training set.
    scale_ : np.ndarray of shape (n_features,)
        The standard deviation for each feature in the training set.
    """

    def __init__(self) -> None:
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Compute the mean and standard deviation used for scaling.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=0)

        # Avoid division by zero for constant features
        self.scale_[self.scale_ == 0] = 1.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The data to be scaled.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Scaled data.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("This StandardScaler instance is not fitted yet. "
                               "Call 'fit' with appropriate arguments before using this estimator.")

        X = np.asarray(X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The data used to fit and transform.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Scaled data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Undo the scaling of X according to the stored mean and scale.

        Parameters
        ----------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Scaled data to be transformed back to original scale.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            Data in original scale.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("This StandardScaler instance is not fitted yet.")
        X_scaled = np.asarray(X_scaled)
        return X_scaled * self.scale_ + self.mean_