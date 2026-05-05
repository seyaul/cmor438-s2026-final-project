import numpy as np

class PCA:
    """
    Principal Component Analysis.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of components to keep. If None, all components are kept.
    whiten : bool, default=False
        When True, the components are scaled to unit variance.
    random_state : int or None, default=None
        Not used (kept for API consistency).

    Attributes
    ----------
    components_ : np.ndarray of shape (n_components, n_features)
        Principal axes in feature space (unit vectors).
    explained_variance_ : np.ndarray of shape (n_components,)
        Variance explained by each component.
    explained_variance_ratio_ : np.ndarray of shape (n_components,)
        Percentage of variance explained by each component.
    mean_ : np.ndarray of shape (n_features,)
        Mean of the training data.
    """
    def __init__(self, n_components=None, whiten=False, random_state=None):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Fit the model with X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute covariance matrix (or use SVD directly)
        # Using eigendecomposition on covariance matrix
        cov = np.cov(X_centered, rowvar=False)  # shape (n_features, n_features)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Limit to n_components
        if self.n_components is not None:
            eigenvalues = eigenvalues[:self.n_components]
            eigenvectors = eigenvectors[:, :self.n_components]

        self.explained_variance_ = eigenvalues
        total_var = np.sum(np.var(X_centered, axis=0, ddof=0))
        self.explained_variance_ratio_ = eigenvalues / total_var if total_var > 0 else eigenvalues

        # Components as rows (scikit-learn convention)
        self.components_ = eigenvectors.T  # shape (n_components, n_features)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_components)
        """
        if self.components_ is None:
            raise RuntimeError("PCA must be fitted before transform().")
        X_centered = X - self.mean_
        X_transformed = X_centered @ self.components_.T

        if self.whiten:
            # scale by eigenvalues^(-1/2)
            X_transformed /= np.sqrt(self.explained_variance_ + 1e-15)

        return X_transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data and transform it."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.

        Parameters
        ----------
        X_transformed : np.ndarray of shape (n_samples, n_components)

        Returns
        -------
        X_reconstructed : np.ndarray of shape (n_samples, n_features)
        """
        if self.components_ is None:
            raise RuntimeError("PCA must be fitted before inverse_transform().")
        X_reconstructed = X_transformed @ self.components_ + self.mean_
        return X_reconstructed

    def score(self, X: np.ndarray) -> float:
        """Return the mean log‑likelihood per sample (placeholder)."""
        # Could compute reconstruction error; for now returns 0
        return 0.0