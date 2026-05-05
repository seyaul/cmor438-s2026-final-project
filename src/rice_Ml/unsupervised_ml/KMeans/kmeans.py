"""K-Means clustering: partitions n samples into k clusters by iterating assign-then-recompute until convergence."""

import numpy as np


class KMeans:
    """
    K-Means clustering algorithm.

    Partitions data into k clusters by minimising within-cluster inertia.
    Supports random and k-means++ centroid initialisation.

    Attributes set after fit():
        centroids_  — array of shape (k, n_features) — final cluster centres
        labels_     — 1-D int array of length n_samples — cluster index per point
        inertia_    — float — sum of squared distances to nearest centroid
        n_iter_     — int — number of iterations run before convergence
    """

    def __init__(self, k=8, max_iter=300, tol=1e-4, init="kmeans++"):
        """k clusters, up to max_iter EM steps, convergence tolerance tol, init 'random' or 'kmeans++'."""
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if init not in ("random", "kmeans++"):
            raise ValueError("init must be 'random' or 'kmeans++'")
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.init = init

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self, X):
        """Compute k-means clustering on X; returns self."""
        X = np.array(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        if self.k > len(X):
            raise ValueError("k cannot exceed the number of samples")

        self.centroids_ = self._init_centroids(X)

        for i in range(self.max_iter):
            labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, labels)
            if self._has_converged(self.centroids_, new_centroids):
                break
            self.centroids_ = new_centroids

        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, labels)
        self.n_iter_ = i + 1
        return self

    def predict(self, X):
        """Assign each sample in X to the nearest centroid; returns a label array."""
        if not hasattr(self, "centroids_"):
            raise RuntimeError("KMeans is not fitted yet, call fit() first")
        return self._assign_clusters(np.array(X))

    def fit_predict(self, X):
        """Fit and return cluster labels for X in one call."""
        return self.fit(X).predict(X)

    def score(self, X):
        """Return negative inertia for X (higher is better, sklearn convention)."""
        return -1 * self._compute_inertia(np.array(X), self.labels_)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _init_centroids(self, X):
        """Pick k initial centroids from X using self.init strategy."""
        if self.init == "kmeans++":
            centroids = [X[np.random.randint(len(X))]]
            for _ in range(1, self.k):
                diffs = X[:, np.newaxis] - np.array(centroids)
                sq_dists = (diffs ** 2).sum(axis=2)
                min_sq_dists = sq_dists.min(axis=1)
                probs = min_sq_dists / min_sq_dists.sum()
                centroids.append(X[np.random.choice(len(X), p=probs)])
            return np.array(centroids)

        indices = np.random.choice(len(X), size=self.k, replace=False)
        return X[indices]

    def _assign_clusters(self, X):
        """Return the index of the nearest centroid for each sample in X."""
        diffs = X[:, np.newaxis] - self.centroids_   # (n, k, d)
        sq_distances = (diffs ** 2).sum(axis=2)       # (n, k)
        return np.argmin(sq_distances, axis=1)        # (n,)

    def _update_centroids(self, X, labels):
        """Recompute each centroid as the mean of its assigned samples."""
        centroids = []
        for i in range(self.k):
            points_in_cluster = X[labels == i]
            new_centroid = np.mean(points_in_cluster, axis=0)
            centroids.append(new_centroid)
        return np.array(centroids)

    def _has_converged(self, old_centroids, new_centroids):
        """Return True if the max centroid shift is below self.tol."""
        return bool(np.max(np.linalg.norm(old_centroids - new_centroids, axis=1)) < self.tol)

    def _compute_inertia(self, X, labels):
        """Return the sum of squared distances from each sample to its assigned centroid."""
        inertia = 0
        for i in range(self.k):
            points_in_cluster = X[labels == i]
            diffs = points_in_cluster - self.centroids_[i]
            inertia += np.sum(np.linalg.norm(diffs, axis=1) ** 2)
        return inertia
