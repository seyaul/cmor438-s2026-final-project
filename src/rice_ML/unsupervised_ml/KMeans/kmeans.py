"""
kmeans.py

K-Means clustering: partitions n samples into k clusters by iteratively
assigning each point to its nearest centroid and recomputing centroids
until convergence or the iteration limit is reached.
"""

#bad psuedocode
# while epochs < e
#     points_list = []
#     for each point on the x,y 
#         distance = [float(inf)]
#         orbit = None
#         for each centroid 
#             distance_from_centroid = distance calculation
#             if distance_from_centroid < distance:
#                 distance = distance_from_centroid
#                 orbit = centroid
#     points_list.append(point, orbit)
    
#     for nc in orbit:
#         what happens if one centroid isn't in orbit?
#         centroid.index(nc) = np.mean(points_list[:, nc])
        
#     find_centroids(x,y, epochs+1, nc)
    
#     epochs++

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
        """
        k        — number of clusters to form
        max_iter — maximum number of EM iterations
        tol      — convergence tolerance; stop when centroid shift <= tol
        init     — centroid initialisation strategy: 'random' or 'kmeans++'
        """
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.init = init

    # Start with random centroids
    # Assign each point to its nearest centroid
    # Recalculate centroids as the mean of assigned points
    # Repeat until centroids stop moving


    # attributes: clusters to form = k, stop moving threshold = m, epochs = e

    # choose a # of centroids equal to k on x, y 



    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self, X):
        """
        Compute k-means clustering on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        
        X = np.array(X)

        self.centroids_ = self._init_centroids(X)

        for i in range(self.max_iter):
            #assigns all points to centroids
            labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, labels)
            if self._has_converged(self.centroids_, new_centroids):
                break
            self.centroids_ = new_centroids

        #assigns labels, inertia and n_iter before they centroids are stable. 
        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, labels)
        self.n_iter_ = i + 1

        return self


    def predict(self, X):
        """
        Assign each sample in X to the nearest centroid.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if not hasattr(self, 'centroids_'):
            raise RuntimeError("KMeans is not fitted yet, call fit() first")
        return self._assign_clusters(np.array(X))


    def fit_predict(self, X):
        """
        Fit and return cluster labels for X in one call.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        return self.fit(X).predict(X)

    def score(self, X):
        """
        Return negative inertia for X (higher is better, sklearn convention).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        score : float
            Negative sum of squared distances to the nearest centroid.
        """
        
        return -1 * self._compute_inertia(np.array(X), self.labels_)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _init_centroids(self, X):
        """
        Initialise k centroids from X using self.init strategy.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        centroids : ndarray of shape (k, n_features)
        """
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
        """
        Assign each sample to its nearest centroid.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,) — cluster indices
        """
        
        # original O(n·k·d) Python loop — kept for reference
        # returnal = []
        # for point in X:
        #     nerest_centroid = float('inf')
        #     nerest_index = None
        #     for i, c in enumerate(self.centroids_):
        #         distance = np.linalg.norm(point - c)
        #         if distance < nerest_centroid:
        #             nerest_centroid = distance
        #             nerest_index = i
        #     returnal.append(nerest_index)
        # return np.array(returnal)

        diffs = X[:, np.newaxis] - self.centroids_   # (n, k, d)
        sq_distances = (diffs ** 2).sum(axis=2)       # (n, k)
        return np.argmin(sq_distances, axis=1)        # (n,)



    def _update_centroids(self, X, labels):
        """
        Recompute each centroid as the mean of its assigned samples.

        Parameters
        ----------
        X      : ndarray of shape (n_samples, n_features)
        labels : ndarray of shape (n_samples,)

        Returns
        -------
        centroids : ndarray of shape (k, n_features)
        """
        centroids = []
        #k is number of clusters in k-means, labels[i] is the indexical centroid
        for i in range(self.k):
            points_in_cluster = X[labels == i]
            new_centroid = np.mean(points_in_cluster, axis=0)
            centroids.append(new_centroid)
        return np.array(centroids)


    def _has_converged(self, old_centroids, new_centroids):
        """
        Check whether centroids have shifted by less than self.tol.

        Parameters
        ----------
        old_centroids : ndarray of shape (k, n_features)
        new_centroids : ndarray of shape (k, n_features)

        Returns
        -------
        converged : bool
        """
        return np.max(np.linalg.norm(old_centroids-new_centroids, axis=1)) < self.tol


    def _compute_inertia(self, X, labels):
        """
        Compute sum of squared distances from each sample to its centroid.

        Parameters
        ----------
        X      : ndarray of shape (n_samples, n_features)
        labels : ndarray of shape (n_samples,)

        Returns
        -------
        inertia : float
        """
        inertia = 0
        for i in range(self.k):
            points_in_cluster = X[labels == i]
            diffs = points_in_cluster - self.centroids_[i]
            inertia += np.sum(np.linalg.norm(diffs, axis=1)**2)
        return inertia
