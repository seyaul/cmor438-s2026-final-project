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

Start with random centroids
Assign each point to its nearest centroid
Recalculate centroids as the mean of assigned points
Repeat until centroids stop moving


attributes: clusters to form = k, stop moving threshold = m, epochs = e

choose a # of centroids equal to k on x, y 



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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass
