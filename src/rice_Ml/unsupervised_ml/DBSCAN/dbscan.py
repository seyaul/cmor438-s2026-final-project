"""
dbscan.py

DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
groups tightly packed points into clusters and labels outliers as noise.

Unlike K-Means, DBSCAN does not require k upfront and can find arbitrarily
shaped clusters.
"""

import numpy as np


class DBSCAN:
    """
    DBSCAN clustering — density-based, handles noise and non-convex shapes.

    A point is a *core point* if it has at least min_samples neighbours
    (including itself) within eps distance.  Core points seed clusters;
    *border points* are within eps of a core but are not cores themselves;
    everything else is labelled noise (-1).

    Attributes set after fit():
        labels_              — int array of shape (n_samples,); -1 = noise
        core_sample_indices_ — sorted indices of core points
        n_clusters_          — number of clusters found (noise excluded)
    """

    def __init__(self, eps=0.5, min_samples=5, metric="euclidean"):
        """
        eps         — neighbourhood radius; two points are neighbours if
                      their distance is <= eps
        min_samples — points needed in the eps-neighbourhood (including the
                      point itself) for a point to be a core point
        metric      — 'euclidean', 'manhattan', or a callable f(a, b) -> float
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self, X):
        """
        Run DBSCAN on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
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

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _get_neighbors(self, X, idx):
        """
        Return indices of all points within eps of X[idx] (including idx).

        Parameters
        ----------
        X   : ndarray of shape (n_samples, n_features)
        idx : int

        Returns
        -------
        neighbors : ndarray of int indices
        """
        pass

    def _expand_cluster(self, X, labels, idx, neighbors, cluster_id, core_set):
        """
        Grow cluster cluster_id outward from core point idx via BFS.

        Modifies labels and core_set in-place.

        Parameters
        ----------
        X          : ndarray of shape (n_samples, n_features)
        labels     : ndarray of shape (n_samples,) — modified in-place
        idx        : int — seed core point
        neighbors  : ndarray — initial neighborhood of idx
        cluster_id : int
        core_set   : set — accumulates discovered core indices
        """
        pass
