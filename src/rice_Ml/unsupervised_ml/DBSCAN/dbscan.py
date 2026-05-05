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
        """Run DBSCAN on X; returns self."""
        labels_ = np.full(len(X), -1)
        cluster_id = 0

        core_set = set()  # sets have faster lookup

        for point in range(len(X)):
            if labels_[point]== -1 and len(self._get_neighbors(X, point)) >= self.min_samples:
                core_set.add(point)
                self._expand_cluster(X, labels_, point, self._get_neighbors(X,point), cluster_id, core_set)
                cluster_id += 1
        
        self.labels_ = labels_
        self.core_sample_indices_ = sorted(core_set)
        self.n_clusters_ = cluster_id
        return self

    def fit_predict(self, X):
        """Fit and return cluster labels for X in one call."""
        return self.fit(X).labels_
        

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _get_neighbors(self, X, idx):
        """Return indices of all points within eps of X[idx] (including idx)."""
        distances = np.linalg.norm(X - X[idx], axis=1)
        return np.where(distances <= self.eps)[0]


    def _expand_cluster(self, X, labels, idx, neighbors, cluster_id, core_set):
        """Grow cluster_id outward from core point idx via BFS; modifies labels and core_set in-place."""
        queue = list(neighbors)
        labels[idx] = cluster_id

        while queue:
            curr = queue.pop(0)
            labels[curr] = cluster_id

            for q in self._get_neighbors(X, curr):
                if labels[q] == -1:  # -1 means unvisited/noise
                    labels[q] = cluster_id
                    if len(self._get_neighbors(X, q)) >= self.min_samples:
                        core_set.add(q)
                        queue.extend(self._get_neighbors(X, q))


