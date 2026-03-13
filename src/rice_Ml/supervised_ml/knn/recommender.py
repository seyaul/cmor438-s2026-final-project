"""
K-Nearest Neighbours user-based collaborative filtering recommender.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from ._base import _resolve_metric


class KNNRecommender:
    """User-based collaborative filtering recommender using KNN.

    Finds the *k* most similar users to a target user (by distance over their
    rating vectors), then scores unrated items as an inverse-distance-weighted
    sum of those neighbours' ratings.

    Missing ratings in the input matrix should be encoded as ``0`` or
    ``numpy.nan``; ``nan`` values are converted to ``0`` before any distance
    computation, so all built-in metrics remain applicable.

    Parameters
    ----------
    k : int, default 5
        Number of similar users to consult when generating recommendations.
    metric : str or callable, default ``"euclidean"``
        Distance function.  Accepts ``"euclidean"`` or ``"taxicab"`` for the
        built-ins in :mod:`rice_Ml.measures_ml.distances`, or any callable
        with signature ``f(u, v) -> float``.

    Attributes
    ----------
    n_users_ : int
        Number of users seen during :meth:`fit`.
    n_items_ : int
        Number of items seen during :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> from rice_Ml.supervised_ml.knn.recommender import KNNRecommender
    >>> R = np.array([
    ...     [5, 4, 0, 0],   # user 0
    ...     [5, 5, 0, 0],   # user 1 — very similar to user 0
    ...     [0, 0, 4, 5],   # user 2 — very different
    ...     [0, 0, 5, 4],   # user 3 — similar to user 2
    ... ], dtype=float)
    >>> rec = KNNRecommender(k=1).fit(R)
    >>> rec.similar_users(0, n=1)
    (array([1]), array([1.]))
    """

    def __init__(self, k: int = 5, metric: str | Callable = "euclidean") -> None:
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}.")
        self.k = k
        self.metric = metric
        self._distance_fn = _resolve_metric(metric)
        self._R: np.ndarray | None = None
        self.n_users_: int | None = None
        self.n_items_: int | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, R: ArrayLike) -> "KNNRecommender":
        """Fit on a user-item rating matrix.

        Parameters
        ----------
        R : array_like of shape (n_users, n_items)
            Rating matrix.  Rows represent users; columns represent items.
            Missing ratings should be ``0`` or ``numpy.nan`` (NaN is silently
            converted to ``0``).

        Returns
        -------
        self : KNNRecommender

        Raises
        ------
        TypeError
            If *R* cannot be converted to a numeric array.
        ValueError
            If *R* is not 2-D, is empty, or has fewer users than ``k``.
        """
        if R is None:
            raise TypeError("R must not be None.")
        try:
            R_arr = np.asarray(R, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise TypeError(
                "R must be convertible to a numeric 2-D array."
            ) from exc
        if R_arr.ndim != 2:
            raise ValueError(f"R must be 2-D, got shape {R_arr.shape}.")
        if R_arr.shape[0] == 0 or R_arr.shape[1] == 0:
            raise ValueError("R must not be empty.")
        if R_arr.shape[0] <= self.k:
            raise ValueError(
                f"k={self.k} must be strictly less than the number of users "
                f"({R_arr.shape[0]}).  Reduce k or provide more users."
            )

        # Replace NaN with 0 so distance functions always receive finite vectors
        R_arr = np.where(np.isnan(R_arr), 0.0, R_arr)

        self._R = R_arr
        self.n_users_ = R_arr.shape[0]
        self.n_items_ = R_arr.shape[1]
        self._is_fitted = True
        return self

    def similar_users(
        self, user_idx: int, n: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the *n* most similar users to *user_idx*.

        Parameters
        ----------
        user_idx : int
            Index of the target user in the training matrix (0-based).
        n : int, default 5
            Number of similar users to return.

        Returns
        -------
        indices : numpy.ndarray of shape (n,)
            Indices of the *n* nearest other users (ascending distance).
        distances : numpy.ndarray of shape (n,)
            Corresponding distances (lower = more similar).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        TypeError
            If *user_idx* or *n* are not integers.
        ValueError
            If *user_idx* is out of range or *n* < 1.
        """
        self._check_fitted()
        user_idx = self._validate_user_idx(user_idx)
        n = self._validate_n(n, max_val=self.n_users_ - 1, name="n")

        target = self._R[user_idx]
        other_indices = [i for i in range(self.n_users_) if i != user_idx]
        dists = np.array(
            [self._distance_fn(target, self._R[i]) for i in other_indices],
            dtype=np.float64,
        )
        order = np.argsort(dists, kind="stable")[:n]
        chosen = np.array(other_indices)[order]
        return chosen, dists[order]

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> np.ndarray:
        """Recommend the top-*n* items for a user.

        Items are scored by inverse-distance-weighted average rating across the
        *k* most similar users.  If ``exclude_seen=True``, any item the target
        user has already rated (non-zero in the training matrix) is removed
        from the candidate pool before ranking.

        Parameters
        ----------
        user_idx : int
            Index of the target user (0-based).
        n : int, default 10
            Maximum number of items to return.  If fewer scoreable items exist
            the result will be shorter than *n*.
        exclude_seen : bool, default True
            Whether to exclude items the user has already rated.

        Returns
        -------
        item_indices : numpy.ndarray of shape (≤n,)
            Indices of recommended items in descending predicted-rating order.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If *user_idx* is out of range, or *n* < 1.
        """
        self._check_fitted()
        user_idx = self._validate_user_idx(user_idx)
        n = self._validate_n(n, max_val=self.n_items_, name="n")

        neighbor_idx, dists = self.similar_users(user_idx, n=self.k)

        # Inverse-distance weights (exact matches get weight 1, dominate alone)
        zero_mask = dists == 0.0
        if zero_mask.any():
            weights = zero_mask.astype(np.float64)
        else:
            weights = 1.0 / dists
        weights /= weights.sum()

        # Aggregate neighbour ratings into per-item scores
        neighbor_ratings = self._R[neighbor_idx]          # (k, n_items)
        scores = weights @ neighbor_ratings                # (n_items,)

        # Build candidate set
        if exclude_seen:
            seen_mask = self._R[user_idx] != 0.0
            scores = np.where(seen_mask, -np.inf, scores)

        # Return top-n by score (ties broken by lower item index)
        candidate_mask = scores > -np.inf
        if not candidate_mask.any():
            return np.array([], dtype=np.intp)

        ranked = np.argsort(-scores, kind="stable")
        ranked = ranked[scores[ranked] > -np.inf][:n]
        return ranked

    def predict_rating(self, user_idx: int, item_idx: int) -> float:
        """Predict the rating user *user_idx* would give item *item_idx*.

        Parameters
        ----------
        user_idx : int
        item_idx : int

        Returns
        -------
        predicted_rating : float

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If either index is out of range.
        """
        self._check_fitted()
        user_idx = self._validate_user_idx(user_idx)
        if not isinstance(item_idx, int) or isinstance(item_idx, bool):
            raise TypeError(f"item_idx must be an integer, got {type(item_idx).__name__!r}.")
        if not (0 <= item_idx < self.n_items_):
            raise ValueError(
                f"item_idx {item_idx} is out of range for {self.n_items_} items."
            )

        neighbor_idx, dists = self.similar_users(user_idx, n=self.k)
        neighbor_item_ratings = self._R[neighbor_idx, item_idx]

        zero_mask = dists == 0.0
        if zero_mask.any():
            return float(neighbor_item_ratings[zero_mask].mean())
        inv = 1.0 / dists
        return float(np.dot(inv, neighbor_item_ratings) / inv.sum())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet. Call fit() first."
            )

    def _validate_user_idx(self, user_idx: int) -> int:
        if not isinstance(user_idx, int) or isinstance(user_idx, bool):
            raise TypeError(
                f"user_idx must be an integer, got {type(user_idx).__name__!r}."
            )
        if not (0 <= user_idx < self.n_users_):
            raise ValueError(
                f"user_idx {user_idx} is out of range for {self.n_users_} users."
            )
        return user_idx

    def _validate_n(self, n: int, *, max_val: int, name: str) -> int:
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(f"{name} must be a positive integer, got {n!r}.")
        return min(n, max_val)

    def __repr__(self) -> str:
        return f"KNNRecommender(k={self.k}, metric={self.metric!r})"
