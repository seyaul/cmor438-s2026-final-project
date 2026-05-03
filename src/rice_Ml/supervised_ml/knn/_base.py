"""
Shared base class for all KNN estimators.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from rice_Ml.measures_ml.distances import euclidean, taxicab

_BUILTIN_METRICS: dict[str, Callable] = {
    "euclidean": euclidean,
    "taxicab": taxicab,
}


def _resolve_metric(metric: str | Callable) -> Callable:
    """Return a distance callable for the given *metric* specification.

    Parameters
    ----------
    metric : str or callable
        ``"euclidean"`` or ``"taxicab"`` to use the built-ins from
        :mod:`rice_Ml.measures_ml.distances`, or any callable with the
        signature ``f(u, v) -> float``.

    Returns
    -------
    callable

    Raises
    ------
    TypeError
        If *metric* is neither a str nor callable.
    ValueError
        If *metric* is a str not in the known set.
    """
    if callable(metric):
        return metric
    if not isinstance(metric, str):
        raise TypeError(
            f"metric must be a str or callable, got {type(metric).__name__!r}."
        )
    if metric not in _BUILTIN_METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}. "
            f"Choose from {sorted(_BUILTIN_METRICS)} or pass a callable."
        )
    return _BUILTIN_METRICS[metric]


class _KNNBase:
    """Shared fit/predict machinery for KNN-based supervised estimators.

    Subclasses must implement :meth:`fit` and :meth:`predict`.

    Parameters
    ----------
    k : int, default 5
        Number of nearest neighbours to use.
    metric : str or callable, default ``"euclidean"``
        Distance function.  Use ``"euclidean"`` or ``"taxicab"`` to select a
        built-in, or supply any callable ``f(u, v) -> float``.
    """

    def __init__(self, k: int = 5, metric: str | Callable = "euclidean") -> None:
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}.")
        self.k = k
        self.metric = metric
        self._distance_fn = _resolve_metric(metric)
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_X(self, X: ArrayLike, *, name: str = "X") -> np.ndarray:
        """Convert *X* to a validated float64 2-D array."""
        if X is None:
            raise TypeError(f"{name} must not be None.")
        try:
            arr = np.asarray(X, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise TypeError(
                f"{name} must be convertible to a numeric 2-D array."
            ) from exc
        if arr.ndim != 2:
            raise ValueError(
                f"{name} must be 2-D, got shape {arr.shape}."
            )
        if arr.shape[0] == 0:
            raise ValueError(f"{name} must not be empty.")
        if not np.isfinite(arr).all():
            raise ValueError(
                f"{name} must not contain NaN or infinite values."
            )
        return arr

    def _validate_y(
        self, y: ArrayLike, n_samples: int, *, name: str = "y"
    ) -> np.ndarray:
        """Convert *y* to a validated 1-D array aligned with *n_samples*."""
        if y is None:
            raise TypeError(f"{name} must not be None.")
        try:
            arr = np.asarray(y)
        except (ValueError, TypeError) as exc:
            raise TypeError(f"{name} must be array-like.") from exc
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {arr.shape}.")
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"X and {name} must have the same number of samples. "
                f"Got X: {n_samples}, {name}: {arr.shape[0]}."
            )
        return arr

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet. Call fit() first."
            )

    def _check_predict_X(self, X: np.ndarray) -> None:
        """Verify prediction input is feature-compatible with training data."""
        if X.shape[1] != self._X_train.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} feature(s) but the model was fitted with "
                f"{self._X_train.shape[1]}."
            )

    def _neighbor_indices(self, x: np.ndarray) -> np.ndarray:
        """Return the indices of the *k* nearest training samples to *x*."""
        distances = np.fromiter(
            (self._distance_fn(x, row) for row in self._X_train),
            dtype=np.float64,
            count=len(self._X_train),
        )
        return np.argsort(distances, kind="stable")[: self.k]

    def _neighbor_indices_and_distances(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(indices, distances)`` for the *k* nearest training samples."""
        distances = np.fromiter(
            (self._distance_fn(x, row) for row in self._X_train),
            dtype=np.float64,
            count=len(self._X_train),
        )
        idx = np.argsort(distances, kind="stable")[: self.k]
        return idx, distances[idx]
