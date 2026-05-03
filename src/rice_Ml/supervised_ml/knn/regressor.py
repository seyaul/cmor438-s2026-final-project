"""
K-Nearest Neighbours regressor.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from ._base import _KNNBase


class KNNRegressor(_KNNBase):
    """K-Nearest Neighbours regressor.

    Predicts a continuous target value for a query point by averaging the
    targets of its *k* nearest training neighbours.  When ``weights="distance"``
    each neighbour's contribution is proportional to ``1 / distance``; exact
    matches (distance = 0) receive full weight and their average is returned
    directly.

    Parameters
    ----------
    k : int, default 5
        Number of nearest neighbours to consider.
    metric : str or callable, default ``"euclidean"``
        Distance function.  Accepts ``"euclidean"`` or ``"taxicab"`` for the
        built-ins in :mod:`rice_Ml.measures_ml.distances`, or any callable
        with signature ``f(u, v) -> float``.
    weights : {"uniform", "distance"}, default ``"uniform"``
        Weighting strategy for aggregating neighbour targets:

        - ``"uniform"``  — arithmetic mean of neighbour targets.
        - ``"distance"`` — inverse-distance weighted mean.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> from rice_Ml.supervised_ml.knn.regressor import KNNRegressor
    >>> X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y_train = np.array([0.0, 1.0, 4.0, 9.0])
    >>> reg = KNNRegressor(k=2).fit(X_train, y_train)
    >>> reg.predict([[1.5]])   # neighbours: y=1.0 and y=4.0  →  mean = 2.5
    array([2.5])
    """

    def __init__(
        self,
        k: int = 5,
        metric: str | Callable = "euclidean",
        weights: str = "uniform",
    ) -> None:
        super().__init__(k=k, metric=metric)
        if weights not in ("uniform", "distance"):
            raise ValueError(
                f"weights must be 'uniform' or 'distance', got {weights!r}."
            )
        self.weights = weights
        self.n_features_in_: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNRegressor":
        """Store training data.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Training feature matrix.  Must be finite and numeric.
        y : array_like of shape (n_samples,)
            Continuous target values.  Must be finite and numeric.

        Returns
        -------
        self : KNNRegressor

        Raises
        ------
        TypeError
            If *X* or *y* cannot be converted to numeric arrays.
        ValueError
            If shapes are incompatible, values are non-finite, or ``k``
            exceeds the sample count.
        """
        X_arr = self._validate_X(X)
        y_arr = self._validate_y(y, X_arr.shape[0])

        # y must be numeric for regression
        try:
            y_float = y_arr.astype(np.float64)
        except (ValueError, TypeError) as exc:
            raise TypeError("y must contain numeric values for regression.") from exc
        if not np.isfinite(y_float).all():
            raise ValueError("y must not contain NaN or infinite values.")

        if self.k > X_arr.shape[0]:
            raise ValueError(
                f"k={self.k} exceeds the number of training samples "
                f"({X_arr.shape[0]}). Reduce k."
            )
        self._X_train = X_arr
        self._y_train = y_float
        self.n_features_in_ = X_arr.shape[1]
        self._is_fitted = True
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict continuous target values for *X*.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples,) of float64

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If feature counts don't match training data.
        """
        self._check_fitted()
        X_arr = self._validate_X(X)
        self._check_predict_X(X_arr)

        predictions = np.empty(len(X_arr), dtype=np.float64)

        for i, x in enumerate(X_arr):
            idx, dists = self._neighbor_indices_and_distances(x)
            neighbor_targets = self._y_train[idx]

            if self.weights == "uniform":
                predictions[i] = neighbor_targets.mean()
            else:  # "distance"
                zero_mask = dists == 0.0
                if zero_mask.any():
                    # Exact match(es) — return their mean, ignoring others
                    predictions[i] = neighbor_targets[zero_mask].mean()
                else:
                    inv_dists = 1.0 / dists
                    predictions[i] = np.dot(inv_dists, neighbor_targets) / inv_dists.sum()

        return predictions

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the coefficient of determination R² on *(X, y)*.

        R² = 1 - SS_res / SS_tot.  A perfect predictor scores 1.0; a
        constant-mean predictor scores 0.0; negative values indicate the
        model is worse than the mean.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
        y : array_like of shape (n_samples,)

        Returns
        -------
        r2 : float
        """
        self._check_fitted()
        y_true = np.asarray(y, dtype=np.float64)
        y_pred = self.predict(X)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return float(1.0 - ss_res / ss_tot)

    def __repr__(self) -> str:
        return (
            f"KNNRegressor(k={self.k}, metric={self.metric!r}, "
            f"weights={self.weights!r})"
        )
