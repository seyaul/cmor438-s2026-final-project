"""
K-Nearest Neighbours classifier.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from ._base import _KNNBase


class KNNClassifier(_KNNBase):
    """K-Nearest Neighbours classifier.

    Predicts the class label of a query point by majority vote among its *k*
    nearest training neighbours.  Ties are broken by selecting the label that
    appears first in sorted order (i.e. the smallest class label wins).

    Parameters
    ----------
    k : int, default 5
        Number of nearest neighbours to consider.
    metric : str or callable, default ``"euclidean"``
        Distance function.  Accepts ``"euclidean"`` or ``"taxicab"`` for the
        built-ins in :mod:`rice_Ml.measures_ml.distances`, or any callable
        with signature ``f(u, v) -> float``.

    Attributes
    ----------
    classes_ : numpy.ndarray, shape (n_classes,)
        Unique class labels seen during :meth:`fit`, in sorted order.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> from rice_Ml.supervised_ml.knn.classifier import KNNClassifier
    >>> X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    >>> y_train = np.array([0, 0, 0, 1])
    >>> clf = KNNClassifier(k=3).fit(X_train, y_train)
    >>> clf.predict([[0.1, 0.1]])
    array([0])
    >>> clf.predict([[4.9, 4.9]])
    array([1])
    """

    def __init__(self, k: int = 5, metric: str | Callable = "euclidean") -> None:
        super().__init__(k=k, metric=metric)
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNClassifier":
        """Store training data and derive class metadata.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Training feature matrix.  Must be finite and numeric.
        y : array_like of shape (n_samples,)
            Target class labels.  May be integers, floats, or strings.

        Returns
        -------
        self : KNNClassifier

        Raises
        ------
        TypeError
            If *X* or *y* cannot be converted to arrays.
        ValueError
            If *X* is not 2-D, *y* is not 1-D, their sample counts differ,
            *X* contains non-finite values, or ``k`` exceeds the sample count.
        """
        X_arr = self._validate_X(X)
        y_arr = self._validate_y(y, X_arr.shape[0])
        if self.k > X_arr.shape[0]:
            raise ValueError(
                f"k={self.k} exceeds the number of training samples "
                f"({X_arr.shape[0]}). Reduce k."
            )
        self._X_train = X_arr
        self._y_train = y_arr
        self.classes_ = np.unique(y_arr)
        self.n_features_in_ = X_arr.shape[1]
        self._is_fitted = True
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict class labels for *X*.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples,)
            Predicted class label for each sample.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If the number of features in *X* does not match training data.
        """
        self._check_fitted()
        X_arr = self._validate_X(X)
        self._check_predict_X(X_arr)

        predictions = []
        for x in X_arr:
            idx = self._neighbor_indices(x)
            neighbor_labels = self._y_train[idx]
            # Counter.most_common is stable — first-seen wins ties; we pre-sort
            # so the lowest label wins, matching sklearn convention.
            vote = Counter(sorted(neighbor_labels)).most_common(1)[0][0]
            predictions.append(vote)

        return np.array(predictions, dtype=self._y_train.dtype)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict class membership probabilities for *X*.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)

        Returns
        -------
        proba : numpy.ndarray of shape (n_samples, n_classes)
            Each row sums to 1.  Column *j* is the fraction of the *k*
            neighbours that belong to ``self.classes_[j]``.

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

        class_to_col = {c: i for i, c in enumerate(self.classes_)}
        proba = np.zeros((len(X_arr), len(self.classes_)), dtype=np.float64)

        for i, x in enumerate(X_arr):
            idx = self._neighbor_indices(x)
            for label in self._y_train[idx]:
                proba[i, class_to_col[label]] += 1
            proba[i] /= self.k

        return proba

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy on *(X, y)*.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
        y : array_like of shape (n_samples,)

        Returns
        -------
        accuracy : float in [0, 1]
        """
        self._check_fitted()
        y_arr = np.asarray(y)
        return float(np.mean(self.predict(X) == y_arr))

    def __repr__(self) -> str:
        return f"KNNClassifier(k={self.k}, metric={self.metric!r})"
