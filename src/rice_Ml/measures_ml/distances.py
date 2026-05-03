"""
Distance metrics for rice_Ml.

All functions accept array-like inputs and return a non-negative float.
Inputs are validated eagerly so callers get clear error messages.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_vectors(u: ArrayLike, v: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert *u* and *v* to 1-D float64 arrays and validate compatibility.

    Parameters
    ----------
    u : array_like
        First input vector.
    v : array_like
        Second input vector.

    Returns
    -------
    u_arr : numpy.ndarray, shape (n,)
    v_arr : numpy.ndarray, shape (n,)

    Raises
    ------
    TypeError
        If either input cannot be converted to a numeric NumPy array.
    ValueError
        If either array is not 1-D, is empty, or the two arrays differ in
        length.
    """
    if u is None or v is None:
        raise TypeError(
            f"Both vectors must be convertible to numeric arrays. Got types "
            f"{type(u).__name__!r} and {type(v).__name__!r}."
        )

    try:
        u_arr = np.asarray(u, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"Both vectors must be convertible to numeric arrays. Got types "
            f"{type(u).__name__!r} and {type(v).__name__!r}."
        ) from exc

    if u_arr.ndim != 1 or v_arr.ndim != 1:
        raise ValueError(
            f"Both vectors must be 1-D. Got shapes {u_arr.shape} and {v_arr.shape}."
        )
    if u_arr.size == 0 or v_arr.size == 0:
        raise ValueError("Vectors must not be empty.")
    if u_arr.shape != v_arr.shape:
        raise ValueError(
            f"Vectors must have the same length. "
            f"Got {u_arr.shape[0]} and {v_arr.shape[0]}."
        )
    if not (np.isfinite(u_arr).all() and np.isfinite(v_arr).all()):
        raise ValueError("Vectors must not contain NaN or infinite values.")

    return u_arr, v_arr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def euclidean(u: ArrayLike, v: ArrayLike) -> float:
    r"""Compute the Euclidean (L2) distance between two vectors.

    The Euclidean distance between vectors **u** and **v** in
    :math:`\mathbb{R}^n` is defined as

    .. math::

        d(u, v) = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}

    Parameters
    ----------
    u : array_like of shape (n,)
        First input vector. Must be 1-D, non-empty, finite, and numeric.
    v : array_like of shape (n,)
        Second input vector. Must match *u* in length.

    Returns
    -------
    distance : float
        Non-negative Euclidean distance between *u* and *v*.

    Raises
    ------
    TypeError
        If either input cannot be converted to a numeric array.
    ValueError
        If either input is not 1-D, is empty, contains non-finite values,
        or the two inputs differ in length.

    Examples
    --------
    Distance between two 2-D points:

    >>> euclidean([0, 0], [3, 4])
    5.0

    Distance between identical vectors is zero:

    >>> euclidean([1, 2, 3], [1, 2, 3])
    0.0

    Works with NumPy arrays directly:

    >>> import numpy as np
    >>> euclidean(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    1.4142135623730951

    Mismatched lengths raise ``ValueError``:

    >>> euclidean([1, 2], [1, 2, 3])
    Traceback (most recent call last):
        ...
    ValueError: Vectors must have the same length. Got 2 and 3.

    Non-numeric input raises ``TypeError``:

    >>> euclidean(["a", "b"], [1, 2])
    Traceback (most recent call last):
        ...
    TypeError: Both vectors must be convertible to numeric arrays. Got types 'list' and 'list'.
    """
    u_arr, v_arr = _validate_vectors(u, v)
    diff = u_arr - v_arr
    return float(np.sqrt(np.dot(diff, diff)))


def taxicab(u: ArrayLike, v: ArrayLike) -> float:
    r"""Compute the taxicab (Manhattan / L1) distance between two vectors.

    Also known as the *Manhattan distance* or *city-block distance*, the
    taxicab distance between vectors **u** and **v** in :math:`\mathbb{R}^n`
    is defined as

    .. math::

        d(u, v) = \sum_{i=1}^{n} |u_i - v_i|

    Parameters
    ----------
    u : array_like of shape (n,)
        First input vector. Must be 1-D, non-empty, finite, and numeric.
    v : array_like of shape (n,)
        Second input vector. Must match *u* in length.

    Returns
    -------
    distance : float
        Non-negative taxicab distance between *u* and *v*.

    Raises
    ------
    TypeError
        If either input cannot be converted to a numeric array.
    ValueError
        If either input is not 1-D, is empty, contains non-finite values,
        or the two inputs differ in length.

    Examples
    --------
    Classic 2-D grid example (3 blocks east, 4 blocks north):

    >>> taxicab([0, 0], [3, 4])
    7.0

    Distance between identical vectors is zero:

    >>> taxicab([1, 2, 3], [1, 2, 3])
    0.0

    Works with NumPy arrays:

    >>> import numpy as np
    >>> taxicab(np.array([1.0, 2.0, 3.0]), np.array([4.0, 0.0, 3.0]))
    5.0

    Mismatched lengths raise ``ValueError``:

    >>> taxicab([1, 2], [1, 2, 3])
    Traceback (most recent call last):
        ...
    ValueError: Vectors must have the same length. Got 2 and 3.

    Non-numeric input raises ``TypeError``:

    >>> taxicab(["a", "b"], [1, 2])
    Traceback (most recent call last):
        ...
    TypeError: Both vectors must be convertible to numeric arrays. Got types 'list' and 'list'.
    """
    u_arr, v_arr = _validate_vectors(u, v)
    return float(np.sum(np.abs(u_arr - v_arr)))
