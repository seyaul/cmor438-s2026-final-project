"""
Baseline 1-D Convolutional Neural Network for frame-level classification.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ...base.base_model import BaseModel
from .layers import Conv1D, MaxPool1D, Flatten, Dense


# ---------------------------------------------------------------------------
# Module-level helpers (no state, no import cost)
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def _cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    log_p = np.log(probs[np.arange(len(y)), y] + 1e-15)
    return float(-log_p.mean())


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CNN(BaseModel):
    """Baseline 1-D CNN for frame-level audio feature classification.

    Treats each input sample as a 1-D signal with a single channel, then
    learns local patterns via two convolutional layers before classifying.

    Fixed architecture
    ------------------
    Input (batch, n_features)
        → reshape → (batch, 1, n_features)
        → Conv1D(1→16, k=3) → ReLU
        → MaxPool1D(2)
        → Conv1D(16→32, k=3) → ReLU
        → Flatten
        → Dense(flat→64) → ReLU
        → Dense(64→n_classes) → Softmax

    For the default 18-feature input this gives:
        Conv1D output: (batch, 16, 16)
        After pool:    (batch, 16,  8)
        After conv2:   (batch, 32,  6)
        Flat:          (batch, 192)
        Dense1:        (batch, 64)
        Dense2:        (batch, n_classes)

    Parameters
    ----------
    learning_rate : float, default 1e-3
    epochs : int, default 50
    batch_size : int, default 64
    random_state : int or None, default None

    Attributes
    ----------
    classes_ : numpy.ndarray
        Unique class labels seen during fit, in sorted order.
    loss_history_ : list of float
        Mean cross-entropy loss per training epoch.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        random_state: int | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self.loss_history_: list[float] = []

        # Populated by _build()
        self._conv1: Conv1D | None = None
        self._pool: MaxPool1D | None = None
        self._conv2: Conv1D | None = None
        self._flatten: Flatten | None = None
        self._dense1: Dense | None = None
        self._dense2: Dense | None = None

        # ReLU pre-activation caches (set during forward)
        self._z1: np.ndarray | None = None
        self._z2: np.ndarray | None = None
        self._z3: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "CNN":
        """Train the CNN on (X, y).

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Feature matrix. Each row is one frame (e.g. 18 audio features).
        y : array_like of shape (n_samples,)
            Integer class labels. Labels need not start at 0 — they are
            remapped internally.

        Returns
        -------
        self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X_arr.shape}.")
        if y_arr.ndim != 1 or len(y_arr) != len(X_arr):
            raise ValueError("y must be 1-D with the same number of rows as X.")

        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_mapped = np.array([label_map[c] for c in y_arr], dtype=np.int32)

        self._build(n_features=X_arr.shape[1], n_classes=n_classes)
        self.loss_history_ = []

        n = len(X_arr)
        for _ in range(self.epochs):
            idx = np.random.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, self.batch_size):
                batch_idx = idx[start : start + self.batch_size]
                epoch_loss += self._train_step(X_arr[batch_idx], y_mapped[batch_idx])
                n_batches += 1
            self.loss_history_.append(epoch_loss / n_batches)

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict class labels for X.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Labels drawn from the original label space seen during fit.
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_classes)
            Rows sum to 1. Column order matches ``self.classes_``.
        """
        self._check_fitted()
        return self._forward(np.asarray(X, dtype=np.float64))

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy on (X, y).

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
        y : array_like of shape (n_samples,)

        Returns
        -------
        float in [0, 1]
        """
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def __repr__(self) -> str:
        return (
            f"CNN(learning_rate={self.learning_rate}, epochs={self.epochs}, "
            f"batch_size={self.batch_size})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build(self, n_features: int, n_classes: int) -> None:
        # Compute flattened size after both conv + pool layers
        after_pool = (n_features - 2) // 2   # Conv1(k=3) then MaxPool(2)
        after_conv2 = after_pool - 2          # Conv2(k=3)
        flat_size = 32 * after_conv2

        if flat_size <= 0:
            raise ValueError(
                f"n_features={n_features} is too small for this architecture. "
                "Need at least 8 features."
            )

        self._conv1 = Conv1D(in_channels=1, out_channels=16, kernel_size=3)
        self._pool = MaxPool1D(pool_size=2)
        self._conv2 = Conv1D(in_channels=16, out_channels=32, kernel_size=3)
        self._flatten = Flatten()
        self._dense1 = Dense(flat_size, 64)
        self._dense2 = Dense(64, n_classes)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        # (batch, n_features) → (batch, 1, n_features)
        out = X[:, np.newaxis, :]

        out = self._conv1.forward(out)
        self._z1 = out.copy()
        out = _relu(out)

        out = self._pool.forward(out)

        out = self._conv2.forward(out)
        self._z2 = out.copy()
        out = _relu(out)

        out = self._flatten.forward(out)

        out = self._dense1.forward(out)
        self._z3 = out.copy()
        out = _relu(out)

        out = self._dense2.forward(out)
        return _softmax(out)

    def _train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        probs = self._forward(X)
        loss = _cross_entropy(probs, y)

        # Combined softmax + cross-entropy gradient
        n = len(y)
        dout = probs.copy()
        dout[np.arange(n), y] -= 1.0
        dout /= n

        dout = self._dense2.backward(dout)
        dout = dout * _relu_grad(self._z3)
        dout = self._dense1.backward(dout)
        dout = self._flatten.backward(dout)
        dout = dout * _relu_grad(self._z2)
        dout = self._conv2.backward(dout)
        dout = self._pool.backward(dout)
        dout = dout * _relu_grad(self._z1)
        self._conv1.backward(dout)

        # SGD update
        for layer in (self._conv1, self._conv2, self._dense1, self._dense2):
            for key in layer.params:
                layer.params[key] -= self.learning_rate * layer.grads[key]

        return loss

    def _check_fitted(self) -> None:
        if self.classes_ is None:
            raise RuntimeError("This CNN instance is not fitted yet. Call fit() first.")
