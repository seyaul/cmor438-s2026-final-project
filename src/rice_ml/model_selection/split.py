import numpy as np
from typing import Tuple, List, Union, Optional

def train_test_split(
    *arrays,
    test_size: Union[float, int, None] = None,
    train_size: Union[float, int, None] = None,
    shuffle: bool = True,
    random_state: Optional[int] = None,
    stratify: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Split arrays or matrices into random train and test subsets.

    Parameters
    ----------
    *arrays : sequence of array-like
        Arrays to split. All must have the same first dimension (n_samples).
    test_size : float, int, or None, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If train_size is also None, test_size=0.25.
    train_size : float, int, or None, default=None
        If float, between 0.0 and 1.0, represents proportion of dataset to
        include in train split. If int, absolute number of train samples.
        If None, the value is automatically set to the complement of test_size.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    random_state : int or None, default=None
        Seed for reproducible shuffling.
    stratify : array-like or None, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels. (Currently not implemented – placeholder.)

    Returns
    -------
    splitting : list of np.ndarray
        List containing train-test split of inputs. For two arrays,
        returns [X_train, X_test, y_train, y_test].

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 1, 0, 1])
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    n_samples = arrays[0].shape[0]
    for arr in arrays[1:]:
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"All arrays must have same first dimension. "
                f"Got {n_samples} and {arr.shape[0]}."
            )

    # Determine test_size and train_size
    if test_size is None and train_size is None:
        test_size = 0.25

    if test_size is not None:
        if isinstance(test_size, float):
            if test_size < 0.0 or test_size > 1.0:
                raise ValueError("test_size float must be between 0.0 and 1.0")
            n_test = int(np.ceil(test_size * n_samples))
        else:
            n_test = test_size
    else:
        n_test = None

    if train_size is not None:
        if isinstance(train_size, float):
            if train_size < 0.0 or train_size > 1.0:
                raise ValueError("train_size float must be between 0.0 and 1.0")
            n_train = int(np.ceil(train_size * n_samples))
        else:
            n_train = train_size
    else:
        n_train = None

    # Resolve sizes
    if n_test is None:
        n_test = n_samples - n_train
    if n_train is None:
        n_train = n_samples - n_test

    if n_train + n_test > n_samples:
        raise ValueError(
            f"Train size ({n_train}) + test size ({n_test}) exceeds "
            f"number of samples ({n_samples})"
        )

    # Generate indices
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)

    train_indices = indices[:n_train]
    test_indices = indices[n_train : n_train + n_test]

    # Split each array
    result = []
    for arr in arrays:
        result.append(arr[train_indices])
        result.append(arr[test_indices])
    return result


class KFold:
    """
    K-Folds cross-validator.

    Provides train/test indices to split data into train/test sets.
    Split dataset into k consecutive folds (without shuffling by default).
    Each fold is used once as a test set while the k-1 remaining folds form the training set.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
    random_state : int or None, default=None
        Seed for reproducible shuffling.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,) or None
            Target values (ignored, present for API consistency).

        Yields
        ------
        train : np.ndarray
            Training indices for that split.
        test : np.ndarray
            Testing indices for that split.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, test_idx
            current = stop


def cross_val_score(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: Optional[Union[int, KFold]] = None,
    scoring: Optional[str] = None,
) -> List[float]:
    """
    Evaluate a score by cross-validation.

    Parameters
    ----------
    estimator : object
        Model with fit() and predict() methods.
    X : np.ndarray
        Features.
    y : np.ndarray
        Target.
    cv : int or KFold instance, default=None
        Number of folds (default=5) or a KFold object.
    scoring : str or None, default=None
        Scoring metric. If None, uses estimator's score() method.

    Returns
    -------
    scores : list of float
        Array of scores for each run of cross-validation.
    """
    if cv is None:
        cv = 5
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True)

    scores = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        estimator_clone = estimator.__class__(**estimator.__dict__)
        estimator_clone.fit(X_train, y_train)

        if scoring is None:
            score = estimator_clone.score(X_test, y_test)
        else:
            y_pred = estimator_clone.predict(X_test)
            if scoring == 'r2':
                from ..metrics.regression import r2_score
                score = r2_score(y_test, y_pred)
            elif scoring == 'rmse':
                from ..metrics.regression import rmse
                score = -rmse(y_test, y_pred)  # negative so "higher is better"
            else:
                raise ValueError(f"Unknown scoring: {scoring}")
        scores.append(score)
    return scores