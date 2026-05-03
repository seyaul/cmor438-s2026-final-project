"""Binary decision tree classifier with gini/entropy splitting."""

import numpy as np


class Node:
    """Single node in the tree — either a decision split or a leaf."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """
    Binary decision tree classifier built by recursive greedy splitting.

    Attributes set after fit():
        root_       — root Node of the fitted tree
        classes_    — sorted array of unique class labels seen during fit
        n_features_ — number of features in the training data
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
        random_state=None,
    ):
        """
        max_depth         — maximum tree depth; None grows until pure
        min_samples_split — minimum samples required to attempt a split
        criterion         — impurity measure: 'gini' or 'entropy'
        max_features      — features considered per split: None (all), int, 'sqrt', or 'log2'
        random_state      — seed for feature subsampling RNG
        """
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 1):
            raise ValueError(f"max depth must be a positive integer, got {max_depth!r}.")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError(
                f"minimum samples to split on must be an integer >= 2, got {min_samples_split!r}."
            )
        if criterion not in ("gini", "entropy"):
            raise ValueError(f'criterion must be "gini" or "entropy", got {criterion!r}.')
        if max_features is not None and max_features not in ("sqrt", "log2") and (
            not isinstance(max_features, int) or max_features < 1
        ):
            raise ValueError(
                f"max_features must be None, 'sqrt', 'log2', or a positive int, got {max_features!r}."
            )

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._n_features_split = None

    def fit(self, X, y):
        """Build the decision tree on (X, y). Returns self."""
        X = np.array(X)
        y = np.array(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length, got {len(X)} and {len(y)}.")

        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self._n_features_split = self._resolve_max_features(X.shape[1])

        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        """Classify each row of X; returns a 1-D label array."""
        return np.array([self._traverse(row, self.root_) for row in X])

    def score(self, X, y):
        """Return accuracy on (X, y)."""
        return np.mean(self.predict(X) == np.array(y))

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _resolve_max_features(self, n_features):
        if self.max_features is None:
            return n_features
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        return min(self.max_features, n_features)

    def _build_tree(self, X, y, depth):
        if self._impurity(y) == 0 or len(y) < self.min_samples_split:
            return Node(value=self._majority_class(y))
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=self._majority_class(y))

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value=self._majority_class(y))

        left_mask = X[:, feature] <= threshold
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def _best_split(self, X, y):
        """Return (feature_index, threshold) with highest information gain, or (None, None)."""
        n_total = len(y)
        parent_impurity = self._impurity(y)
        best_feature, best_thresh, best_gain = None, None, float("-inf")

        n_features_split = self._n_features_split or X.shape[1]
        feature_indices = self._rng.choice(X.shape[1], size=n_features_split, replace=False)

        for feature in feature_indices:
            for thresh in np.unique(X[:, feature]):
                left_mask = X[:, feature] <= thresh
                n_left = left_mask.sum()
                n_right = n_total - n_left
                if n_left == 0 or n_right == 0:
                    continue

                gain = parent_impurity - (
                    n_left / n_total * self._impurity(y[left_mask])
                    + n_right / n_total * self._impurity(y[~left_mask])
                )
                if gain > best_gain:
                    best_feature, best_thresh, best_gain = feature, thresh, gain

        return best_feature, best_thresh

    def _impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    def _majority_class(self, y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)
