"""Binary decision tree classifier with gini/entropy splitting."""

import numpy as np

from rice_Ml.base.base_model import BaseModel


class Node:
    """Single node in the tree — either a decision split or a leaf."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.proba = proba  # class probability vector at leaf, shape (n_classes,)

    def is_leaf(self):
        return self.value is not None


class DecisionTree(BaseModel):
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

    def predict_proba(self, X):
        """Return class probability estimates; shape (n_samples, n_classes)."""
        return np.array([self._traverse_proba(row, self.root_) for row in X])

    def score(self, X, y):
        """Return accuracy on (X, y)."""
        from rice_Ml.metrics import accuracy
        return accuracy(self.predict(X), np.array(y))

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _resolve_max_features(self, n_features):
        """Map self.max_features to a concrete feature count."""
        if self.max_features is None:
            return n_features
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        return min(self.max_features, n_features)

    def _leaf(self, y):
        """Create a leaf node storing both the majority class and class probabilities."""
        counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        return Node(value=self._majority_class(y), proba=counts / counts.sum())

    def _build_tree(self, X, y, depth):
        """Recursively split (X, y); return a leaf when stopping criteria are met."""
        if self._impurity(y) == 0 or len(y) < self.min_samples_split:
            return self._leaf(y)
        if self.max_depth is not None and depth >= self.max_depth:
            return self._leaf(y)

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return self._leaf(y)

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

        y_int = y.astype(np.intp)
        n_classes = y_int.max() + 1

        for feature in feature_indices:
            col = X[:, feature]
            thresholds = np.unique(np.percentile(col, np.linspace(0, 100, 20)))

            order = np.argsort(col)
            col_sorted = col[order]
            y_sorted = y_int[order]

            # cumulative class counts: cumsum[i] = counts in y_sorted[:i+1]
            onehot = np.zeros((n_total, n_classes), dtype=np.int32)
            onehot[np.arange(n_total), y_sorted] = 1
            cumsum = onehot.cumsum(axis=0)  # O(n) once per feature
            total_counts = cumsum[-1]

            for thresh in thresholds:
                n_left = int(np.searchsorted(col_sorted, thresh, side="right"))
                if n_left == 0 or n_left == n_total:
                    continue
                n_right = n_total - n_left
                left_counts  = cumsum[n_left - 1].astype(float)
                right_counts = (total_counts - cumsum[n_left - 1]).astype(float)

                p_l = left_counts  / n_left
                p_r = right_counts / n_right
                g_l = 1.0 - np.dot(p_l, p_l) if self.criterion == "gini" else -np.sum(p_l[p_l>0] * np.log2(p_l[p_l>0]))
                g_r = 1.0 - np.dot(p_r, p_r) if self.criterion == "gini" else -np.sum(p_r[p_r>0] * np.log2(p_r[p_r>0]))

                gain = parent_impurity - (n_left / n_total * g_l + n_right / n_total * g_r)
                if gain > best_gain:
                    best_feature, best_thresh, best_gain = feature, thresh, gain

        return best_feature, best_thresh

    def _impurity(self, y):
        """Compute gini or entropy impurity of label array y."""
        counts = np.bincount(y.view(np.intp) if y.dtype.kind == 'i' else y.astype(np.intp))
        counts = counts[counts > 0]
        p = counts / len(y)
        if self.criterion == "gini":
            return 1.0 - np.dot(p, p)
        return -np.sum(p * np.log2(p))

    def _majority_class(self, y):
        """Return the most frequent label in y."""
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def _traverse(self, x, node):
        """Walk the tree from node to the matching leaf and return its label."""
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def _traverse_proba(self, x, node):
        """Walk the tree and return the leaf's class probability vector."""
        if node.is_leaf():
            return node.proba
        if x[node.feature] <= node.threshold:
            return self._traverse_proba(x, node.left)
        return self._traverse_proba(x, node.right)
