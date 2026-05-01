"""
decision_tree.py

a binary decision tree classifier that recursively splits the feature space
by finding the best threshold on a single feature at each node.

labels can be any discrete class values. for regression, extend predict()
to average leaf values instead of taking the majority vote.
"""

import numpy as np


class Node:
    """represents a single node in the tree — either a split or a leaf."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # internal node fields
        self.feature = feature       # index of feature to split on
        self.threshold = threshold   # split threshold value

        self.left = left             # left child Node (feature <= threshold)
        self.right = right           # right child Node (feature > threshold)

        # leaf node field — majority class label
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """
    binary decision tree classifier built by recursive greedy splitting.

    splits are chosen to minimise impurity (gini or entropy) at each node.
    training stops when max_depth is reached or a node has too few samples
    to split.

    attributes set after fit():
        root_        — the root Node of the fitted tree
        classes_     — sorted array of unique class labels seen during fit
        n_features_  — number of features in the training data
    """

    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        """
        max_depth         — maximum tree depth; None means grow until pure
        min_samples_split — minimum samples required to attempt a split
        criterion         — impurity measure: 'gini' or 'entropy'
        """
        # TODO: validate parameters and assign to self
        raise NotImplementedError

    def fit(self, X, y):
        """
        build the decision tree on training data X and labels y.

        X must be a 2-D array of shape (n_samples, n_features).
        y must be a 1-D array of length n_samples.
        returns self so you can chain: DecisionTree().fit(X, y).predict(X)
        """
        # TODO: convert X and y to numpy arrays, validate shapes,
        #       set self.classes_ and self.n_features_,
        #       then call self._build_tree(X, y, depth=0) and store as self.root_
        raise NotImplementedError

    def predict(self, X):
        """
        classify each row of X and return a 1-D label array.
        """
        # TODO: call self._traverse(row, self.root_) for each row in X
        raise NotImplementedError

    def score(self, X, y):
        """
        return fraction of correctly classified samples (accuracy).
        """
        # TODO: return np.mean(self.predict(X) == np.array(y))
        raise NotImplementedError

    # ------------------------------------------------------------------
    # internal helpers — implement these to build the tree
    # ------------------------------------------------------------------

    def _build_tree(self, X, y, depth):
        """
        recursively build and return a Node for the data at this split.

        base cases: pure node, too few samples, or depth limit reached
        → return a leaf Node with the majority class.
        """
        # TODO: check stopping conditions (pure, min_samples_split, max_depth)
        #       find the best split with _best_split(X, y)
        #       partition X and y, recurse on left/right subsets
        raise NotImplementedError

    def _best_split(self, X, y):
        """
        search every feature and threshold for the split that most reduces
        impurity. return (feature_index, threshold) or (None, None) if no
        valid split exists.
        """
        # TODO: iterate over features and sorted unique thresholds,
        #       compute impurity gain for each candidate split,
        #       return the feature/threshold with the highest gain
        raise NotImplementedError

    def _impurity(self, y):
        """
        compute impurity of label array y using self.criterion.
        returns a float (lower is purer).
        """
        # TODO: implement gini = 1 - sum(p_k^2) and
        #       entropy = -sum(p_k * log2(p_k))
        raise NotImplementedError

    def _majority_class(self, y):
        """return the most common label in y."""
        # TODO: use np.unique with return_counts=True and return the argmax label
        raise NotImplementedError

    def _traverse(self, x, node):
        """
        walk a single sample x down the tree from node, return leaf value.
        """
        # TODO: if node.is_leaf() return node.value
        #       else recurse left or right based on x[node.feature] <= node.threshold
        raise NotImplementedError
