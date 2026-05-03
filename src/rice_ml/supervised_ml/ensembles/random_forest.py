"""Random Forest classifier via bootstrap aggregation of decision trees."""

import numpy as np

from rice_ml.supervised_ml.DecisionTree.decision_tree import DecisionTree


class RandomForest:
    """
    Ensemble of decision trees trained on bootstrap samples with random feature subsets.

    Each tree sees a different random draw of rows (with replacement) and at every
    split considers only a random subset of features ('sqrt' of total by default).
    Predictions are made by majority vote across all trees.

    Attributes set after fit():
        estimators_ — list of fitted DecisionTree objects
        classes_    — sorted array of unique class labels
        n_features_ — number of input features
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        criterion="gini",
        random_state=None,
    ):
        """
        n_estimators      — number of trees in the forest
        max_depth         — maximum depth of each tree; None grows until pure
        min_samples_split — minimum samples required to split a node
        max_features      — features considered per split: 'sqrt', 'log2', int, or None (all)
        criterion         — impurity measure passed to each tree: 'gini' or 'entropy'
        random_state      — seed for reproducibility
        """
        if not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError(f"n_estimators must be a positive integer, got {n_estimators!r}.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state

    def fit(self, X, y):
        """Bootstrap-sample the data and fit one DecisionTree per estimator. Returns self."""
        X = np.array(X)
        y = np.array(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length, got {len(X)} and {len(y)}.")

        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]

        rng = np.random.default_rng(self.random_state)
        # draw a seed for each tree so they are independent but the forest is reproducible
        seeds = rng.integers(0, 2**31, size=self.n_estimators)

        n_samples = len(X)
        self.estimators_ = []
        for seed in seeds:
            tree_rng = np.random.default_rng(seed)
            indices = tree_rng.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=int(seed),
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """Return majority-vote class label for each row of X."""
        votes = np.array([tree.predict(X) for tree in self.estimators_])
        return np.array([self._majority_vote(votes[:, i]) for i in range(len(X))])

    def predict_proba(self, X):
        """Return fraction of trees voting for each class; shape (n_samples, n_classes)."""
        votes = np.array([tree.predict(X) for tree in self.estimators_])
        n_samples = len(X)
        proba = np.zeros((n_samples, len(self.classes_)))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for i in range(n_samples):
            for vote in votes[:, i]:
                if vote in class_to_idx:
                    proba[i, class_to_idx[vote]] += 1
        proba /= self.n_estimators
        return proba

    def score(self, X, y):
        """Return accuracy on (X, y)."""
        return np.mean(self.predict(X) == np.array(y))

    def _majority_vote(self, votes):
        labels, counts = np.unique(votes, return_counts=True)
        return labels[np.argmax(counts)]
