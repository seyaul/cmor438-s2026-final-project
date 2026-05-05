"""AdaBoost classifier using decision stumps as weak learners (SAMME algorithm)."""

import numpy as np

from rice_Ml.supervised_ml.DecisionTree.decision_tree import DecisionTree


class AdaBoost:
    """
    Adaptive Boosting classifier (SAMME variant for multiclass support).

    Sequentially fits decision stumps, each one focusing on the mistakes of
    the previous. Final prediction is a weighted majority vote across all stumps.

    Attributes set after fit():
        estimators_ — list of fitted DecisionTree stumps
        alphas_     — weight of each stump's vote
        classes_    — sorted array of unique class labels
    """

    def __init__(self, n_estimators=50, learning_rate=1.0, stump_depth=1, random_state=None):
        """
        n_estimators  — number of boosting rounds
        learning_rate — shrinks each stump's contribution; trades off with n_estimators
        stump_depth   — max_depth of each weak learner (1 = classic stump, 2–3 for more expressiveness)
        random_state  — seed for reproducibility
        """
        if not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError(f"n_estimators must be a positive integer, got {n_estimators!r}.")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate!r}.")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.stump_depth = stump_depth
        self.random_state = random_state

    def fit(self, X, y):
        """Fit boosted stumps on (X, y). Returns self."""
        X = np.array(X)
        y = np.array(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length, got {len(X)} and {len(y)}.")

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples = len(X)

        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31, size=self.n_estimators)

        weights = np.ones(n_samples) / n_samples
        self.estimators_: list[DecisionTree] = []
        self.alphas_: list[float] = []

        for seed in seeds:
            # weighted bootstrap: sample proportional to current weights
            indices = rng.choice(n_samples, size=n_samples, replace=True, p=weights)
            stump = DecisionTree(max_depth=self.stump_depth, random_state=int(seed))
            stump.fit(X[indices], y[indices])

            pred = stump.predict(X)
            incorrect = (pred != y).astype(float)
            err = float(np.dot(weights, incorrect))
            err = np.clip(err, 1e-10, 1 - 1e-10)

            # SAMME: adds log(K-1) term so alpha stays positive for multiclass
            alpha = self.learning_rate * (np.log((1 - err) / err) + np.log(max(n_classes - 1, 1)))

            weights *= np.exp(alpha * incorrect)
            weights /= weights.sum()

            self.estimators_.append(stump)
            self.alphas_.append(alpha)

        return self

    def predict(self, X):
        """Return the class with the highest weighted vote for each row of X."""
        scores = self._decision_scores(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """Return softmax-normalised decision scores as class probabilities."""
        scores = self._decision_scores(X)
        # shift for numerical stability before softmax
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def score(self, X, y):
        """Return accuracy on (X, y)."""
        return np.mean(self.predict(X) == np.array(y))

    def _decision_scores(self, X):
        """Accumulate alpha-weighted votes into a (n_samples, n_classes) score matrix."""
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        scores = np.zeros((len(X), len(self.classes_)))
        for alpha, stump in zip(self.alphas_, self.estimators_):
            for i, pred in enumerate(stump.predict(X)):
                if pred in class_to_idx:
                    scores[i, class_to_idx[pred]] += alpha
        return scores
