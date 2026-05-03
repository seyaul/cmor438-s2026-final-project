"""Stacking ensemble: combines base estimator predictions via a meta-learner."""

import numpy as np
from rice_ML.model_selection.split import KFold


class StackingClassifier:
    """
    Stacking ensemble that trains a meta-learner on out-of-fold predictions.

    Base estimators are trained on k-fold splits; their predictions on the
    held-out fold form the meta-features used to train the meta-learner.
    All base estimators are then re-fit on the full training set so they can
    predict on new data.

    Attributes set after fit():
        estimators_     — list of base estimators re-fit on full training data
        meta_estimator_ — meta-learner fit on out-of-fold meta-features
        classes_        — sorted array of unique class labels
    """

    def __init__(self, estimators, meta_estimator, cv=5):
        """
        estimators      — list of (name, estimator) tuples; each must implement fit/predict
        meta_estimator  — model trained on base estimator outputs; must implement fit/predict
        cv              — number of cross-validation folds for generating meta-features
        """
        if not estimators:
            raise ValueError("estimators must be a non-empty list.")
        if not isinstance(cv, int) or cv < 2:
            raise ValueError(f"cv must be an integer >= 2, got {cv!r}.")

        self.estimators = estimators
        self.meta_estimator = meta_estimator
        self.cv = cv

    def fit(self, X, y):
        """Generate out-of-fold meta-features, fit meta-learner, re-fit base estimators. Returns self."""
        X = np.array(X)
        y = np.array(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length, got {len(X)} and {len(y)}.")

        self.classes_ = np.unique(y)
        n_samples = len(X)
        n_base = len(self.estimators)

        # --- build out-of-fold meta-features ---
        meta_X = np.zeros((n_samples, n_base))
        kfold = KFold(n_splits=self.cv)

        for fold_train_idx, fold_val_idx in kfold.split(X):
            for col, (_, est) in enumerate(self.estimators):
                est.fit(X[fold_train_idx], y[fold_train_idx])
                meta_X[fold_val_idx, col] = est.predict(X[fold_val_idx])

        # --- fit meta-learner on OOF predictions ---
        self.meta_estimator_ = self.meta_estimator
        self.meta_estimator_.fit(meta_X, y)

        # --- re-fit all base estimators on the full training set ---
        self.estimators_ = []
        for name, est in self.estimators:
            est.fit(X, y)
            self.estimators_.append((name, est))

        return self

    def predict(self, X):
        """Return meta-learner prediction for each row of X."""
        return self.meta_estimator_.predict(self._meta_features(X))

    def score(self, X, y):
        """Return accuracy on (X, y)."""
        return np.mean(self.predict(X) == np.array(y))

    def _meta_features(self, X):
        """Collect base estimator predictions into a (n_samples, n_base) matrix."""
        return np.column_stack([est.predict(X) for _, est in self.estimators_])

