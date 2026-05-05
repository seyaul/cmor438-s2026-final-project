"""
perceptron.py

a single-layer binary classifier that learns a linear decision boundary.
it works by stepping through the training data, updating weights whenever
it misclassifies a sample, and stopping once it gets everything right.

labels must be 1 or -1. for non-linearly-separable data it will run until
the epoch limit and converged_ will be False.
"""

import numpy as np
import matplotlib.pyplot as plt


class Perceptron(object):
    """
    single-layer perceptron for binary classification.

    trains on labeled data (y = 1 or -1) by adjusting a weight vector
    and bias whenever a sample is misclassified. stops early once the
    training set is perfectly classified, or after 'epochs' passes.

    attributes set after train():
        w_b_         — weight vector plus bias, shape (n_features + 1,)
        mistakes_    — list of misclassification counts per epoch
        converged_   — True if training stopped early, False if it hit the limit
        n_epochs_run_ — how many epochs actually ran
    """

    def __init__(self, eta=0.1, epochs=50, random_state=None):
        """Learning rate eta, epoch limit, and optional random_state seed."""
        # random_state pins the weight initialization so results are reproducible
        if not isinstance(eta, float | int) or eta <= 0:
            raise ValueError(f"eta must be a positive integer or float, got {eta!r}.")

        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"epochs must be a integer above 0, got {epochs!r}.")

        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state


    def train(self, X, y):
        """Fit the perceptron to (X, y); returns self."""
        X = np.array(X)
        y = np.array(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length, got {len(X)} and {len(y)}.")
        if not np.all(np.isin(y, [1, -1])):
            raise ValueError("y must contain only 1 and -1.")

        np.random.seed(self.random_state)
        self.w_b_ = np.random.rand(1 + X.shape[1])

        epoch_counter = 0
        self.mistakes_ = []
        self.loss_history_ = []

        while epoch_counter < self.epochs:
            errors = 0
            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                if prediction - yi != 0:
                    adjustment = self.eta * (prediction - yi)
                    self.w_b_[:-1] -= adjustment * xi
                    self.w_b_[-1] -= adjustment
                    errors += int(adjustment != 0)

            if errors == 0:
                self.converged_ = True
                self.n_epochs_run_ = epoch_counter + 1  # epoch_counter is 0-indexed
                return self
            else:
                self.mistakes_.append(errors)
                # perceptron criterion loss: sum of -y_i * net_input(x_i) for misclassified samples
                net = self.net_input(X)
                misclassified = np.where(net >= 0, 1, -1) != y
                loss = float(np.sum(-y[misclassified] * net[misclassified]))
                self.loss_history_.append(loss)
            epoch_counter += 1

        self.converged_ = False
        self.n_epochs_run_ = self.epochs
        return self

    def net_input(self, X):
        """
        compute the weighted sum of inputs plus bias.
        returns a scalar for a single sample or an array for a batch.
        """
        return np.dot(X, self.w_b_[:-1]) + self.w_b_[-1]


    def predict(self, X):
        """Classify X as 1 or -1; X can be a single sample or a batch."""
        return np.where(self.net_input(X) >= 0, 1, -1)

    def score(self, X, y):
        """Return accuracy on (X, y)."""
        return np.mean(self.predict(X) == np.array(y))
