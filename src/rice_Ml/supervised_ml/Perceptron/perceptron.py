"""
perceptron.py

a single-layer binary classifier that learns a linear decision boundary.
it works by stepping through the training data, updating weights whenever
it misclassifies a sample, and stopping once it gets everything right.

labels must be 1 or -1. for non-linearly-separable data it will run until
the epoch limit and converged_ will be False.
"""

import numpy as np

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

    #eta is learning rate / our neuron cost function derivative 1/2 * 2
    #random_state pins the weight initialization so results are reproducible
    def __init__(self, eta=0.1, epochs=50, random_state=None):
        """
        set up the perceptron with a learning rate, epoch limit, and optional seed.

        eta          — step size for each weight update, must be > 0
        epochs       — maximum number of full passes over the training data
        random_state — seed for numpy's random number generator; pass an int
                       to get the same starting weights every run
        """
        #checks come before assignment
        if not isinstance(eta, float | int) or eta <= 0:
            raise ValueError(f"eta must be a positive integer or float, got {eta!r}.")

        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"epochs must be a integer above 0, got {epochs!r}.")

        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state


    def train(self, X, y):
        """
        fit the perceptron to training data X and labels y.

        X must be a 2-d array of shape (n_samples, n_features).
        y must be a 1-d array of 1s and -1s with length n_samples.
        returns self so you can chain: Perceptron().train(X, y).predict(X)
        """
        X = np.array(X)
        y = np.array(y)

        #make sure inputs are the right shape and labels are binary before doing anything
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length, got {len(X)} and {len(y)}.")
        if not np.all(np.isin(y, [1, -1])):
            raise ValueError("y must contain only 1 and -1.")
 

        #initialize w1, w2, and b — one weight per feature plus the bias term
        np.random.seed(self.random_state)
        self.w_b_ = np.random.rand(1 + X.shape[1])

        epoch_counter = 0
        self.mistakes_ = []

        #loop over each sample, update weights whenever we get one wrong
        while epoch_counter < self.epochs:
            errors = 0
            for xi, yi in zip(X, y):

                #our alpha adjustment is the value we multiply by
                # bias = alpha * xi
                # weight = alpha
                prediction = self.predict(xi)
                if prediction - yi != 0:
                    adjustment = self.eta * (prediction - yi)
                    self.w_b_[:-1] -= adjustment * xi
                    self.w_b_[-1] -= adjustment
                    #counts each misclassification this epoch
                    errors += int(adjustment != 0)

            #if no misclassifications this epoch the boundary is perfect — stop early
            if errors == 0:
                self.converged_ = True
                #epoch_counter is 0-indexed so add 1 for the real count
                self.n_epochs_run_ = epoch_counter + 1
                return self
            else:
                #record how many mistakes we made this epoch to track learning progress
                self.mistakes_.append(errors)
            epoch_counter += 1

        #hit the epoch limit without converging
        self.converged_ = False
        self.n_epochs_run_ = self.epochs
        return self


    #dot product of inputs and weights, plus the bias — raw score before thresholding
    def net_input(self, X):
        """
        compute the weighted sum of inputs plus bias.
        returns a scalar for a single sample or an array for a batch.
        """
        return np.dot(X, self.w_b_[:-1]) + self.w_b_[-1]


    #if net input >= 0 predict class 1, otherwise -1
    def predict(self, X):
        """
        classify X as 1 or -1 based on the learned weights.
        X can be a single sample (1-d) or a batch (2-d).
        """
        return np.where(self.net_input(X) >= 0, 1, -1)


    #fraction of correctly classified samples — 1.0 = perfect
    def score(self, X, y):
        """
        return the accuracy of the model on X against true labels y.
        1.0 means every sample was classified correctly.
        """
        return np.mean(self.predict(X) == np.array(y))
