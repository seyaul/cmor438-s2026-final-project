"""
decision_tree.py

a binary decision tree classifier that recursively splits the feature space
by finding the best threshold on a single feature at each node.

labels can be any discrete class values. for regression, extend predict()
to average leaf values instead of taking the majority vote.
"""

import numpy as np

# Training:
# 1. At current node, try every possible split on every feature
# 2. Pick the split that creates the purest groups (Gini Impurity)
# 3. Split the data into two branches
# 4. Repeat steps 1-3 on each branch recursively
# 5. Stop when groups are pure enough OR max depth reached

# Predicting for a new point:
# 6. Follow the yes/no questions down the tree
# 7. Output the label at the final node

class Node:
    """represents a single node in the tree — either a split or a leaf."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # internal node fields

        self.feature = feature       # index of feature to split on (e.g. feature 0 = septal length) 
        self.threshold = threshold   # value to split on (e.g <=2.5 goes left, >2.5 goes right)

        self.left = left             # left child Node (feature <= threshold)
        self.right = right           # right child Node (feature > threshold)

        # leaf node field — majority class label
        self.value = value

    #checks if a node is a leaf. If its value is None, then it is an internal node with a split decision. 
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

         #checks come before assignment
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 1):
            raise ValueError(f"max depth must be a positive integer, got {max_depth!r}.")

        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError(f"minimum samples to split on must be a integer above 2, got {min_samples_split!r}.")
        
        if criterion not in ("gini", "entropy"):
            raise ValueError(f'criterion must be "gini" or "entropy", got {criterion!r}.')

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
       

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
        
        X = np.array(X)
        y = np.array(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length, got {len(X)} and {len(y)}.")
        
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]

        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        """
        classify each row of X and return a 1-D label array.
        """
        # TODO: call self._traverse(row, self.root_) for each row in X
        
        return np.array([self._traverse(row, self.root_) for row in X])
        
        # or!
        # results = []
        # for row in X:
        #     results.append(self._traverse(row, self.root_))
        # return np.array(results)

    def score(self, X, y):
        """
        return fraction of correctly classified samples (accuracy).
        """
        return np.mean(self.predict(X) == np.array(y))

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
        
        #base case: 

        if self._impurity(y) == 0 or len(y) < self.min_samples_split :
            return Node(value=self._majority_class(y))
        
        elif self.max_depth != None and self.max_depth <= depth:
            return Node(value=self._majority_class(y))
        
        #Recurisve Case:

        # Find the best split using _best_split(X, y)
        # If no valid split exists, return a leaf
        # Split X and y into left and right subsets
        # Recursively build left and right subtrees
        # Return an internal Node with the split info

        #the best feature and its threshold
        feature, threshold = self._best_split(X, y)

        #if no valid split exists, return a leaf
        if feature == None and threshold == None:
            return Node(value=self._majority_class(y))
        
        #boolean masks for left and right subsets. all rows, only column feature 
        #X[:, feature] = give me every sample's value for this particular feature
        left_mask = X[:, feature] <= threshol
        right_mask = X[:, feature] > threshold

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        #recurisvely splits nodes until we hit a leaf 
        left = self._build_tree(X_left, y_left, depth+1)
        right = self._build_tree(X_right, y_right, depth+1)
        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def _best_split(self, X, y):
        """
        search every feature and threshold for the split that most reduces
        impurity. return (feature_index, threshold) or (None, None) if no
        valid split exists.
        """
        best = [None, None, float('-inf')]
        parent_impurity = self._impurity(y)
        n = len(y)

        for feature in range(X.shape[1]):
            for thresh in np.unique(X[:, feature]):
                left_mask = X[:, feature] <= thresh
                right_mask = ~left_mask
                n_left, n_right = left_mask.sum(), right_mask.sum()
                if n_left == 0 or n_right == 0:
                    continue

                gain = parent_impurity - (
                    n_left / n * self._impurity(y[left_mask]) +
                    n_right / n * self._impurity(y[right_mask])
                )
                if gain > best[2]:
                    best = [feature, thresh, gain]

        return best[0], best[1]



    def _impurity(self, y):
        """
        compute impurity of label array y using self.criterion.
        returns a float (lower is purer).
        """
        # TODO: implement gini = 1 - sum(p_k^2) and
        #       entropy = -sum(p_k * log2(p_k))

        labels, counts = np.unique(y, return_counts=True)
        
        if self.criterion == "gini":
            p = counts / np.sum(counts)
            gini = 1- np.sum(p**2)
            return gini

        else:
            p = counts/np.sum(counts)
            p = p[p > 0] # remove zero probabilities before log
            entropy = -1 * np.sum(p*np.log2(p))
            return entropy
        

    def _majority_class(self, y):
        """return the most common label in y."""
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]
    
    def _traverse(self, x, node):
        """
        walk a single sample x down the tree from node, return leaf value.
        """
        if node.is_leaf() == True:
            return node.value
        else:
            if x[node.feature] <= node.threshold:
                return self._traverse(x, node.left)
                
            else:
                return self._traverse(x, node.right)
