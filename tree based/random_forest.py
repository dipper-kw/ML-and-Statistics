import numpy as np
from collections import Counter
from decision_tree import DecisionTree

def bootstrap_sample(X, y):
    """
    Create a bootstrap sample of the dataset.
    
    Args:
        X (np.ndarray): The input feature matrix.
        y (np.array): The target values.
    
    Returns:
        (np.ndarray, np.array): A bootstrap sample of the dataset.
    """
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]

def most_common_label(y):
    """
    Identify the most common label in the target array.
    
    Args:
        y (np.array): The target values.
    
    Returns:
        The most common label.
    """
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common

class RandomForest:
    """
    Random Forest classifier.

    Attributes:
        n_trees (int): The number of trees in the forest.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        max_depth (int): The maximum depth of the trees.
        n_feats (int): The number of features to consider when looking for the best split.
        trees (list): The list of trees in the forest.
    """

    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        """
        Initialize the Random Forest.

        Args:
            n_trees (int): The number of trees in the forest.
            min_samples_split (int): The minimum number of samples required to split an internal node.
            max_depth (int): The maximum depth of the trees.
            n_feats (int): The number of features to consider when looking for the best split.
        """
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        """
        Fit the Random Forest to the training data.

        Args:
            X (np.ndarray): The input feature matrix.
            y (np.array): The target values.
        """
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=self.n_feats
            )
            X_samp, y_samp = bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict the class for each sample in X.

        Args:
            X (np.ndarray): The input feature matrix.
        
        Returns:
            np.array: The predicted classes.
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
