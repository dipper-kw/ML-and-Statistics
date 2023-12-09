"""
This module implements the k-Nearest Neighbors algorithm.
It includes a class KNN for building and using a kNN model for classification.
"""

from collections import Counter
import numpy as np


def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2) ** 2))


class KNN:
    """
    k-Nearest Neighbors classifier.
    
    Attributes:
        k (int): Number of neighbors to use for classification.
        x_train (array-like): Training data.
        y_train (array-like): Training labels.
    """
    def __init__(self, k=3):
        """Initialize the KNN object with k neighbors."""
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        """
        Fit the model using x as training data and y as target values.

        Args:
            x (array-like): Training data, shape (n_samples, n_features)
            y (array-like): Target values, shape (n_samples,)
        """
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        """
        Predict the class labels for the provided data.

        Args:
            x (array-like): Test samples, shape (n_samples, n_features)

        Returns:
            array, shape (n_samples,): Class labels for each data sample.
        """
        predicted_labels = [self._predict(sample) for sample in x]
        return np.array(predicted_labels)

    def _predict(self, sample):
        """Private method to predict a single point."""
        # Compute distances
        distances = [euclidean_distance(sample, train_sample) for train_sample in self.x_train]

        # Get the k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)

        return most_common[0][0]
