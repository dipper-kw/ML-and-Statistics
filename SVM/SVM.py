import numpy as np

"""
This module implements the Support Vector Machine (SVM) algorithm, 
a supervised machine learning model used for classification tasks.
"""

class SVM:
    """
    Support Vector Machine classifier.

    Attributes:
        learning_rate (float): The learning rate for the optimization.
        lambda_param (float): The regularization parameter.
        n_iters (int): The number of iterations over the training data.
        weights (np.ndarray): The weights of the SVM model.
        bias (float): The bias term.
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Initialize the SVM model.

        Args:
            learning_rate (float): The learning rate.
            lambda_param (float): The regularization parameter.
            n_iters (int): The number of iterations.
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, features, labels):
        """
        Fit the SVM model to the training data.

        Args:
            features (np.ndarray): Training data, shape (n_samples, n_features).
            labels (np.ndarray): Training labels, shape (n_samples,).
        """
        _, n_features = features.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        labels_ = np.where(labels <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, feature in enumerate(features):
                condition = labels_[idx] * (np.dot(feature, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(feature, labels_[idx]))
                    self.bias -= self.learning_rate * labels_[idx]

    def predict(self, features):
        """
        Predict class labels for samples in features.

        Args:
            features (np.ndarray): Data to predict, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,).
        """
        approx = np.dot(features, self.weights) - self.bias
        return np.sign(approx)
