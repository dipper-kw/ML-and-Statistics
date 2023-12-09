import numpy as np

"""
A simple implementation of Logistic Regression using gradient descent.
This module provides a Logistic Regression classifier.
"""

class LogisticRegression:
    """
    Logistic Regression model using gradient descent for optimization.

    Attributes:
        learning_rate (float): Learning rate for gradient descent.
        n_iters (int): Number of iterations for the optimization algorithm.
        weights (np.ndarray): Weights vector for logistic regression.
        bias (float): Bias term.
    """

    def __init__(self, learning_rate=0.001, n_iters=1000):
        """
        Initialize the Logistic Regression model.

        Args:
            learning_rate (float): Learning rate.
            n_iters (int): Number of iterations.
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, features, labels):
        """
        Fit the logistic regression model to the training data.

        Args:
            features (np.ndarray): Training data, shape (n_samples, n_features).
            labels (np.ndarray): Training labels, shape (n_samples,).
        """
        n_samples, n_features = features.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(features, self.weights) + self.bias
            predictions = self._sigmoid(linear_model)

            d_weights = (1 / n_samples) * np.dot(features.T, (predictions - labels))
            d_bias = (1 / n_samples) * np.sum(predictions - labels)

            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias

    def predict(self, features):
        """
        Predict class labels for samples in features.

        Args:
            features (np.ndarray): Data to predict, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,).
        """
        linear_model = np.dot(features, self.weights) + self.bias
        predictions = self._sigmoid(linear_model)
        prediction_classes = [1 if i > 0.5 else 0 for i in predictions]
        return np.array(prediction_classes)

    def _sigmoid(self, z):
        """
        Apply the sigmoid function.

        Args:
            z (np.ndarray): Input array.

        Returns:
            np.ndarray: Output of sigmoid function.
        """
        return 1 / (1 + np.exp(-z))
