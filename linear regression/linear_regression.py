import numpy as np

class LinearRegression:
    """
    Simple linear regression model using gradient descent.

    Attributes:
        learning_rate (float): The learning rate for gradient descent.
        n_iters (int): The number of iterations for the gradient descent.
        weights (np.ndarray): The weights of the linear model.
        bias (float): The bias of the linear model.
    """

    def __init__(self, learning_rate, n_iters) -> None:
        """
        Initialize the LinearRegression model.

        Args:
            learning_rate (float): The learning rate.
            n_iters (int): The number of iterations.
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Target values, shape (n_samples,).
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            d_weights = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            d_bias = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias

    def predict(self, X):
        """
        Predict using the linear model.

        Args:
            X (np.ndarray): Data, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values.
        """
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

# Ensure a final newline at the end of the file
