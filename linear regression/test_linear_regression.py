import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

# Creating synthetic data for testing
np.random.seed(0)
X_test = 2 - 3 * np.random.normal(0, 1, (20, 1))
y_test = X_test - 2 * (X_test ** 2) + np.random.normal(-3, 3, (20, 1))

def test_linear_regression_fit_predict():
    """
    Test the fit and predict methods of the LinearRegression class.
    """
    model = LinearRegression(learning_rate=0.01, n_iters=1000)
    model.fit(X_test, y_test.ravel())
    
    # Check if weights and bias are not None (model is trained)
    assert model.weights is not None
    assert model.bias is not None
    
    # Predict and check if output is not None
    predictions = model.predict(X_test)
    assert predictions is not None

def plot_linear_regression_results():
    """
    Plot the results of the linear regression model.
    """
    model = LinearRegression(learning_rate=0.01, n_iters=1000)
    model.fit(X_test, y_test.ravel())
    predictions = model.predict(X_test)

    # Plot
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, predictions, color='red', label='Predicted')
    plt.title("Linear Regression Test")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_linear_regression_results()
