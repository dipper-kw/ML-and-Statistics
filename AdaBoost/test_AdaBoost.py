import numpy as np
import matplotlib.pyplot as plt
from adaboost import AdaBoost  # Assuming AdaBoost is in 'adaboost.py'

def create_synthetic_data():
    """
    Create a simple synthetic dataset for testing.

    Returns:
    X (ndarray): Generated synthetic feature data.
    y (ndarray): Generated synthetic target labels.
    """
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.array([1 if x[0] - x[1] > 0 else -1 for x in X])

    return X, y

def test_adaboost():
    """
    Test the AdaBoost classifier on a synthetic dataset.
    """
    X, y = create_synthetic_data()

    # Initialize AdaBoost classifier with 5 weak learners
    clf = AdaBoost(n_clf=5)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y)
    print(f"Accuracy: {accuracy:.2f}")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class +1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')
    plt.legend()
    plt.title('AdaBoost Classifier on Synthetic Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

if __name__ == "__main__":
    test_adaboost()
