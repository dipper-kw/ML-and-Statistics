"""
This module contains tests for the KNN classifier implemented in the knn module.
It includes tests for the Euclidean distance function, the fitting and prediction of the KNN model,
and a function to plot the results for visual inspection.
"""

import numpy as np
import matplotlib.pyplot as plt
from knn import KNN, euclidean_distance

# Test Data
np.random.seed(0)
X_test = np.random.rand(10, 2)
y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

def test_euclidean_distance():
    """Test the euclidean_distance function."""
    assert euclidean_distance(np.array([0, 0]), np.array([3, 4])) == 5

def test_knn_fit():
    """Test the fit method of the KNN class."""
    knn = KNN(k=3)
    knn.fit(X_test, y_test)
    assert knn.x_train is not None
    assert knn.y_train is not None

def test_knn_predict():
    """Test the predict method of the KNN class."""
    knn = KNN(k=3)
    knn.fit(X_test, y_test)
    predictions = knn.predict(X_test)
    assert len(predictions) == len(y_test)
    assert all(isinstance(label, np.int64) for label in predictions)

def plot_knn_results():
    """Plot the KNN results with decision boundaries."""
    knn = KNN(k=3)
    knn.fit(X_test, y_test)

    # Create mesh grid for plotting decision boundary
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    mesh_x, mesh_y = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                 np.arange(y_min, y_max, 0.1))

    # Predict class for each point in the mesh
    class_predictions = knn.predict(np.c_[mesh_x.ravel(), mesh_y.ravel()])
    class_predictions = class_predictions.reshape(mesh_x.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(mesh_x, mesh_y, class_predictions, alpha=0.4)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, edgecolor='k')
    plt.title("KNN Classifier Results")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

if __name__ == "__main__":
    plot_knn_results()
