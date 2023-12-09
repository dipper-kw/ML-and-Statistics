import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

"""
Test module for Logistic Regression model. Includes functions to generate synthetic data,
test the model's fit and predict methods, and plot the decision boundary.
"""

def generate_synthetic_data():
    """
    Generate synthetic data for binary classification.
    
    Returns:
        features (np.ndarray): Generated features.
        labels (np.ndarray): Generated labels.
    """
    np.random.seed(0)
    features_class_1 = np.random.normal(2, 1, (100, 2))
    features_class_2 = np.random.normal(-2, 1, (100, 2))
    features = np.vstack((features_class_1, features_class_2))
    labels = np.array([1]*100 + [0]*100)
    return features, labels

def test_logistic_regression():
    """
    Test the Logistic Regression model.
    """
    features, labels = generate_synthetic_data()
    model = LogisticRegression(learning_rate=0.01, n_iters=1000)
    model.fit(features, labels)

    assert model.weights is not None
    assert model.bias is not None

    predictions = model.predict(features)
    assert len(predictions) == len(labels)
    assert all(pred in [0, 1] for pred in predictions)

def plot_decision_boundary():
    """
    Plot the decision boundary of the Logistic Regression model.
    """
    features, labels = generate_synthetic_data()
    model = LogisticRegression(learning_rate=0.01, n_iters=1000)
    model.fit(features, labels.ravel())

    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_predictions = model.predict(np.c_[grid_x.ravel(), grid_y.ravel()])
    grid_predictions = grid_predictions.reshape(grid_x.shape)

    plt.contourf(grid_x, grid_y, grid_predictions, alpha=0.8)
    plt.scatter(features[:, 0], features[:, 1], c=labels, edgecolor='k')
    plt.title('Logistic Regression Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

if __name__ == "__main__":
    test_logistic_regression()
    plot_decision_boundary()
