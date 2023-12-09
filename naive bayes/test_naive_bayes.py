import numpy as np
import matplotlib.pyplot as plt
from naive_bayes import NaiveBayes

"""
Test module for the Naive Bayes classifier. Includes functions to generate synthetic data,
test the classifier's fit and predict methods, and visualize the decision boundary.
"""

def generate_synthetic_data():
    """
    Generate synthetic data for binary classification.
    
    Returns:
        features (np.ndarray): Generated features.
        labels (np.ndarray): Generated labels.
    """
    np.random.seed(0)
    features_class_1 = np.random.normal(0, 1, (100, 2))
    features_class_2 = np.random.normal(2, 1, (100, 2))
    features = np.vstack((features_class_1, features_class_2))
    labels = np.array([0]*100 + [1]*100)
    return features, labels

def test_naive_bayes_fit_predict():
    """
    Test the fit and predict methods of the NaiveBayes class.
    """
    features, labels = generate_synthetic_data()
    model = NaiveBayes()
    model.fit(features, labels)

    predictions = model.predict(features)
    assert len(predictions) == len(labels)

def plot_naive_bayes_results():
    """
    Plot the results of the Naive Bayes model.
    """
    features, labels = generate_synthetic_data()
    model = NaiveBayes()
    model.fit(features, labels)

    # Create a mesh to plot in
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Plot decision boundary
    grid_predictions = model.predict(np.c_[grid_x.ravel(), grid_y.ravel()])
    grid_predictions = np.array(grid_predictions).reshape(grid_x.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(grid_x, grid_y, grid_predictions, alpha=0.8)
    plt.scatter(features[:, 0], features[:, 1], c=labels, edgecolor='k')
    plt.title("Naive Bayes Classifier Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

if __name__ == "__main__":
    test_naive_bayes_fit_predict()
    plot_naive_bayes_results()
