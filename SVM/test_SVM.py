import numpy as np
import matplotlib.pyplot as plt
from SVM import SVM

def generate_synthetic_data():
    """
    Generate synthetic data for binary classification.
    
    Returns:
        features (np.ndarray): Generated features.
        labels (np.ndarray): Generated labels.
    """
    np.random.seed(0)
    features_class_1 = np.random.normal(0, 1, (50, 2))
    features_class_2 = np.random.normal(2, 1, (50, 2))
    features = np.vstack((features_class_1, features_class_2))
    labels = np.array([-1]*50 + [1]*50)
    return features, labels

def test_svm_fit_predict():
    """
    Test the fit and predict methods of the SVM class.
    """
    features, labels = generate_synthetic_data()
    model = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
    model.fit(features, labels)

    predictions = model.predict(features)
    assert len(predictions) == len(labels)
    assert all(pred in [-1, 1] for pred in predictions)

def plot_svm_results():
    """
    Plot the results of the SVM model.
    """
    features, labels = generate_synthetic_data()
    model = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
    model.fit(features, labels)

    # Create a mesh to plot in
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Plot decision boundary
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(features[:, 0], features[:, 1], c=labels, edgecolor='k')
    plt.title("SVM Classifier Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

if __name__ == "__main__":
    test_svm_fit_predict()
    plot_svm_results()
