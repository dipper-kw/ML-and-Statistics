import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans  # Assuming KMeans class is in 'kmeans.py'

def create_synthetic_data():
    """
    Create synthetic dataset for testing KMeans.
    
    Returns:
    ndarray: Generated synthetic dataset.
    """
    # Generate synthetic data for 3 clusters
    np.random.seed(42)
    cluster_1 = np.random.randn(100, 2) + np.array([5, 5])
    cluster_2 = np.random.randn(100, 2) + np.array([10, 10])
    cluster_3 = np.random.randn(100, 2) + np.array([20, 5])

    return np.vstack((cluster_1, cluster_2, cluster_3))

def test_kmeans():
    """
    Test the KMeans algorithm on synthetic dataset.
    """
    X = create_synthetic_data()

    # KMeans clustering
    k = 3
    clf = KMeans(k=k, max_iters=150, plot_steps=False)
    y_pred = clf.predict(X)

    # Plotting the results
    plt.figure(figsize=(12, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, edgecolor='k', s=50, cmap='viridis')

    # Plotting the centroids
    centroids = np.array(clf.centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', edgecolor='black')

    plt.title(f"KMeans Clustering with k={k}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

if __name__ == "__main__":
    test_kmeans()
