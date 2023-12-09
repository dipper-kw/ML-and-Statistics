import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def euclidean_distance(x1, x2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    x1 (ndarray): First point.
    x2 (ndarray): Second point.

    Returns:
    float: Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KMeans:
    """
    KMeans clustering algorithm.

    Attributes:
    k (int): Number of clusters.
    max_iters (int): Maximum number of iterations.
    plot_steps (bool): Whether to plot steps during training.
    clusters (list): List of clusters, each cluster contains indices of data points.
    centroids (list): List of centroids, one for each cluster.
    """

    def __init__(self, k=5, max_iters=100, plot_steps=False):
        """
        Initialize the KMeans instance.

        Parameters:
        k (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        plot_steps (bool): Whether to plot steps during training.
        """
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []

    def predict(self, X):
        """
        Predict the cluster for each sample in X.

        Parameters:
        X (ndarray): Data points to cluster.

        Returns:
        ndarray: Cluster labels for each data point in X.
        """
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialize centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # Get labels for each point based on cluster
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    def _create_clusters(self, centroids):
        # Assign points to the nearest centroid
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # Find closest centroid for a given point
        distances = [euclidean_distance(sample, point) for point in centroids]
        return np.argmin(distances)

    def _get_centroids(self, clusters):
        # Calculate centroids as mean of assigned points
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # Check if centroids have converged
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.k)]
        return sum(distances) == 0

    def plot(self):
        # Plot the clusters and centroids
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        plt.show()
