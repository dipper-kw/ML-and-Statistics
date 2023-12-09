# K-Means Clustering

## Basics

* Data Points: $$\mathbf{x}_i \in \mathbb{R}^d$$, Number of Clusters: $$k$$
* K-Means aims to partition the data into $$k$$ clusters, each represented by the mean of its points.

## Algorithm

### 1. Model Representation

K-Means Clustering is an unsupervised learning algorithm that groups data into $$k$$ clusters. Each cluster is represented by its centroid, the mean of the points in the cluster.

### 2. Initialize Centroids

Randomly select $$k$$ data points as the initial centroids or cluster centers: $$\mathbf{c}_1, \mathbf{c}_2, \ldots, \mathbf{c}_k$$.

### 3. Assign Points to Nearest Centroid

Each data point $$\mathbf{x}_i$$ is assigned to the cluster of the nearest centroid, determined by the Euclidean distance: $$\min_{j} \|\mathbf{x}_i - \mathbf{c}_j\|^2$$.

### 4. Update Centroids

Update each centroid to be the mean of the data points that are assigned to its cluster: $$\mathbf{c}_j = \frac{1}{|S_j|} \sum_{\mathbf{x}_i \in S_j} \mathbf{x}_i$$ where $$S_j$$ is the set of data points assigned to the $$j$$th cluster.

### 5. Repeat Assignment and Update Steps

Repeat steps 3 and 4 until the centroids do not change significantly or a maximum number of iterations is reached.

### 6. Final Clusters

The algorithm results in $$k$$ clusters with data points assigned to the cluster of their nearest centroid.

### 7. Advantages and Disadvantages

- **Advantages**: Simple to understand and implement, efficient in terms of computational cost.
- **Disadvantages**: Assumes spherical clusters and is sensitive to the initial choice of centroids and outliers.

K-Means is widely used in market segmentation, document clustering, image segmentation, and pattern recognition. Its ease of use and quick execution make it a popular choice for many clustering applications.
