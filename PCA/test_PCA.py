# test_PCA.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from PCA import PCA  # Assuming your PCA class is in PCA.py

def test_pca():
    # Load dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Initialize and fit PCA
    pca = PCA(2)  # Reduce to 2 dimensions
    pca.fit(X)

    # Transform the data
    X_projected = pca.transform(X)

    # Plot the transformed data
    plt.scatter(X_projected[:, 0], X_projected[:, 1],
                c=y, edgecolor='none', alpha=0.8,
                cmap=plt.cm.get_cmap('viridis', 3))

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.title('PCA of Iris Dataset')
    plt.show()

if __name__ == "__main__":
    test_pca()
