# Principal Component Analysis (PCA)

## Basics

* Feature vectors: $$\mathbf{x}_i \in \mathbb{R}^d$$, Reduced dimensionality: $$k$$ (where $$k < d$$)
* PCA aims to reduce the dimensionality of the data while retaining as much variance as possible.

## Algorithm

### 1. Model Representation

Principal Component Analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of possibly correlated variables into a set of linearly uncorrelated variables called principal components.

### 2. Standardize the Data

Standardize the feature vectors to have a mean of zero and a standard deviation of one: $$\mathbf{x}_{i,\text{standardized}} = \frac{\mathbf{x}_i - \mu}{\sigma}$$.

### 3. Compute the Covariance Matrix

Calculate the covariance matrix $$\Sigma$$ to understand how each variable relates to the other.

### 4. Calculate Eigenvalues and Eigenvectors

Compute the eigenvalues ($$\lambda$$) and eigenvectors ($$\mathbf{v}$$) of the covariance matrix. Eigenvectors determine the directions of the new feature space, and eigenvalues determine their magnitude.

### 5. Sort Eigenvectors

Sort the eigenvectors by decreasing eigenvalues and choose the first $$k$$ eigenvectors to form a $$d \times k$$ dimensional matrix $$W$$.

### 6. Transform the Original Dataset

Transform the original $$d$$ dimensional dataset into a new $$k$$ dimensional subspace: $$\mathbf{X}_{\text{new}} = \mathbf{X}W$$.

### 7. Interpretation of Components

Each principal component represents a direction in the original data that maximizes the variance along that axis.

### 8. Advantages and Disadvantages

- **Advantages**: Reduces computational cost, removes correlated features, and improves algorithm performance.
- **Disadvantages**: Can be hard to interpret, and dimensionality reduction may lead to some data loss.

PCA is widely used in exploratory data analysis, pattern recognition, and in making predictive models. It's particularly useful in dealing with multicollinearity and in visualizing high-dimensional data.
