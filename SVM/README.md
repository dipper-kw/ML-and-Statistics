# Support Vector Machine (SVM)

## Basics

* Feature vectors: $${\mathbf{x}_i} \in \mathbb{R}^d$$, Target values (binary): $${y_i} \in \{-1, +1\}$$
* The goal of SVM is to find the optimal separating hyperplane which maximizes the margin between two classes.

## Algorithm

### 1. Model Representation

SVM is a supervised learning algorithm used for binary classification. It aims to find the best boundary (hyperplane) that separates data points of different classes.

- The decision function is: $$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$$.

### 2. Maximize the Margin

The SVM algorithm seeks to maximize the margin between the data points and the hyperplane. The margin is defined as the distance between the hyperplane and the nearest data points from each class, known as support vectors.

### 3. Solving the Optimization Problem

To find the optimal hyperplane, SVM solves the following optimization problem:

$$
\min_{\mathbf{w}, b} \frac{1}{2} ||\mathbf{w}||^2 \text{ subject to } y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \forall i
$$

### 4. Kernel Trick

For non-linearly separable data, SVM uses the kernel trick to transform data into a higher-dimensional space where it is linearly separable. Common kernels include polynomial, radial basis function (RBF), and sigmoid.

### 5. Prediction

The prediction for a new input $$\mathbf{x}_{\text{new}}$$ is based on the sign of the decision function. The class label is determined by whether the function's output is above or below the decision boundary.

### 6. Advantages and Disadvantages

- **Advantages**: Effective in high-dimensional spaces, memory efficient, and versatile (different kernel functions can be specified).
- **Disadvantages**: Requires careful choice of the kernel, not suitable for large datasets, and the results are not directly probabilistic.

