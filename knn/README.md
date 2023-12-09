# KNN

## Basics

* Feature vectors: $${\mathbf{x}_i} \in \mathbb{R}^d$$, Target values: $${y_i} \in \mathbb{R}$$
* Euclidean distance: $$d(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{k=1}^{d}(\mathbf{x}_{ik} - \mathbf{x}_{jk})^2}$$ 

## Algorithm

1. **Identifying Neighbors**: For a given query point $$\mathbf{x}_q$$, the algorithm identifies the 'k' closest data points ($$\mathbf{x}_i$$) in the training set, based on the Euclidean distance:
   $$d(\mathbf{x}_q, \mathbf{x}_i) = \sqrt{\sum_{k=1}^{d}(\mathbf{x}_{qk} - \mathbf{x}_{ik})^2}$$

2. **Aggregate of Neighbors' Values**: 
   - In classification, the prediction ($$\hat{y}$$) is given by a majority vote among the 'k' nearest neighbors:
     $$\hat{y} = \text{mode}\{y_i : \mathbf{x}_i \text{ is among the k-nearest neighbors of } \mathbf{x}_q\}$$
   - In regression, the prediction is the mean or median of the target values of these neighbors:
     $$\hat{y} = \frac{1}{k} \sum_{\mathbf{x}_i \in \text{Nearest}_k(\mathbf{x}_q)} y_i$$

3. **Prediction**: 
   - For classification, the predicted class is the one with the highest vote among the neighbors.
   - For regression, the prediction is the computed mean or median value of the target values of the nearest neighbors.

