# AdaBoost (Adaptive Boosting)

## Basics

* AdaBoost is a popular ensemble learning technique that combines multiple weak learners to create a strong classifier.
* Each weak learner is typically a simple decision tree.
* The algorithm focuses more on training instances that previous weak learners classified incorrectly.

## Algorithm

### 1. Model Representation

AdaBoost combines multiple weak classifiers to form a strong classifier. Each weak classifier typically makes decisions based on a single feature or a small subset of features.

### 2. Initialize Weights

All training data points are given an initial weight: $$w_i = \frac{1}{N}$$, where $$N$$ is the number of data points and $$i$$ is the index of each data point.

### 3. Train Weak Learners

For each iteration $$t$$:
   - A weak learner $$L_t$$ is trained on the weighted training data.
   - The learner's error rate $$\epsilon_t$$ is calculated: $$\epsilon_t = \frac{\sum_{i=1}^{N} w_i \cdot \text{error}(i)}{\sum_{i=1}^{N} w_i}$$ where $$\text{error}(i)$$ is 1 if the prediction is wrong, and 0 if correct.

### 4. Update Weights

After training a weak learner:
   - The weight of the learner $$\alpha_t$$ is calculated: $$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t + 1e-10}\right)$$.
   - Update the weights of the instances: 
     - If correctly classified: $$w_i = w_i \cdot e^{-\alpha_t \cdot y_i \cdot \text{prediction}_i}$$.
     - If incorrectly classified, the weight remains the same.
   - Normalize weights: $$w_i = \frac{w_i}{\sum_{j=1}^{N} w_j}$$.

### 5. Combine Weak Learners

After training all weak learners, they are combined into a final classifier. Each weak learner's contribution is weighted based on its accuracy: $$F(x) = \sum_{t=1}^{T} \alpha_t \cdot L_t(x)$$.

### 6. Final Model

The final model makes predictions based on the sign of the weighted majority vote (or sum) of the weak learners' predictions: $$\text{y\_pred} = \text{sign}\left(\sum_{t=1}^{T} \alpha_t \cdot L_t(x)\right)$$.

### 7. Advantages and Disadvantages

- **Advantages**: Can be used with a variety of classifiers, improves classification accuracy, and is less prone to overfitting.
- **Disadvantages**: Sensitive to noisy data and outliers, and performance depends on data and weak learner.

AdaBoost is widely used for binary classification tasks, such as face detection and other object recognition tasks. It is known for its effectiveness in improving the accuracy of weak models and dealing with imbalanced datasets.
