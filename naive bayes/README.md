# Naive Bayes

## Basics

* Feature vectors: $${\mathbf{x}_i} \in \mathbb{R}^d$$, Target values (categorical): $${y_i} \in \{1, 2, \ldots, K\}$$
* Assumes feature independence given the class label.

## Algorithm

### 1. Model Representation

Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.

- The probability of a class given a feature vector: $$P(y|\mathbf{x})$$.

### 2. Calculate Class Probabilities

Calculate the prior probability for each class: $$P(y_k)$$, the likelihood: $$P(\mathbf{x}_i | y_k)$$, and the evidence: $$P(\mathbf{x}_i)$$.

### 3. Apply Bayes' Theorem

For a given feature vector $$\mathbf{x}_i$$, the classifier predicts the class $$y$$ that maximizes the posterior probability:

$$P(y|\mathbf{x}_i) = \frac{P(\mathbf{x}_i | y) P(y)}{P(\mathbf{x}_i)}$$

### 4. Dealing with Continuous Data

If features are continuous, assume Gaussian distribution to estimate the likelihood: 

$$P(\mathbf{x}_i | y_k) = \frac{1}{\sqrt{2\pi\sigma_{y_k}^2}} e^{ -\frac{(\mathbf{x}_i - \mu_{y_k})^2}{2\sigma_{y_k}^2} }$$

### 5. Prediction

For a new input feature vector $$\mathbf{x}_{\text{new}}$$, Naive Bayes predicts the class $$y_{\text{pred}}$$ by selecting the class with the highest posterior probability.

### 6. Advantages and Disadvantages

- **Advantages**: Simple, efficient, and works well with high-dimensional data.
- **Disadvantages**: Relies on an assumption of independent features, which is rarely true in real-world scenarios.

Naive Bayes is widely used for text classification tasks such as spam filtering and sentiment analysis. Despite its simplicity, it can yield surprisingly good results.
