# Logistic Regression

## Basics

* Feature vectors: $${\mathbf{x}_i} \in \mathbb{R}^d$$, Target values (binary): $${y_i} \in \{0, 1\}$$
* $\mathbf{w} \in \mathbb{R}^d$, $b\in\mathbb{R}$
* Sigmoid function: $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

## Algorithm

### 1. Initialization

Initialize the parameters of the model. In logistic regression, these parameters are the weight vector $$\mathbf{w}$$ and the bias $$b$$. Typically, these are initialized to zeros or small random values.

- $$\mathbf{w} \in \mathbb{R}^d$$ represents the weight vector, where $$d$$ is the number of features.
- $$b \in \mathbb{R}$$ represents the bias term.

### 2. Define the Cost Function

The cost function in logistic regression, typically the binary cross-entropy or log loss, is defined as:

$$J(\mathbf{w},b) = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

where $$\hat{y}_i = \sigma(\mathbf{w}^T \mathbf{x}_i + b)$$ is the predicted probability that $$y_i = 1$$.

### 3. Calculate the Gradients

The gradients of $$J(\mathbf{w}, b)$$ with respect to $$\mathbf{w}$$ and $$b$$ are calculated as:

- Gradient with respect to $$\mathbf{w}$$:
  $$\frac{\partial J(\mathbf{w}, b)}{\partial \mathbf{w}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) \mathbf{x}_i$$
- Gradient with respect to $$b$$:
  $$\frac{\partial J(\mathbf{w}, b)}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)$$

### 4. Update Parameters using Gradient Descent

Update the parameters $$\mathbf{w}$$ and $$b$$ in each iteration of training:

- Update rule for the weights:
  $$\mathbf{w}_t = \mathbf{w}_{t-1} - \alpha \frac{\partial J(\mathbf{w}, b)}{\partial \mathbf{w}}$$
- Update rule for the bias:
  $$b_t = b_{t-1} - \alpha \frac{\partial J(\mathbf{w}, b)}{\partial b}$$

### 5. Iterate the Process

Repeat the gradient calculation and parameter update for a predefined number of iterations or until the cost function converges.

### 6. Model Training Completion

After training, the parameters $$\mathbf{w}$$ and $$b$$ will minimize the cost function. The model can then be used to predict probabilities of binary outcomes.

### 7. Making Predictions

For a new input feature vector $$\mathbf{x}_{\text{new}}$$, the predicted probability is calculated as:

$$\hat{y}_{\text{pred}} = \sigma(\mathbf{w}^T \mathbf{x}_{\text{new}} + b)$$

where $$\sigma$$ is the sigmoid function. The output can be interpreted as the probability of the instance belonging to the positive class.

Logistic regression is a fundamental algorithm for binary classification in machine learning, effectively used when the target variable is categorical.