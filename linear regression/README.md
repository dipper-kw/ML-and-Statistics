# linear regression

## basics

* $\mathbf{x}_i \in \mathbb{R}^d$, $y_i \in \mathbb{R}$, $\mathbf{w} \in \mathbb{R}^d$, $b\in\mathbb{R}$

* $J(\mathbf{w},b)=\frac{1}{N}\sum_{i=1}^{n}(y_i - \mathbf{w}^T \mathbf{x}_i - b)$

* $$
  \begin{align*}
  \frac{\partial J(\mathbf{w}, b)}{\partial \mathbf{w}} &= \frac{-2}{N} \sum_{i=1}^{N} \mathbf{x}_i (y_i - \mathbf{w}^T \mathbf{x}_i - b) \\
  \frac{\partial J(\mathbf{w}, b)}{\partial b} &= \frac{-2}{N} \sum_{i=1}^{N} (y_i - \mathbf{w}^T \mathbf{x}_i - b)
  \end{align*}
  $$

* $\mathbf{w}_t = \mathbf{w}_{t-1} - \alpha d\mathbf{w}_{t-1}$



## algorithmn

### 1. Initialization

Start by initializing the parameters of the model. In linear regression, these parameters are the weight vector $$\mathbf{w}$$ and the bias $$b$$. Typically, these are initialized to zeros or small random values.

- $$\mathbf{w} \in \mathbb{R}^d$$ represents the weight vector, where $$d$$ is the number of features.
- $$b \in \mathbb{R}$$ represents the bias term.

### 2. Define the Cost Function

The cost function $$J(\mathbf{w},b)$$ is defined to measure the difference between the predicted values and the actual values. For linear regression, this is often the mean squared error (MSE):

$$J(\mathbf{w},b) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \mathbf{w}^T \mathbf{x}_i - b)^2$$

- $$\mathbf{x}_i$$ is the feature vector of the $$i$$-th sample.
- $$y_i$$ is the actual value for the $$i$$-th sample.
- $$N$$ is the total number of samples.

### 3. Calculate the Gradients

The gradients of $$J(\mathbf{w}, b)$$ with respect to $$\mathbf{w}$$ and $$b$$ are calculated. These gradients indicate the direction to adjust the parameters to minimize the cost function.

- Gradient with respect to $$\mathbf{w}$$:
  $$\frac{\partial J(\mathbf{w}, b)}{\partial \mathbf{w}} = \frac{-2}{N} \sum_{i=1}^{N} \mathbf{x}_i (y_i - \mathbf{w}^T \mathbf{x}_i - b)$$
- Gradient with respect to $$b$$:
  $$\frac{\partial J(\mathbf{w}, b)}{\partial b} = \frac{-2}{N} \sum_{i=1}^{N} (y_i - \mathbf{w}^T \mathbf{x}_i - b)$$

### 4. Update Parameters using Gradient Descent

In each iteration of training, update the parameters $$\mathbf{w}$$ and $$b$$ in the opposite direction of the gradient:

- Update rule for the weights:
  $$\mathbf{w}_t = \mathbf{w}_{t-1} - \alpha \frac{\partial J(\mathbf{w}, b)}{\partial \mathbf{w}}$$
- Update rule for the bias:
  $$b_t = b_{t-1} - \alpha \frac{\partial J(\mathbf{w}, b)}{\partial b}$$

Where $$\alpha$$ is the learning rate, a hyperparameter that controls the step size during the optimization.

### 5. Iterate the Process

Repeat steps 3 and 4 for a predefined number of iterations or until the cost function converges to a minimum value. The convergence criteria can be based on the change in cost function between iterations being below a certain threshold.

### 6. Model Training Completion

After the iterative process, the model's parameters $$\mathbf{w}$$ and $$b$$ will have been adjusted to minimize the cost function. The model is now trained and can be used to make predictions on new data.

### 7. Making Predictions

For a new input feature vector $$\mathbf{x}_{\text{new}}$$, the predicted output $$y_{\text{pred}}$$ is calculated as:

$$y_{\text{pred}} = \mathbf{w}^T \mathbf{x}_{\text{new}} + b$$

This algorithm outlines the fundamental approach to training a linear regression model using gradient descent, a common optimization technique in machine learning.
