# Decision Tree

## Basics

* Feature vectors: $$\mathbf{x}_i \in \mathbb{R}^d$$, Target values: $$y_i$$ can be either categorical (for classification) or continuous (for regression).
* A decision tree is a flowchart-like tree structure where an internal node represents a feature, the branch represents a decision rule, and each leaf node represents the outcome.

### Mathematical Formulas

1. **Entropy**: A measure of the randomness or impurity in the dataset. It is calculated using the formula:
   
   $$
   H(S) = -\sum_{i=1}^{n} p_i \log_2 p_i
   $$

   Where $$H(S)$$ is the entropy of set $$S$$, $$n$$ is the number of classes, and $$p_i$$ is the proportion of class $$i$$ in the set.

2. **Gini Index**: A measure of impurity or variability, and is calculated using the formula:

   $$
   G(S) = 1 - \sum_{i=1}^{n} (p_i)^2
   $$

   Where $$G(S)$$ is the Gini index of set $$S$$, and $$p_i$$ is the proportion of class $$i$$ in the set.

### Information Gain

Information Gain is used to determine which feature to split on at a given node in the tree.

1. **Using Entropy**:

   Before the split:
   $$
   H(S)
   $$

   After the split on feature:
   $$
   H(S|feature) = \sum_{j=1}^{k} \frac{|S_j|}{|S|} H(S_j)
   $$

   Information Gain:
   $$
   IG(S, feature) = H(S) - H(S|feature)
   $$

2. **Using Gini Index**:

   Before the split:
   $$
   G(S)
   $$

   After the split on feature:
   $$
   G(S|feature) = \sum_{j=1}^{k} \frac{|S_j|}{|S|} G(S_j)
   $$

   Information Gain:
   $$
   IG(S, feature) = G(S) - G(S|feature)
   $$

### Example

Consider a dataset with two classes (A and B). Suppose in a particular node of the decision tree, there are 10 instances, 6 of class A and 4 of class B.

- The probability of picking a class A instance randomly is $$p(A) = \frac{6}{10} = 0.6$$
- The probability of picking a class B instance randomly is $$p(B) = \frac{4}{10} = 0.4$$

Using these probabilities, we can calculate the entropy and Gini index:

- **Entropy**: 
  $$
  H(S) = -[0.6 \log_2(0.6) + 0.4 \log_2(0.4)] \approx 0.971
  $$

- **Gini Index**:
  $$
  G(S) = 1 - [(0.6)^2 + (0.4)^2] = 0.48
  $$



# Approach

## Train algorithm: Build the tree

- Start at the top node and at each node select the best split based on the best information gain.
- Greedy search: Loop over all features and over all thresholds (all possible feature values).
- Save the best split feature and split threshold at each node.
- Build the tree recursively.
- Apply some stopping criteria to stop growing, e.g. here: maximum depth, minimum samples at node, no more class distribution in node.
- When we have a leaf node, store the most common class label of this node.

## Predict: Traverse tree

- Traverse the tree recursively.
- At each node look at the best split feature of the test feature vector x and go left or right depending on `x[feature_idx] <= threshold`.
- When we reach the leaf node we return the stored most common class label.