import numpy as np

"""
Implementation of Naive Bayes classifier. This classifier is based on applying 
Bayes' theorem with strong (naive) independence assumptions between the features.
"""

class NaiveBayes:
    """
    Naive Bayes Classifier using Gaussian distribution for continuous data.

    Attributes:
        classes (np.ndarray): Unique class labels in the dataset.
        mean (np.ndarray): Mean of features for each class.
        var (np.ndarray): Variance of features for each class.
        priors (np.ndarray): Prior probabilities for each class.
    """

    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit the Naive Bayes model to the training data.

        Args:
            features (np.ndarray): Training data, shape (n_samples, n_features).
            labels (np.ndarray): Training labels, shape (n_samples,).
        """
        n_samples, n_features = features.shape
        self.classes = np.unique(labels)
        n_classes = len(self.classes)

        # Initialize mean, var, and priors for each class
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            features_c = features[labels == c]
            self.mean[idx, :] = features_c.mean(axis=0)
            self.var[idx, :] = features_c.var(axis=0)
            self.priors[idx] = features_c.shape[0] / float(n_samples)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in features.

        Args:
            features (np.ndarray): Data to predict, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,).
        """
        predictions = [self._predict(feature) for feature in features]
        return np.array(predictions)

    def _predict(self, feature: np.ndarray) -> np.ndarray:
        posteriors = []

        for idx, _ in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, feature)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx: int, feature: np.ndarray) -> np.ndarray:
        """
        Probability density function for Gaussian Naive Bayes.

        Args:
            class_idx (int): Class index.
            feature (np.ndarray): Single data feature.

        Returns:
            np.ndarray: Probability density of feature given a class.
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-(feature - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
