import numpy as np

class DecisionStump:
    """
    A decision stump used as a weak learner in AdaBoost.
    """
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None
    
    def predict(self, X):
        """
        Make predictions using the decision stump.

        Parameters:
        X (ndarray): Data points.

        Returns:
        ndarray: Predictions for each data point.
        """
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions


class AdaBoost:
    """
    AdaBoost classifier.

    Attributes:
    n_clf (int): The number of weak learners (decision stumps) to use.
    """
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        """
        Fit the AdaBoost model on training data.

        Parameters:
        X (ndarray): Training data.
        y (ndarray): Target values.
        """
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))
        self.clfs = []

        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1
                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error

            clf.alpha = 0.5 * np.log((1.0 - min_error) / (min_error + 1e-10))
            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)
            self.clfs.append(clf)

    def predict(self, X):
        """
        Make predictions using the AdaBoost model.

        Parameters:
        X (ndarray): Data points to predict.

        Returns:
        ndarray: Predicted labels.
        """
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred
