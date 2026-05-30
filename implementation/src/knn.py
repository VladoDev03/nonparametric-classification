import numpy as np
from .utils import euclidean_distance


class KNNClassifier:
    def __init__(self, k=3, rejection_threshold=0.0):
        """
        k: Number of neighbors to consider.
        rejection_threshold: Minimum proportion of the majority class required to make a prediction (0.0 to 1.0).
        """
        self.k = k
        self.rejection_threshold = rejection_threshold
        self.X_train = None
        self.y_train = None


    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)


    def _predict_single(self, x):
        # Calculate distances from the new observation to all training points
        distances = [euclidean_distance(x, train_x) for train_x in self.X_train]

        # Find the indices of the k nearest points
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]

        # Count the representation of individual classes
        labels, counts = np.unique(k_labels, return_counts=True)
        max_idx = np.argmax(counts)
        most_common_label = labels[max_idx]

        # Reject classification if the proportion is below the threshold
        proportion = counts[max_idx] / self.k

        if proportion < self.rejection_threshold:
            return -1

        return most_common_label


    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])
