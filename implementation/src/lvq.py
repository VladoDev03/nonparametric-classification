import numpy as np
from .utils import euclidean_distance


class LVQ:
    def __init__(self, n_codevectors_per_class=2, lr=0.1, epochs=100):
        """
        n_codevectors_per_class: Number of codevectors representing each class.
        lr: Learning rate (eta).
        epochs: Number of training iterations.
        """
        self.n_codevectors_per_class = n_codevectors_per_class
        self.lr = lr
        self.epochs = epochs
        self.codevectors = None
        self.codevector_labels = None


    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        unique_labels = np.unique(y)

        # Initialize codevectors by randomly selecting from the training set
        init_codevectors = []
        init_labels = []

        for label in unique_labels:
            class_samples = X[y == label]
            indices = np.random.choice(len(class_samples), self.n_codevectors_per_class, replace=False)

            for idx in indices:
                init_codevectors.append(class_samples[idx])
                init_labels.append(label)

        self.codevectors = np.array(init_codevectors)
        self.codevector_labels = np.array(init_labels)

        # Iterative training procedure
        for epoch in range(self.epochs):
            for x, target_label in zip(X, y):
                # Find the closest codevector m_c
                distances = [euclidean_distance(x, m) for m in self.codevectors]
                c_idx = np.argmin(distances)
                m_c = self.codevectors[c_idx]

                # Update m_c depending on whether the class matches
                if self.codevector_labels[c_idx] == target_label:
                    self.codevectors[c_idx] = m_c + self.lr * (x - m_c)  # Move closer
                else:
                    self.codevectors[c_idx] = m_c - self.lr * (x - m_c)  # Move away


    def predict(self, X):
        # 1-NN classification on the optimized codevectors
        predictions = []

        for x in X:
            distances = [euclidean_distance(x, m) for m in self.codevectors]
            closest_idx = np.argmin(distances)
            predictions.append(self.codevector_labels[closest_idx])

        return np.array(predictions)
