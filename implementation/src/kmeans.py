import numpy as np
from .utils import euclidean_distance

class KMeansClustering:
    def __init__(self, k=3, max_iters=100):
        """
        k: Number of clusters.
        max_iters: Maximum number of iterations for convergence.
        """
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None


    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]

        # Initialize cluster centers randomly
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            old_labels = self.labels

            # Assign each observation to the nearest center
            labels = []

            for x in X:
                distances = [euclidean_distance(x, m) for m in self.centroids]
                labels.append(np.argmin(distances))

            self.labels = np.array(labels)

            # Recalculate cluster centers
            new_centroids = []

            for j in range(self.k):
                cluster_points = X[self.labels == j]

                # If a cluster becomes empty, drop it
                if len(cluster_points) == 0:
                    continue

                new_center = np.mean(cluster_points, axis=0)
                new_centroids.append(new_center)

            self.centroids = np.array(new_centroids)
            self.k = len(self.centroids)

            # Stop if the assignments no longer change
            if old_labels is not None and np.array_equal(self.labels, old_labels):
                break

        return self.labels
