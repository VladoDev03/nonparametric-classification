import numpy as np
import pandas as pd
from src import KNNClassifier, LVQ, KMeansClustering


def load_and_preprocess(filepath):
    """Loads the dataset and applies Min-Max normalization (0 to 1)."""
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Min-Max Normalization: (X - min) / (max - min)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    # Adding a small epsilon to avoid division by zero
    X_normalized = (X - X_min) / (X_max - X_min + 1e-8)

    return X_normalized, y


def train_test_split(X, y, test_size=0.2):
    """Splits the dataset into training and testing subsets."""
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))

    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def calculate_accuracy(y_true, y_pred):
    """Calculates the percentage of correct predictions."""
    return np.mean(y_true == y_pred) * 100


def run_experiment(dataset_name, filepath, k_classes):
    print(f"\n{'=' * 45}")
    print(f" EXPERIMENT WITH DATASET: {dataset_name.upper()}")
    print(f"{'=' * 45}")

    X, y = load_and_preprocess(filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(f"Total samples: {len(X)}")
    print(f"Training size: {len(X_train)} | Testing size: {len(X_test)} | Features: {X.shape[1]}")

    # 1. k-NN Classification
    knn = KNNClassifier(k=5)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_acc = calculate_accuracy(y_test, knn_preds)
    print(f"\n[k-NN] Accuracy: {knn_acc:.2f}%")

    # 2. LVQ Classification
    lvq = LVQ(n_codevectors_per_class=3, lr=0.1, epochs=50)
    lvq.fit(X_train, y_train)
    lvq_preds = lvq.predict(X_test)
    lvq_acc = calculate_accuracy(y_test, lvq_preds)
    print(f"[LVQ] Accuracy: {lvq_acc:.2f}% (using {len(lvq.codevectors)} codevectors total)")

    # 3. k-means Clustering (Unsupervised)
    kmeans = KMeansClustering(k=k_classes)
    kmeans.fit(X)
    print(f"[k-means] Found centers: {len(kmeans.centroids)}")
    print("Note: k-means clusters the data based on distance, without using target labels.")


if __name__ == '__main__':
    # Ensure prepare_data.py has been executed first
    try:
        run_experiment("Iris", "data/iris.csv", k_classes=3)
        run_experiment("Wine", "data/wine.csv", k_classes=3)
    except FileNotFoundError:
        print("ERROR: Data files are missing. Please run prepare_data.py first!")
