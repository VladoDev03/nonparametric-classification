import numpy as np
import pandas as pd
from src import KNNClassifier


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
