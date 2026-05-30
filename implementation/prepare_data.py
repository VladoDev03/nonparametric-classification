import os
import pandas as pd
from sklearn.datasets import load_iris, load_wine


def save_dataset_to_csv(loader_func, filename):
    # Load data from scikit-learn
    data = loader_func()

    # Create a DataFrame with the features
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Add the target labels column
    df['target'] = data.target

    # Save to CSV file
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', filename)
    df.to_csv(filepath, index=False)
    print(f"Successfully created: {filepath} ({df.shape[0]} rows, {df.shape[1] - 1} features)")


if __name__ == '__main__':
    print("Preparing datasets...")
    save_dataset_to_csv(load_iris, 'iris.csv')
    save_dataset_to_csv(load_wine, 'wine.csv')
    print("Done!")