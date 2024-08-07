import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    # This is a placeholder. Implement actual data loading logic based on your dataset.
    X = np.load(f"{path}_features.npy")
    y = np.load(f"{path}_labels.npy")
    return X, y

def preprocess_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Implement any necessary preprocessing steps
    # This is a simple example; adjust as needed
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state)