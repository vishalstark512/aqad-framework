import numpy as np
from typing import List, Any
from sklearn.preprocessing import KBinsDiscretizer

def adaptive_quantization(X: np.ndarray, n_bins: int = 10, encode: str = 'ordinal', 
                          strategy: str = 'uniform') -> np.ndarray:
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    return discretizer.fit_transform(X)

def categorical_grouping(X: np.ndarray, n_groups: int = 5) -> np.ndarray:
    unique_values, inverse = np.unique(X, return_inverse=True)
    if len(unique_values) <= n_groups:
        return inverse
    else:
        value_counts = np.bincount(inverse)
        sorted_indices = np.argsort(value_counts)
        grouping = np.zeros_like(sorted_indices)
        grouping[sorted_indices] = np.repeat(range(n_groups), 
                                             np.ceil(len(unique_values) / n_groups))[:len(unique_values)]
        return grouping[inverse]

def apply_quantization_schemes(X: np.ndarray, schemes: List[Any], 
                               categorical_columns: List[int] = None) -> List[np.ndarray]:
    results = []
    for scheme in schemes:
        X_quantized = X.copy()
        if categorical_columns:
            for col in categorical_columns:
                X_quantized[:, col] = categorical_grouping(X[:, col])
        
        for col in range(X.shape[1]):
            if categorical_columns is None or col not in categorical_columns:
                X_quantized[:, col] = scheme(X[:, col].reshape(-1, 1)).ravel()
        
        results.append(X_quantized)
    return results