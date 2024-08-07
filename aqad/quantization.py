import numpy as np
from typing import List, Any

def uniform_quantization(X: np.ndarray, num_bins: int) -> np.ndarray:
    return np.digitize(X, bins=np.linspace(X.min(), X.max(), num_bins))

def quantile_quantization(X: np.ndarray, num_quantiles: int) -> np.ndarray:
    quantiles = np.percentile(X, np.linspace(0, 100, num_quantiles))
    return np.digitize(X, bins=quantiles)

def apply_quantization_schemes(X: np.ndarray, schemes: List[Any]) -> List[np.ndarray]:
    return [scheme(X) for scheme in schemes]