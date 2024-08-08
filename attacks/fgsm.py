import numpy as np
import pandas as pd
from typing import Any, List
from sklearn.base import BaseEstimator

def fgsm_attack(model: BaseEstimator, X: pd.DataFrame, y: np.ndarray, epsilon: float, categorical_columns: List[str]) -> pd.DataFrame:
    """
    Perform FGSM attack on scikit-learn models, preserving categorical features.
    
    Args:
    model (BaseEstimator): A trained scikit-learn model
    X (pd.DataFrame): Input features
    y (np.ndarray): True labels
    epsilon (float): Perturbation size
    categorical_columns (List[str]): Names of categorical columns
    
    Returns:
    pd.DataFrame: Perturbed input features
    """
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
    else:
        raise ValueError("Model must have predict_proba method")
    
    # Initialize grad DataFrame with float dtype
    grad = pd.DataFrame(np.zeros(X.shape, dtype=float), index=X.index, columns=X.columns)
    
    for i in range(X.shape[0]):
        true_class = y[i]
        prob_true_class = probs[i, true_class]
        grad.iloc[i] = -1 * (1 - prob_true_class) * model.feature_importances_
    
    # Create adversarial examples
    X_adv = X.copy()
    for col in X.columns:
        if col not in categorical_columns:
            X_adv[col] = X[col].astype(float) + epsilon * np.sign(grad[col])
    
    # Clip numerical features to ensure valid values (assuming features are normalized)
    for col in X.columns:
        if col not in categorical_columns:
            X_adv[col] = np.clip(X_adv[col], 0, 1)
    
    return X_adv