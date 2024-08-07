import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

class AdaptiveThreshold(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator or LogisticRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.base_estimator.fit(X.reshape(-1, 1), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.base_estimator.predict(X.reshape(-1, 1))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.base_estimator.predict_proba(X.reshape(-1, 1))