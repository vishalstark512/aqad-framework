import numpy as np
from typing import List
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import List, Any


class AEDEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, models: List[Any]):
        self.models = models

    def fit(self, X: np.ndarray, y: np.ndarray):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)