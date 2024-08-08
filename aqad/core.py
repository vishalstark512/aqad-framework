import numpy as np
import pandas as pd
from typing import List, Tuple, Any
from .quantization import apply_quantization_schemes
from .aed import AEDEnsemble
from .thresholding import AdaptiveThreshold

class AQADFramework:
    def __init__(self, base_model: Any, quantization_schemes: List[Any], 
                 aed_models: List[Any], threshold_model: Any, 
                 categorical_columns: List[str] = None,
                 feature_names: List[str] = None):
        
        self.base_model = base_model
        self.quantization_schemes = quantization_schemes
        self.aed_ensemble = AEDEnsemble(aed_models)
        self.threshold_model = threshold_model
        self.categorical_columns = categorical_columns
        self.feature_names = feature_names

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Convert X to DataFrame if it's not already
        X = self._ensure_dataframe(X)
        
        # Apply quantization schemes
        X_quantized = apply_quantization_schemes(X, self.quantization_schemes, self.categorical_columns)
        
        # Generate logits
        original_logits = self.base_model.predict_proba(X)
        quantized_logits = [self.base_model.predict_proba(X_q) for X_q in X_quantized]
        
        # Compute logit differences
        logit_diffs = self._compute_logit_differences(original_logits, quantized_logits)
        
        # Train AED ensemble
        self.aed_ensemble.fit(logit_diffs, y)
        
        # Train adaptive threshold
        aed_scores = self.aed_ensemble.predict(logit_diffs)
        self.threshold_model.fit(aed_scores, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Convert X to DataFrame if it's not already
        X = self._ensure_dataframe(X)
        
        # Apply quantization schemes
        X_quantized = apply_quantization_schemes(X, self.quantization_schemes, self.categorical_columns)
        
        # Generate logits
        original_logits = self.base_model.predict_proba(X)
        quantized_logits = [self.base_model.predict_proba(X_q) for X_q in X_quantized]
        
        # Compute logit differences
        logit_diffs = self._compute_logit_differences(original_logits, quantized_logits)
        
        # Get AED ensemble scores
        aed_scores = self.aed_ensemble.predict(logit_diffs)
        
        # Apply threshold
        predictions = self.threshold_model.predict(aed_scores)
        
        return predictions

    def _compute_logit_differences(self, original_logits: np.ndarray, 
                                   quantized_logits: List[np.ndarray]) -> np.ndarray:
        diffs = [original_logits - q_logits for q_logits in quantized_logits]
        return np.concatenate(diffs, axis=1)

    def _ensure_dataframe(self, X: np.ndarray) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return X