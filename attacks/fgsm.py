import numpy as np
import torch
from typing import Any

def fgsm_attack(model: Any, X: np.ndarray, y: np.ndarray, epsilon: float) -> np.ndarray:
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    X_tensor.requires_grad = True
    
    outputs = model(X_tensor)
    loss = torch.nn.functional.cross_entropy(outputs, y_tensor)
    loss.backward()

    X_grad = X_tensor.grad.data.sign()
    perturbed_X = X_tensor + epsilon * X_grad
    perturbed_X = torch.clamp(perturbed_X, 0, 1)  # Ensure valid pixel values

    return perturbed_X.detach().numpy()