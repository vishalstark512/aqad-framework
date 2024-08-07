import yaml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_model(model_config: Dict[str, Any]) -> Any:
    model_type = model_config['type']
    model_params = model_config['params']

    if model_type == 'RandomForestClassifier':
        return RandomForestClassifier(**model_params)
    elif model_type == 'GradientBoostingClassifier':
        return GradientBoostingClassifier(**model_params)
    elif model_type == 'LogisticRegression':
        return LogisticRegression(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_quantization_scheme(scheme_config: Dict[str, Any]):
    scheme_type = scheme_config['type']
    scheme_params = scheme_config['params']

    if scheme_type == 'uniform':
        from .quantization import uniform_quantization
        return lambda X: uniform_quantization(X, **scheme_params)
    elif scheme_type == 'quantile':
        from .quantization import quantile_quantization
        return lambda X: quantile_quantization(X, **scheme_params)
    else:
        raise ValueError(f"Unsupported quantization scheme: {scheme_type}")