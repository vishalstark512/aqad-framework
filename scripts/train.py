import argparse
import numpy as np
import pandas as pd
import yaml
from aqad.core import AQADFramework
from aqad.utils import load_config, get_model, get_quantization_scheme
from aqad.thresholding import AdaptiveThreshold
from data.data_loader import load_and_preprocess, DataLoader
from attacks.fgsm import fgsm_attack
from typing import List, Any, Optional

def is_likely_categorical(series, threshold=10):
    if series.dtype == 'object':
        return True
    elif series.dtype.name == 'category':
        return True
    elif series.dtype.kind in 'iufc':
        return series.nunique() < min(threshold, len(series) * 0.05)
    return False

def generate_config(data_path: str, target_column: str, 
                    categorical_columns: Optional[List[str]] = None, 
                    numerical_columns: Optional[List[str]] = None):
    # Load data
    data = pd.read_csv(data_path)
    
    # Automatically identify categorical and numerical columns if not provided
    if categorical_columns is None and numerical_columns is None:
        categorical_columns = [col for col in data.columns if is_likely_categorical(data[col]) and col != target_column]
        numerical_columns = [col for col in data.columns if col not in categorical_columns and col != target_column]
    
    # Example configuration structure
    config = {
        'target_column': target_column,
        'categorical_columns': categorical_columns,
        'numerical_columns': numerical_columns,
        'base_model': {
            'type': "RandomForestClassifier",
            'params': {
                'n_estimators': 100,
                'max_depth': 10
            }
        },
        'quantization_schemes': [
            {
                'type': "adaptive",
                'params': {
                    'n_bins': 10,
                    'encode': "ordinal",
                    'strategy': "uniform"
                }
            },
            {
                'type': "adaptive",
                'params': {
                    'n_bins': 20,
                    'encode': "ordinal",
                    'strategy': "quantile"
                }
            }
        ],
        'aed_models': [
            {
                'type': "RandomForestClassifier",
                'params': {
                    'n_estimators': 50,
                    'max_depth': 5
                }
            },
            {
                'type': "GradientBoostingClassifier",
                'params': {
                    'n_estimators': 50,
                    'max_depth': 3
                }
            }
        ],
        'threshold_model': {
            'type': "LogisticRegression",
            'params': {
                'C': 1.0
            }
        }
    }
    
    # Save the generated config to a file
    config_path = 'generated_config.yaml'
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
        
    return config_path

def main(data_path: str):
    # Generate configuration file based on the data
    config_path = generate_config(data_path, target_column="target")
    
    # Load configuration
    config = load_config(config_path)

    # Load and preprocess data
    categorical_columns = config.get('categorical_columns', [])
    numerical_columns = config.get('numerical_columns', [])
    X, y = load_and_preprocess(data_path, categorical_columns, numerical_columns)
    
    loader = DataLoader(categorical_columns, numerical_columns)
    X_train, X_test, y_train, y_test = loader.split_data(X, y)

    print(X_train.shape, y_train.shape)

    # Initialize base model
    base_model = get_model(config['base_model'])

    # Initialize quantization schemes
    quantization_schemes = [get_quantization_scheme(scheme) for scheme in config['quantization_schemes']]

    # Initialize AED models
    aed_models = [get_model(model_config) for model_config in config['aed_models']]

    # Initialize threshold model
    threshold_model = AdaptiveThreshold(get_model(config['threshold_model']))

    print("Base Model: ", base_model, "Quantization Scheme: ", quantization_schemes, "AED Model: ", aed_models, "Threshold model: ", threshold_model)

    # Create AQAD framework
    aqad = AQADFramework(base_model, quantization_schemes, aed_models, threshold_model, 
                         categorical_columns=list(range(len(categorical_columns))))

    # Train base model
    base_model.fit(X_train, y_train)

    # Generate adversarial examples
    X_train_adv = fgsm_attack(base_model, X_train, y_train, epsilon=0.1)

    # Combine original and adversarial examples
    X_train_combined = np.vstack([X_train, X_train_adv])
    y_train_combined = np.hstack([y_train, y_train])

    # Train AQAD framework
    aqad.fit(X_train_combined, y_train_combined)

    # Evaluate
    y_pred = aqad.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test accuracy: {accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AQAD framework")
    parser.add_argument("--data", type=str, required=True, help="Path to data files")
    args = parser.parse_args()

    main(args.data)
