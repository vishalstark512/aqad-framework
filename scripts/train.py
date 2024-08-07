import argparse
import numpy as np
from aqad.core import AQADFramework
from aqad.utils import load_config, get_model, get_quantization_scheme
from aqad.thresholding import AdaptiveThreshold
from data.data_loader import load_and_preprocess, DataLoader
from attacks.fgsm import fgsm_attack

def main(config_path: str, data_path: str):
    # Load configuration
    config = load_config(config_path)

    # Load and preprocess data
    categorical_columns = config.get('categorical_columns', [])
    numerical_columns = config.get('numerical_columns', [])
    X, y = load_and_preprocess(data_path, categorical_columns, numerical_columns)
    
    loader = DataLoader(categorical_columns, numerical_columns)
    X_train, X_test, y_train, y_test = loader.split_data(X, y)

    # Initialize base model
    base_model = get_model(config['base_model'])

    # Initialize quantization schemes
    quantization_schemes = [get_quantization_scheme(scheme) for scheme in config['quantization_schemes']]

    # Initialize AED models
    aed_models = [get_model(model_config) for model_config in config['aed_models']]

    # Initialize threshold model
    threshold_model = AdaptiveThreshold(get_model(config['threshold_model']))

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
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data", type=str, required=True, help="Path to data files")
    args = parser.parse_args()

    main(args.config, args.data)