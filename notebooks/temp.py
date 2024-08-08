import argparse
import numpy as np
import pandas as pd
import yaml
from aqad.core import AQADFramework
from aqad.utils import load_config, get_model, get_quantization_scheme
from aqad.thresholding import AdaptiveThreshold
from data.data_loader import load_and_preprocess, DataLoader
from attacks.fgsm import fgsm_attack


config_path = "D:/mine/aqad-framework/config/generated_config.yaml"
data_path = "D:/mine/aqad-framework/synthetic_dataset.csv"

# Load configuration
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Load and preprocess data
X, y, new_names = load_and_preprocess(config, data_path, one_hot_encode=False)

# Split data
loader = DataLoader(config)
X_train, X_test, y_train, y_test = loader.split_data(X, y)

print(X_train.shape, y_train.shape)

#converting it to a dataframe
temp = pd.DataFrame(X, columns=new_names)


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
                     categorical_columns=['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4'],
                     feature_names=list(temp.columns))

# Train base model
base_model.fit(X_train, y_train)

# Generate adversarial examples
X_train_adv = fgsm_attack(base_model, pd.DataFrame(X_train, columns=list(temp.columns)), y_train, epsilon=0.1, 
                          categorical_columns=['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4'])