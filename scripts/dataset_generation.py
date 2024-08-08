import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def generate_synthetic_dataset(n_samples=1000, n_features=20, n_informative=10, 
                               n_redundant=5, n_classes=2, n_clusters_per_class=2,
                               n_categorical=5):
    # Generate numerical features
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                               n_informative=n_informative, n_redundant=n_redundant, 
                               n_classes=n_classes, n_clusters_per_class=n_clusters_per_class,
                               random_state=42)
    
    # Convert some features to categorical
    for i in range(n_categorical):
        X[:, i] = np.random.randint(0, 5, size=n_samples)  # 5 categories
    
    # Create a DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add some missing values
    for col in df.columns:
        mask = np.random.random(n_samples) < 0.05  # 5% missing values
        df.loc[mask, col] = np.nan
    
    # Add the target column
    df['target'] = y
    
    return df

# Generate a synthetic dataset
synthetic_data = generate_synthetic_dataset()

# Save the dataset
synthetic_data.to_csv('synthetic_dataset.csv', index=False)

print("Synthetic dataset shape:", synthetic_data.shape)
print("\nFeature types:")
print(synthetic_data.dtypes)
print("\nSample data:")
print(synthetic_data.head())