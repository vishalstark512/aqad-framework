import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from typing import Tuple, List
from sklearn.pipeline import Pipeline

class DataLoader:
    def __init__(self, config: dict, one_hot_encode: bool = True):
        self.categorical_columns = config['categorical_columns']
        self.numerical_columns = config['numerical_columns']
        self.target_column = config['target_column']
        self.one_hot_encode = one_hot_encode
        self.preprocessor = None
        self.feature_names = None

    def load_data(self, path: str) -> pd.DataFrame:
        if path.endswith('.csv'):
            data = pd.read_csv(path)
        elif path.endswith('.parquet'):
            data = pd.read_parquet(path)
        elif path.endswith('.feather'):
            data = pd.read_feather(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        data.dropna(subset=[self.target_column], inplace=True)
        return data

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column].values

        if not self.preprocessor:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            if self.one_hot_encode:
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value=np.nan)),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
            else:
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value=np.nan)),
                    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ])

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numerical_columns),
                    ('cat', categorical_transformer, self.categorical_columns)
                ])

            self.preprocessor.fit(X)

            numeric_features = self.numerical_columns
            if self.one_hot_encode:
                try:
                    # Try the new method name first
                    categorical_features = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_columns)
                except AttributeError:
                    # If that fails, try the old method name
                    categorical_features = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names(self.categorical_columns)
            else:
                categorical_features = self.categorical_columns

            self.feature_names = np.concatenate([numeric_features, categorical_features])

        X_preprocessed = self.preprocessor.transform(X)
        
        print("All features:", X.columns.tolist())
        print("Categorical features:", self.categorical_columns)
        print("Numerical features:", self.numerical_columns)
        print("Target variable:", self.target_column)
        print("Preprocessed feature names:", self.feature_names.tolist())
        print("One-hot encoding:", "Yes" if self.one_hot_encode else "No")
        
        return X_preprocessed, y, self.feature_names

    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def load_and_preprocess(config: dict, data_path: str, one_hot_encode: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(config, one_hot_encode)
    data = loader.load_data(data_path)
    return loader.preprocess_data(data)