import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from typing import Tuple, Union, List

class DataLoader:
    def __init__(self, categorical_columns: List[str] = None, 
                 numerical_columns: List[str] = None):
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        self.preprocessor = None

    def load_data(self, path: str) -> pd.DataFrame:
        # Detect file type and load accordingly
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.parquet'):
            return pd.read_parquet(path)
        elif path.endswith('.feather'):
            return pd.read_feather(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.preprocessor:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numerical_columns),
                    ('cat', categorical_transformer, self.categorical_columns)
                ])

            self.preprocessor.fit(data)

        X = self.preprocessor.transform(data)
        y = data['target'].values if 'target' in data.columns else None
        return X, y

    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def load_and_preprocess(path: str, categorical_columns: List[str] = None, 
                        numerical_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(categorical_columns, numerical_columns)
    data = loader.load_data(path)
    return loader.preprocess_data(data)