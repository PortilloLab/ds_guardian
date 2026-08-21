import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class SafeImputerTransformer(BaseEstimator, TransformerMixin):
    """Transformer seguro compatible con Scikit-Learn Pipelines."""
    def __init__(self, strategy_num='median', strategy_cat='most_frequent'):
        self.strategy_num = strategy_num
        self.strategy_cat = strategy_cat
        self.imputer_num = None
        self.imputer_cat = None
        self.cols_num_ = []
        self.cols_cat_ = []
        self.feature_names_in_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.feature_names_in_ = list(X_df.columns)
        self.cols_num_ = list(X_df.select_dtypes(include=[np.number]).columns)
        self.cols_cat_ = list(X_df.select_dtypes(exclude=[np.number]).columns)
        
        self.imputer_num = SimpleImputer(strategy=self.strategy_num)
        self.imputer_cat = SimpleImputer(strategy=self.strategy_cat)
        
        if self.cols_num_:
            self.imputer_num.fit(X_df[self.cols_num_])
        if self.cols_cat_:
            self.imputer_cat.fit(X_df[self.cols_cat_])
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        if self.cols_num_:
            X_df[self.cols_num_] = self.imputer_num.transform(X_df[self.cols_num_])
        if self.cols_cat_:
            X_df[self.cols_cat_] = self.imputer_cat.transform(X_df[self.cols_cat_])
        return X_df

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_)

class SafeScalerTransformer(BaseEstimator, TransformerMixin):
    """Scaler seguro compatible con Scikit-Learn Pipelines."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names_in_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.feature_names_in_ = list(X_df.columns)
        self.scaler.fit(X_df)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        scaled = self.scaler.transform(X_df)
        return pd.DataFrame(scaled, columns=X_df.columns, index=X_df.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_)
