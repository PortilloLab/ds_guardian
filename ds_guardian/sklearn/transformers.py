import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class SafeImputerTransformer(BaseEstimator, TransformerMixin):
    """Imputador seguro compatible con Scikit-Learn Pipelines."""
    def __init__(self, strategy_num='median', strategy_cat='most_frequent'):
        self.strategy_num = strategy_num
        self.strategy_cat = strategy_cat
        self.imputer_num = None
        self.imputer_cat = None
        self.cols_num_ = []
        self.cols_cat_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
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

class SafeOneHotTransformer(BaseEstimator, TransformerMixin):
    """Encoder seguro One-Hot compatible con Scikit-Learn Pipelines."""
    def __init__(self):
        self.columns_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_enc = pd.get_dummies(X_df, drop_first=True)
        self.columns_ = list(X_enc.columns)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_enc = pd.get_dummies(X_df, drop_first=True)
        X_enc = X_enc.reindex(columns=self.columns_, fill_value=0)
        return X_enc.astype(float)

class SafeScalerTransformer(BaseEstimator, TransformerMixin):
    """Escalador seguro compatible con Scikit-Learn Pipelines."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.cols_num_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.cols_num_ = list(X_df.select_dtypes(include=[np.number]).columns)
        if self.cols_num_:
            self.scaler.fit(X_df[self.cols_num_])
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        if self.cols_num_:
            X_df[self.cols_num_] = self.scaler.transform(X_df[self.cols_num_])
        return X_df
