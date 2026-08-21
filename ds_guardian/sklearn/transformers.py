import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class SafeImputerTransformer(BaseEstimator, TransformerMixin):
    """Transformer seguro compatible con Scikit-Learn Pipelines."""
    def __init__(self, strategy_num='median', strategy_cat='most_frequent'):
        self.strategy_num = strategy_num
        self.strategy_cat = strategy_cat
        self.imputer_num = SimpleImputer(strategy=self.strategy_num)
        self.imputer_cat = SimpleImputer(strategy=self.strategy_cat)

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        cols_num = X_df.select_dtypes(include=['number']).columns
        cols_cat = X_df.select_dtypes(exclude=['number']).columns
        
        if len(cols_num) > 0:
            self.imputer_num.fit(X_df[cols_num])
        if len(cols_cat) > 0:
            self.imputer_cat.fit(X_df[cols_cat])
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        cols_num = X_df.select_dtypes(include=['number']).columns
        cols_cat = X_df.select_dtypes(exclude=['number']).columns
        
        if len(cols_num) > 0:
            X_df[cols_num] = self.imputer_num.transform(X_df[cols_num])
        if len(cols_cat) > 0:
            X_df[cols_cat] = self.imputer_cat.transform(X_df[cols_cat])
        return X_df

class SafeScalerTransformer(BaseEstimator, TransformerMixin):
    """Scaler seguro compatible con Scikit-Learn Pipelines."""
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.scaler.fit(X_df)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        scaled = self.scaler.transform(X_df)
        return pd.DataFrame(scaled, columns=X_df.columns, index=X_df.index)
