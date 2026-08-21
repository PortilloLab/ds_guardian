from typing import Union, Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from ..exceptions import DataValidationError
from ..logging import get_logger

logger = get_logger(__name__)

def codificar_variables(
    df_train: pd.DataFrame, 
    df_test: Optional[pd.DataFrame] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Aplica One-Hot Encoding con alineación estricta (join='left') para evitar distorsión dimensional."""
    if df_train is None or df_train.empty:
        raise DataValidationError("El DataFrame de entrenamiento está vacío.")
        
    df_train_enc = pd.get_dummies(df_train, drop_first=True)
    
    if df_test is not None:
        if df_test.empty:
            raise DataValidationError("El DataFrame de test está vacío.")
        df_test_enc = pd.get_dummies(df_test, drop_first=True)
        df_train_enc, df_test_enc = df_train_enc.align(df_test_enc, join='left', axis=1, fill_value=0)
        df_train_enc = df_train_enc.astype(float)
        df_test_enc = df_test_enc.astype(float)
        logger.info(f"Variables codificadas (One-Hot). Columns: {df_train_enc.shape[1]}.")
        return df_train_enc, df_test_enc
        
    df_train_enc = df_train_enc.astype(float)
    logger.info(f"Variables codificadas (One-Hot). Total columnas: {df_train_enc.shape[1]}.")
    return df_train_enc

def codificar_target_encoder(
    df_train: pd.DataFrame, 
    y_train: pd.Series, 
    df_test: Optional[pd.DataFrame] = None, 
    columnas_cat: Optional[List[str]] = None,
    n_splits: int = 5
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Target Encoding seguro Out-of-Fold (OOF) para evitar fuga de información en categóricas."""
    if df_train is None or df_train.empty:
        raise DataValidationError("El DataFrame está vacío.")
        
    df_tr = df_train.copy()
    if columnas_cat is None:
        columnas_cat = list(df_tr.select_dtypes(include=['object', 'category']).columns)
        
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    global_mean = y_train.mean()
    
    for col in columnas_cat:
        oof_series = pd.Series(index=df_tr.index, dtype=float)
        for train_idx, val_idx in kf.split(df_tr):
            col_tr, y_tr = df_tr[col].iloc[train_idx], y_train.iloc[train_idx]
            means = y_tr.groupby(col_tr).mean()
            oof_series.iloc[val_idx] = df_tr[col].iloc[val_idx].map(means)
            
        df_tr[f"{col}_te"] = oof_series.fillna(global_mean)
        df_tr.drop(columns=[col], inplace=True)
        
    logger.info(f"Target Encoding Out-Of-Fold completado en {len(columnas_cat)} columnas.")
    
    if df_test is not None:
        df_te = df_test.copy()
        for col in columnas_cat:
            means = y_train.groupby(df_train[col]).mean()
            df_te[f"{col}_te"] = df_te[col].map(means).fillna(global_mean)
            df_te.drop(columns=[col], inplace=True)
        return df_tr, df_te
        
    return df_tr
