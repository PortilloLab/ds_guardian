import pandas as pd
import numpy as np
from typing import Optional, List
from .preprocessing.cleaning import acotar_outliers_iqr

def optimizar_memoria(df: pd.DataFrame) -> pd.DataFrame:
    """Optimiza el uso de memoria RAM reduciendo dtypes de numéricas."""
    if df is None or df.empty:
        return df
    df_opt = df.copy()
    start_mem = df_opt.memory_usage().sum() / 1024**2
    for col in df_opt.select_dtypes(include=[np.number]).columns:
        col_type = df_opt[col].dtypes
        c_min = df_opt[col].min()
        c_max = df_opt[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df_opt[col] = df_opt[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df_opt[col] = df_opt[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df_opt[col] = df_opt[col].astype(np.int32)
        else:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df_opt[col] = df_opt[col].astype(np.float32)
    end_mem = df_opt.memory_usage().sum() / 1024**2
    print(f"Memoria optimizada: de {start_mem:.2f} MB a {end_mem:.2f} MB.")
    return df_opt

def resumir_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Genera una tabla de resumen con dtypes, nulos y valores únicos."""
    return pd.DataFrame({
        'dtype': df.dtypes,
        'nulos': df.isnull().sum(),
        'pct_nulos': (df.isnull().sum() / len(df)) * 100,
        'unicos': df.nunique()
    })
