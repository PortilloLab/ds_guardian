import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union
from .exceptions import DataValidationError
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
    if df is None or df.empty:
        raise DataValidationError("El DataFrame a resumir está vacío o es None.")
    return pd.DataFrame({
        'dtype': df.dtypes,
        'nulos': df.isnull().sum(),
        'pct_nulos': (df.isnull().sum() / len(df)) * 100,
        'unicos': df.nunique(),
    })


def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un resumen de nulos por columna."""
    if df is None or df.empty:
        raise DataValidationError("El DataFrame no puede estar vacío.")
    return pd.DataFrame({
        'nulos': df.isnull().sum(),
        'pct_nulos': (df.isnull().sum() / len(df)) * 100,
    })


def detectar_outliers_iqr(df: pd.DataFrame, columnas: Optional[List[str]] = None, factor: float = 1.5) -> Dict[str, int]:
    """Detecta outliers por rango IQR y devuelve un diccionario por columna."""
    if df is None or df.empty:
        raise DataValidationError("El DataFrame a auditar está vacío o es None.")
    df_clean = df.copy()
    if columnas is None:
        columnas = list(df_clean.select_dtypes(include=[np.number]).columns)
    outliers = {}
    for col in columnas:
        if col in df_clean.columns and np.issubdtype(df_clean[col].dtype, np.number):
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            n_outliers = int(((df_clean[col] < lower) | (df_clean[col] > upper)).sum())
            outliers[col] = n_outliers
    return outliers


def acotar_outliers_iqr(df: pd.DataFrame, columnas: Optional[List[str]] = None, factor: float = 1.5, df_test: Optional[pd.DataFrame] = None) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
    """Aplica capping IQR usando los límites calculados sobre Train y, opcionalmente, aplica el mismo límite sobre Test."""
    if df is None or df.empty:
        raise DataValidationError("El DataFrame está vacío.")
    df_clean = df.copy()
    if columnas is None:
        columnas = list(df_clean.select_dtypes(include=[np.number]).columns)
    bounds = {}
    for col in columnas:
        if col in df_clean.columns and np.issubdtype(df_clean[col].dtype, np.number):
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            bounds[col] = (lower, upper)
            df_clean[col] = np.clip(df_clean[col], lower, upper)
    if df_test is not None:
        if df_test.empty:
            raise DataValidationError("El DataFrame de prueba (test) está vacío.")
        missing_cols = [col for col in columnas if col not in df_test.columns]
        if missing_cols:
            raise DataValidationError(f"Las columnas indicadas para acotar outliers faltan en df_test: {missing_cols}")
        df_test_clean = df_test.copy()
        for col, (lower, upper) in bounds.items():
            if col in df_test_clean.columns:
                df_test_clean[col] = np.clip(df_test_clean[col], lower, upper)
        return df_clean, df_test_clean
    return df_clean
