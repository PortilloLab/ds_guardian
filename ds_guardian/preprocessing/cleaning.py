from typing import Union, Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from ..exceptions import DataValidationError
from ..logging import get_logger

logger = get_logger(__name__)

def imputar_nulos(
    df_train: pd.DataFrame, 
    df_test: Optional[pd.DataFrame] = None, 
    estrategia_num: str = 'median', 
    estrategia_cat: str = 'most_frequent',
    batch_size: Optional[int] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    if df_train is None or df_train.empty:
        raise DataValidationError("El DataFrame de entrenamiento está vacío.")
        
    df_train_imp = df_train.copy()
    cols_num = df_train_imp.select_dtypes(include=[np.number]).columns
    cols_cat = df_train_imp.select_dtypes(exclude=[np.number]).columns

    imputer_num = SimpleImputer(strategy=estrategia_num)
    imputer_cat = SimpleImputer(strategy=estrategia_cat)
    
    if len(cols_num) > 0:
        df_train_imp[cols_num] = imputer_num.fit_transform(df_train_imp[cols_num])
    if len(cols_cat) > 0:
        df_train_imp[cols_cat] = imputer_cat.fit_transform(df_train_imp[cols_cat])
        
    logger.info(f"Train imputado: {df_train_imp.isnull().sum().sum()} nulos restantes.")
    
    if df_test is not None:
        df_test_imp = df_test.copy()
        if batch_size and len(df_test_imp) > batch_size:
            logger.info(f"Procesando Test en batches de tamaño {batch_size}...")
            chunks = []
            for i in range(0, len(df_test_imp), batch_size):
                chunk = df_test_imp.iloc[i:i+batch_size].copy()
                if len(cols_num) > 0:
                    chunk[cols_num] = imputer_num.transform(chunk[cols_num])
                if len(cols_cat) > 0:
                    chunk[cols_cat] = imputer_cat.transform(chunk[cols_cat])
                chunks.append(chunk)
            df_test_imp = pd.concat(chunks, axis=0)
        else:
            if len(cols_num) > 0:
                df_test_imp[cols_num] = imputer_num.transform(df_test_imp[cols_num])
            if len(cols_cat) > 0:
                df_test_imp[cols_cat] = imputer_cat.transform(df_test_imp[cols_cat])
                
        return df_train_imp, df_test_imp
        
    return df_train_imp

def tratar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    duplicados = df.duplicated().sum()
    if duplicados > 0:
        pct = (duplicados / len(df)) * 100
        logger.info(f"Eliminadas {duplicados} filas duplicadas ({pct:.2f}%).")
        return df.drop_duplicates().reset_index(drop=True)
    return df

def acotar_outliers_iqr(df: pd.DataFrame, columnas: Optional[List[str]] = None, factor: float = 1.5) -> pd.DataFrame:
    if df is None or df.empty:
        raise DataValidationError("El DataFrame está vacío.")
    df_clean = df.copy()
    if columnas is None:
        columnas = list(df_clean.select_dtypes(include=[np.number]).columns)
    for col in columnas:
        if col in df_clean.columns and np.issubdtype(df_clean[col].dtype, np.number):
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
    return df_clean
