from typing import Union, Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from .exceptions import DataValidationError

def imputar_nulos(
    df_train: pd.DataFrame, 
    df_test: Optional[pd.DataFrame] = None, 
    estrategia_num: str = 'median', 
    estrategia_cat: str = 'most_frequent'
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Imputa nulos evitando Data Leakage. Ajusta los imputadores solo en Train y transforma Train y Test.
    
    Args:
        df_train: DataFrame de entrenamiento.
        df_test: DataFrame de test (opcional).
        estrategia_num: Estrategia de imputación para numéricas (median, mean, etc.).
        estrategia_cat: Estrategia de imputación para categóricas.
        
    Returns:
        Si df_test es None: DataFrame de train imputado.
        Si df_test no es None: Tupla (df_train_imp, df_test_imp).
        
    Raises:
        DataValidationError: Si df_train es None o vacío.
    """
    if df_train is None or df_train.empty:
        raise DataValidationError("El DataFrame de entrenamiento está vacío o es None.")
        
    df_train_imp = df_train.copy()
    
    # Separar numéricas y categóricas
    cols_num = df_train_imp.select_dtypes(include=[np.number]).columns
    cols_cat = df_train_imp.select_dtypes(exclude=[np.number]).columns

    # Imputadores
    imputer_num = SimpleImputer(strategy=estrategia_num)
    imputer_cat = SimpleImputer(strategy=estrategia_cat)
    
    # Ajustar y transformar en Train
    if len(cols_num) > 0:
        df_train_imp[cols_num] = imputer_num.fit_transform(df_train_imp[cols_num])
    if len(cols_cat) > 0:
        df_train_imp[cols_cat] = imputer_cat.fit_transform(df_train_imp[cols_cat])
        
    print(f"Train imputado: {df_train_imp.isnull().sum().sum()} nulos restantes.")
    
    # Si hay Test, transformarlo usando lo aprendido de Train
    if df_test is not None:
        if df_test.empty:
            raise DataValidationError("El DataFrame de prueba (test) está vacío.")
        df_test_imp = df_test.copy()
        if len(cols_num) > 0:
            df_test_imp[cols_num] = imputer_num.transform(df_test_imp[cols_num])
        if len(cols_cat) > 0:
            df_test_imp[cols_cat] = imputer_cat.transform(df_test_imp[cols_cat])
            
        print(f"Test imputado: {df_test_imp.isnull().sum().sum()} nulos restantes.")
        return df_train_imp, df_test_imp
        
    return df_train_imp

def tratar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica y elimina filas duplicadas, informando de la operación.
    
    Args:
        df: DataFrame de pandas.
        
    Returns:
        DataFrame sin duplicados.
    """
    if df is None or df.empty:
        return df
        
    duplicados = df.duplicated().sum()
    if duplicados > 0:
        print(f"Se encontraron {duplicados} filas duplicadas ({(duplicados/len(df))*100:.2f}%).")
        df_limpio = df.drop_duplicates().reset_index(drop=True)
        print("Duplicados eliminados.")
        return df_limpio
    else:
        print("No se encontraron filas duplicadas.")
        return df

def codificar_variables(
    df_train: pd.DataFrame, 
    df_test: Optional[pd.DataFrame] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Aplica One-Hot Encoding a las variables categóricas asegurando
    que Train y Test terminen con las mismas columnas.
    
    Args:
        df_train: DataFrame de entrenamiento.
        df_test: DataFrame de test (opcional).
        
    Returns:
        DataFrame(s) codificados.
        
    Raises:
        DataValidationError: Si df_train es None o vacío.
    """
    if df_train is None or df_train.empty:
        raise DataValidationError("El DataFrame de entrenamiento está vacío o es None.")
        
    df_train_enc = pd.get_dummies(df_train, drop_first=True)
    
    if df_test is not None:
        if df_test.empty:
            raise DataValidationError("El DataFrame de prueba (test) está vacío.")
        df_test_enc = pd.get_dummies(df_test, drop_first=True)
        
        # Alinear columnas (añadir faltantes con 0, quitar sobrantes en test)
        df_train_enc, df_test_enc = df_train_enc.align(df_test_enc, join='left', axis=1, fill_value=0)
        
        # Convertir a tipos booleanos/enteros si get_dummies genera booleanos en versiones recientes de pandas
        df_train_enc = df_train_enc.astype(float)
        df_test_enc = df_test_enc.astype(float)
        
        print(f"Variables codificadas. Train y Test tienen ahora {df_train_enc.shape[1]} columnas.")
        return df_train_enc, df_test_enc
        
    df_train_enc = df_train_enc.astype(float)
    print(f"Variables codificadas. El DataFrame tiene ahora {df_train_enc.shape[1]} columnas.")
    return df_train_enc

def escalar_caracteristicas(
    df_train: pd.DataFrame, 
    df_test: Optional[pd.DataFrame] = None, 
    metodo: str = 'standard', 
    columnas: Optional[List[str]] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Escala variables numéricas de forma segura evitando Data Leakage (fuga de datos).
    Ajusta el escalador solo en Train y lo aplica en Train y Test por separado.
    Soporta 'standard' (StandardScaler) y 'minmax' (MinMaxScaler).
    
    Args:
        df_train: DataFrame de entrenamiento.
        df_test: DataFrame de test (opcional).
        metodo: Método de escalamiento ('standard' o 'minmax').
        columnas: Lista de columnas a escalar. Si es None, busca numéricas no booleanas.
        
    Returns:
        DataFrame(s) escalados.
        
    Raises:
        DataValidationError: Si df_train está vacío o el método no es soportado.
    """
    if df_train is None or df_train.empty:
        raise DataValidationError("El DataFrame de entrenamiento está vacío o es None.")
        
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    df_train_scaled = df_train.copy()
    
    if columnas is None:
        # Excluir booleanos que a veces se detectan como np.number pero no se escalan
        columnas = [col for col in df_train_scaled.select_dtypes(include=[np.number]).columns 
                    if len(df_train_scaled[col].unique()) > 2]
        
    if len(columnas) == 0:
        print("No se encontraron variables numéricas adecuadas para escalar.")
        if df_test is not None:
            return df_train_scaled, df_test.copy()
        return df_train_scaled
        
    if metodo == 'standard':
        scaler = StandardScaler()
    elif metodo == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Método no soportado. Usa 'standard' o 'minmax'.")
        
    # Ajustar y transformar en Train
    df_train_scaled[columnas] = scaler.fit_transform(df_train_scaled[columnas])
    print(f"Características escaladas usando {metodo} en columnas: {list(columnas)}")
    
    if df_test is not None:
        if df_test.empty:
            raise DataValidationError("El DataFrame de prueba (test) está vacío.")
        df_test_scaled = df_test.copy()
        df_test_scaled[columnas] = scaler.transform(df_test_scaled[columnas])
        return df_train_scaled, df_test_scaled
        
    return df_train_scaled
