from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from .exceptions import DataValidationError

def resumir_datos(df: pd.DataFrame) -> None:
    """
    Imprime un resumen inicial del DataFrame de forma segura.
    
    Args:
        df: DataFrame de pandas a resumir.
        
    Raises:
        DataValidationError: Si el DataFrame está vacío o es None.
    """
    if df is None or df.empty:
        raise DataValidationError("El DataFrame está vacío o es None.")
    
    print(f"Shape: {df.shape[0]} filas, {df.shape[1]} columnas")
    print("\n--- Tipos de datos ---")
    print(df.dtypes)
    
    print("\n--- Estadísticas descriptivas ---")
    try:
        print(df.describe(include='all').T)
    except Exception as e:
        print(f"No se pudieron calcular estadísticas descriptivas. Error: {e}")

def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el porcentaje de valores nulos por columna de forma segura.
    
    Args:
        df: DataFrame de pandas.
        
    Returns:
        DataFrame con resumen de valores faltantes.
        
    Raises:
        DataValidationError: Si el DataFrame está vacío o es None.
    """
    if df is None or df.empty:
        raise DataValidationError("El DataFrame está vacío o es None.")
        
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'}
    )
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0
    ].sort_values('% of Total Values', ascending=False).round(1)
    
    print(f"Tu DataFrame tiene {df.shape[1]} columnas.\n"
          f"Hay {mis_val_table_ren_columns.shape[0]} columnas que tienen valores nulos.")
    return mis_val_table_ren_columns

def optimizar_memoria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce el consumo de memoria del DataFrame bajando los tipos de datos a los mínimos necesarios.
    
    Args:
        df: DataFrame de pandas.
        
    Returns:
        DataFrame optimizado.
    """
    if df is None or df.empty:
        return df
        
    mem_antes = df.memory_usage().sum() / 1024**2
    print(f"Uso de memoria inicial: {mem_antes:.2f} MB")
    
    # Hacer una copia para evitar SettingWithCopyWarning
    df_opt = df.copy()
    
    for col in df_opt.columns:
        col_type = df_opt[col].dtype
        
        if pd.api.types.is_numeric_dtype(df_opt[col]):
            c_min = df_opt[col].min()
            c_max = df_opt[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df_opt[col] = df_opt[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df_opt[col] = df_opt[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df_opt[col] = df_opt[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df_opt[col] = df_opt[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df_opt[col] = df_opt[col].astype(np.float32)
                else:
                    df_opt[col] = df_opt[col].astype(np.float64)
    
    mem_despues = df_opt.memory_usage().sum() / 1024**2
    print(f"Uso de memoria final: {mem_despues:.2f} MB")
    print(f"Reducción del {(100*(mem_antes - mem_despues)/mem_antes):.1f}%")
    return df_opt

def detectar_outliers_iqr(df: pd.DataFrame) -> Dict[str, int]:
    """
    Detecta y cuenta outliers para variables numéricas usando el método IQR.
    
    Args:
        df: DataFrame de pandas.
        
    Returns:
        Diccionario con las columnas y cantidad de outliers.
    """
    outliers_dict = {}
    if df is None or df.empty:
        return outliers_dict
        
    numericas = df.select_dtypes(include=[np.number]).columns
    
    for col in numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        cant_outliers = ((df[col] < limite_inferior) | (df[col] > limite_superior)).sum()
        if cant_outliers > 0:
            outliers_dict[col] = int(cant_outliers)
            
    if len(outliers_dict) > 0:
        print("\n--- Columnas con posibles Outliers (Método IQR) ---")
        for col, cant in sorted(outliers_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"{col}: {cant} outliers ({(cant/len(df))*100:.2f}%)")
    else:
        print("\nNo se detectaron outliers significativos con el método IQR.")
    
    return outliers_dict

def acotar_outliers_iqr(df: pd.DataFrame, columnas: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Acota los outliers en las columnas numéricas especificadas usando los límites del rango intercuartílico (IQR).
    (Capping/Winsorization). Modifica y retorna una copia del DataFrame.
    
    Args:
        df: DataFrame de pandas.
        columnas: Lista de columnas numéricas a acotar. Si es None, acota todas las numéricas.
        
    Returns:
        DataFrame con outliers acotados.
    """
    if df is None or df.empty:
        return df
        
    df_capped = df.copy()
    if columnas is None:
        columnas = list(df_capped.select_dtypes(include=[np.number]).columns)
        
    for col in columnas:
        if not pd.api.types.is_numeric_dtype(df_capped[col]):
            continue
        Q1 = df_capped[col].quantile(0.25)
        Q3 = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        # Acotar valores fuera de los límites
        df_capped[col] = np.clip(df_capped[col], limite_inferior, limite_superior)
        
    print(f"Outliers acotados en las columnas: {columnas}")
    return df_capped
