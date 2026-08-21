from typing import Union, Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from ..exceptions import DataValidationError
from ..logging import get_logger

logger = get_logger(__name__)

def escalar_caracteristicas(
    df_train: pd.DataFrame, 
    df_test: Optional[pd.DataFrame] = None, 
    metodo: str = 'standard', 
    columnas: Optional[List[str]] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Escala variables numéricas ajustando el escalador únicamente en el conjunto de Train."""
    if df_train is None or df_train.empty:
        raise DataValidationError("El DataFrame de entrenamiento está vacío.")
        
    df_train_scaled = df_train.copy()
    
    if columnas is None:
        columnas = [col for col in df_train_scaled.select_dtypes(include=[np.number]).columns 
                    if len(df_train_scaled[col].unique()) > 2]
        
    if len(columnas) == 0:
        logger.info("No se encontraron columnas numéricas para escalar.")
        if df_test is not None:
            return df_train_scaled, df_test.copy()
        return df_train_scaled
        
    if metodo == 'standard':
        scaler = StandardScaler()
    elif metodo == 'minmax':
        scaler = MinMaxScaler()
    elif metodo == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Método '{metodo}' no soportado. Usa 'standard', 'minmax' o 'robust'.")
        
    df_train_scaled[columnas] = scaler.fit_transform(df_train_scaled[columnas])
    logger.info(f"Escalamiento ({metodo}) aplicado en Train en {len(columnas)} columnas.")
    
    if df_test is not None:
        if df_test.empty:
            raise DataValidationError("El DataFrame de prueba está vacío.")
        df_test_scaled = df_test.copy()
        df_test_scaled[columnas] = scaler.transform(df_test_scaled[columnas])
        logger.info(f"Escalamiento ({metodo}) aplicado en Test usando la distribución de Train.")
        return df_train_scaled, df_test_scaled
        
    return df_train_scaled
