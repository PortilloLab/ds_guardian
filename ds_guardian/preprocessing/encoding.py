from typing import Union, Tuple, Optional
import pandas as pd
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
        
        # Alineamiento perfecto con relleno cero para columnas no presentes en test
        df_train_enc, df_test_enc = df_train_enc.align(df_test_enc, join='left', axis=1, fill_value=0)
        
        df_train_enc = df_train_enc.astype(float)
        df_test_enc = df_test_enc.astype(float)
        
        logger.info(f"Variables codificadas. Train y Test alineados con {df_train_enc.shape[1]} columnas.")
        return df_train_enc, df_test_enc
        
    df_train_enc = df_train_enc.astype(float)
    logger.info(f"Variables codificadas. Total columnas: {df_train_enc.shape[1]}.")
    return df_train_enc
