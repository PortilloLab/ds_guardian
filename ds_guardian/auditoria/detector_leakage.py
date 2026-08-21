from typing import List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from ..logging import get_logger

logger = get_logger(__name__)

def detectar_multicolinealidad(df: pd.DataFrame, umbral: float = 0.95) -> List[Tuple[str, List[str]]]:
    """Escanea el DataFrame en busca de parejas de variables altamente correlacionadas."""
    df_num = df.select_dtypes(include=[np.number])
    if df_num.empty:
        return []
        
    corr_matrix = df_num.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    high_corr_pairs = []
    for col in upper.columns:
        correlated = upper.index[upper[col] > umbral].tolist()
        if correlated:
            high_corr_pairs.append((col, correlated))
    return high_corr_pairs

def detectar_fuga_target(df: pd.DataFrame, y: Any, umbral: float = 0.99) -> List[Tuple[str, float]]:
    """Detecta columnas con correlación peligrosamente alta con la variable objetivo (>0.99)."""
    df_num = df.select_dtypes(include=[np.number])
    if df_num.empty or y is None:
        return []
        
    y_series = pd.Series(y)
    leaks = []
    for col in df_num.columns:
        try:
            corr = float(np.abs(df_num[col].corr(y_series)))
            if corr >= umbral:
                leaks.append((col, corr))
        except Exception:
            pass
    return leaks
