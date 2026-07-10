import pytest
import pandas as pd
import numpy as np
from ds_guardian.eda import optimizar_memoria, detectar_outliers_iqr, acotar_outliers_iqr
from ds_guardian.exceptions import DataValidationError

def test_optimizar_memoria():
    df = pd.DataFrame({
        'A': np.arange(100, dtype=np.int64),
        'B': np.arange(100, dtype=np.float64)
    })
    df_opt = optimizar_memoria(df)
    # Deben ser tipos menores
    assert df_opt['A'].dtype == np.int8
    assert df_opt['B'].dtype == np.float32

def test_detectar_outliers_iqr():
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 100, -100]})
    outliers = detectar_outliers_iqr(df)
    assert 'A' in outliers
    assert outliers['A'] == 2

def test_acotar_outliers_iqr():
    df = pd.DataFrame({'A': [10.0, 11.0, 12.0, 100.0, -50.0]})
    df_capped = acotar_outliers_iqr(df, columnas=['A'])
    # Los valores extremos no deben existir más allá de los límites del IQR
    assert df_capped['A'].max() < 100.0
    assert df_capped['A'].min() > -50.0

def test_dataframe_vacio_raises_error():
    with pytest.raises(DataValidationError):
        from ds_guardian.eda import resumir_datos
        resumir_datos(None)
