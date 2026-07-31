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


def test_acotar_outliers_iqr_train_test_sin_leakage():
    """
    Los límites IQR deben calcularse ÚNICAMENTE con el set de train, y aplicarse
    igual (sin recalcular) sobre el set de test, para evitar Data Leakage.
    """
    df_train = pd.DataFrame({'A': [10.0, 11.0, 12.0, 13.0, 14.0]})  # sin outliers en train
    # Test tiene un valor extremo que, si contaminara el cálculo de los límites,
    # ensancharía el rango permitido. No debe pasar: el límite se calcula solo con train.
    df_test = pd.DataFrame({'A': [10.5, 9999.0]})

    df_train_capped, df_test_capped = acotar_outliers_iqr(df_train, columnas=['A'], df_test=df_test)

    # El valor extremo de test debe quedar acotado según los límites de TRAIN,
    # no debe sobrevivir como 9999.0.
    assert df_test_capped['A'].max() < 100.0
    # Train no debió modificarse por la presencia de test.
    assert df_train_capped['A'].max() == 14.0

def test_dataframe_vacio_raises_error():
    with pytest.raises(DataValidationError):
        from ds_guardian.eda import resumir_datos
        resumir_datos(None)
