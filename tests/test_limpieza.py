import pytest
import pandas as pd
import numpy as np
from ds_guardian.limpieza import imputar_nulos, tratar_duplicados, codificar_variables, escalar_caracteristicas
from ds_guardian.exceptions import DataValidationError

def test_imputar_nulos_sin_leakage():
    df_train = pd.DataFrame({'A': [1.0, 2.0, np.nan]})
    df_test = pd.DataFrame({'A': [np.nan]})
    
    # La media/mediana de Train es 1.5. Test debe imputarse con 1.5.
    df_train_imp, df_test_imp = imputar_nulos(df_train, df_test, estrategia_num='median')
    
    assert df_train_imp.isnull().sum().sum() == 0
    assert df_test_imp.isnull().sum().sum() == 0
    assert df_test_imp.loc[0, 'A'] == 1.5

def test_tratar_duplicados():
    df = pd.DataFrame({'A': [1, 1, 2]})
    df_limpio = tratar_duplicados(df)
    assert len(df_limpio) == 2

def test_codificar_variables():
    df_train = pd.DataFrame({'Cat': ['A', 'B', 'A']})
    df_test = pd.DataFrame({'Cat': ['B']})
    
    df_train_enc, df_test_enc = codificar_variables(df_train, df_test)
    assert 'Cat_B' in df_train_enc.columns
    assert 'Cat_B' in df_test_enc.columns
    assert len(df_train_enc.columns) == len(df_test_enc.columns)

def test_escalar_caracteristicas():
    df_train = pd.DataFrame({'Val': [10.0, 20.0, 30.0]})
    df_train_scaled = escalar_caracteristicas(df_train, metodo='minmax')
    assert df_train_scaled['Val'].min() == 0.0
    assert df_train_scaled['Val'].max() == 1.0


def test_imputar_nulos_rechaza_columnas_faltantes_en_test():
    df_train = pd.DataFrame({'A': [1.0, 2.0, np.nan]})
    df_test = pd.DataFrame({'B': [1.0, 2.0]})

    with pytest.raises(DataValidationError, match='faltan en df_test|No coinciden'):
        imputar_nulos(df_train, df_test)


def test_escalar_caracteristicas_rechaza_columnas_inexistentes():
    df_train = pd.DataFrame({'Val': [10.0, 20.0, 30.0]})

    with pytest.raises(DataValidationError, match='No existe|columnas'):
        escalar_caracteristicas(df_train, columnas=['NoExiste'])
