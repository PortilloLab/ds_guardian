import pytest
import pandas as pd
import numpy as np
from ds_guardian.preprocessing.cleaning import imputar_nulos, tratar_duplicados, acotar_outliers_iqr
from ds_guardian.preprocessing.encoding import codificar_variables
from ds_guardian.preprocessing.scaling import escalar_caracteristicas
from ds_guardian.auditoria.quality import revisar_datos_finales
from ds_guardian.sklearn.transformers import SafeImputerTransformer

def test_imputar_nulos_no_leakage():
    df_train = pd.DataFrame({'a': [1.0, 2.0, np.nan], 'b': ['x', 'y', None]})
    df_test = pd.DataFrame({'a': [np.nan, 4.0], 'b': [None, 'z']})
    
    train_imp, test_imp = imputar_nulos(df_train, df_test)
    assert train_imp.isnull().sum().sum() == 0
    assert test_imp.isnull().sum().sum() == 0

def test_codificar_variables_aligned():
    df_train = pd.DataFrame({'cat': ['A', 'B', 'A']})
    df_test = pd.DataFrame({'cat': ['A', 'C']})
    
    train_enc, test_enc = codificar_variables(df_train, df_test)
    assert train_enc.shape[1] == test_enc.shape[1]

def test_auditoria_datos_finales():
    df_clean = pd.DataFrame({'f1': [1.0, 2.0, 3.0], 'f2': [4.0, 5.0, 6.0]})
    y = pd.Series([0, 1, 0])
    assert revisar_datos_finales(df_clean, y) is True

def test_sklearn_transformer():
    df_train = pd.DataFrame({'a': [1.0, 2.0, np.nan]})
    transformer = SafeImputerTransformer()
    transformer.fit(df_train)
    res = transformer.transform(df_train)
    assert res.isnull().sum().sum() == 0
