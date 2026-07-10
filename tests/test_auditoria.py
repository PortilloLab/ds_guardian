import pytest
import pandas as pd
import numpy as np
from ds_guardian.auditoria import revisar_datos_finales
from ds_guardian.exceptions import DataValidationError

def test_revisar_datos_finales_con_nulos():
    df = pd.DataFrame({'A': [1.0, 2.0, np.nan]})
    # Si hay nulos, no pasa el auditor básico (retorna False)
    assert revisar_datos_finales(df) is False

def test_revisar_datos_finales_con_textos():
    df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': ['Texto', 'Texto', 'Texto']})
    # Si hay texto sin codificar, retorna False
    assert revisar_datos_finales(df) is False

def test_revisar_datos_finales_correcto():
    df = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
    # Si está limpio y numérico, retorna True
    assert revisar_datos_finales(df) is True

def test_revisar_datos_finales_raises_validation_error():
    with pytest.raises(DataValidationError):
        revisar_datos_finales(None)
