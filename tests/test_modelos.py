import pytest
from ds_guardian.modelos import evaluar_clasificacion, evaluar_regresion
from ds_guardian.exceptions import ModelAuditingError

def test_evaluar_clasificacion_raises_length_mismatch():
    y_true = [1, 0, 1]
    y_pred = [1, 0]
    with pytest.raises(ModelAuditingError):
        evaluar_clasificacion(y_true, y_pred)

def test_evaluar_regresion_raises_length_mismatch():
    y_true = [1.0, 2.0]
    y_pred = [1.0, 2.0, 3.0]
    with pytest.raises(ModelAuditingError):
        evaluar_regresion(y_true, y_pred)
