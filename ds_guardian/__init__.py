from .exceptions import DSGuardianError, DataValidationError, DataLeakageError
from .preprocessing.cleaning import imputar_nulos, tratar_duplicados, acotar_outliers_iqr
from .preprocessing.encoding import codificar_variables
from .preprocessing.scaling import escalar_caracteristicas
from .auditoria.quality import revisar_datos_finales
from .auditoria.history import registrar_y_comparar_modelo
from .sklearn.transformers import SafeImputerTransformer, SafeScalerTransformer

__version__ = "1.0.0"
__author__ = "José Daniel Portillo"

__all__ = [
    "imputar_nulos",
    "tratar_duplicados",
    "acotar_outliers_iqr",
    "codificar_variables",
    "escalar_caracteristicas",
    "revisar_datos_finales",
    "registrar_y_comparar_modelo",
    "SafeImputerTransformer",
    "SafeScalerTransformer",
    "DataValidationError",
    "DataLeakageError"
]
