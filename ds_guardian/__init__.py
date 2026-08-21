from .exceptions import DSGuardianError, DataValidationError, DataLeakageError
from .preprocessing.cleaning import imputar_nulos, tratar_duplicados, acotar_outliers_iqr
from .preprocessing.encoding import codificar_variables, codificar_target_encoder
from .preprocessing.scaling import escalar_caracteristicas
from .auditoria.quality import revisar_datos_finales
from .auditoria.history import registrar_y_comparar_modelo
from .auditoria.reporter import generar_reporte_html, generar_reporte_markdown
from .auditoria.detector_leakage import detectar_fuga_temporal, detectar_fuga_target, detectar_multicolinealidad
from .modelos import evaluar_clasificacion, validacion_cruzada_temporal, exportar_modelo
from .sklearn.transformers import SafeImputerTransformer, SafeOneHotTransformer, SafeScalerTransformer

__version__ = "1.0.0"
__author__ = "José Daniel Portillo"

__all__ = [
    "imputar_nulos",
    "tratar_duplicados",
    "acotar_outliers_iqr",
    "codificar_variables",
    "codificar_target_encoder",
    "escalar_caracteristicas",
    "revisar_datos_finales",
    "registrar_y_comparar_modelo",
    "generar_reporte_html",
    "generar_reporte_markdown",
    "detectar_fuga_temporal",
    "detectar_fuga_target",
    "detectar_multicolinealidad",
    "evaluar_clasificacion",
    "validacion_cruzada_temporal",
    "exportar_modelo",
    "SafeImputerTransformer",
    "SafeOneHotTransformer",
    "SafeScalerTransformer",
    "DataValidationError",
    "DataLeakageError"
]
