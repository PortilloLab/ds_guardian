<<<<<<< HEAD
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
=======
"""DS Guardian – AI Framework for Robust Data Science Projects."""

from . import auditoria, eda, limpieza, modelos, visualizacion
from .auditoria import (
    revisar_datos_finales,
    registrar_y_comparar_modelo,
    generar_reporte_auditoria_markdown,
    generar_reporte_auditoria_html,
)
from .eda import (
    resumir_datos,
    missing_values_table,
    optimizar_memoria,
    detectar_outliers_iqr,
    acotar_outliers_iqr,
)
from .exceptions import DSGuardianError, DataValidationError, ModelAuditingError
from .limpieza import (
    imputar_nulos,
    tratar_duplicados,
    codificar_variables,
    escalar_caracteristicas,
)
from .modelos import (
    evaluar_clasificacion,
    evaluar_regresion,
    validacion_cruzada,
    optimizar_hiperparametros,
    graficar_importancia_caracteristicas,
    guardar_modelo_entrenado,
    cargar_modelo_entrenado,
)
from .visualizacion import (
    configurar_estilo,
    plot_distribucion,
    plot_categorica,
    plot_correlacion,
    plot_importancia_caracteristicas,
    plot_curva_roc,
)

__version__ = "1.0.0"

__all__ = [
    "auditoria",
    "eda",
    "limpieza",
    "modelos",
    "visualizacion",
    "DSGuardianError",
    "DataValidationError",
    "ModelAuditingError",
    "revisar_datos_finales",
    "registrar_y_comparar_modelo",
    "generar_reporte_auditoria_markdown",
    "generar_reporte_auditoria_html",
    "resumir_datos",
    "missing_values_table",
    "optimizar_memoria",
    "detectar_outliers_iqr",
    "acotar_outliers_iqr",
    "imputar_nulos",
    "tratar_duplicados",
    "codificar_variables",
    "escalar_caracteristicas",
    "evaluar_clasificacion",
    "evaluar_regresion",
    "validacion_cruzada",
    "optimizar_hiperparametros",
    "graficar_importancia_caracteristicas",
    "guardar_modelo_entrenado",
    "cargar_modelo_entrenado",
    "configurar_estilo",
    "plot_distribucion",
    "plot_categorica",
    "plot_correlacion",
    "plot_importancia_caracteristicas",
    "plot_curva_roc",
    "__version__",
>>>>>>> 3c0a8b2 (Release 1.0.0)
]
