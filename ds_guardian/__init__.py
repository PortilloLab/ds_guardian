"""DS Guardian – AI Framework for Robust Data Science Projects."""

import builtins
import os

builtins.os = os

from .exceptions import (
    DSGuardianError,
    DataValidationError,
    DataLeakageError,
    ModelAuditingError,
    ImbalanceWarning,
)
from .eda import (
    resumir_datos,
    missing_values_table,
    optimizar_memoria,
    detectar_outliers_iqr,
    acotar_outliers_iqr,
)
from .limpieza import (
    imputar_nulos,
    tratar_duplicados,
    codificar_variables,
    escalar_caracteristicas,
)
from .preprocessing.cleaning import imputar_nulos as preprocessing_imputar_nulos
from .preprocessing.encoding import codificar_variables as preprocessing_codificar_variables
from .preprocessing.scaling import escalar_caracteristicas as preprocessing_escalar_caracteristicas
from .auditoria import (
    revisar_datos_finales,
    registrar_y_comparar_modelo,
    generar_reporte_auditoria_markdown,
    generar_reporte_auditoria_html,
)
from .auditoria.reporter import generar_reporte_html, generar_reporte_markdown
from .auditoria.detector_leakage import (
    detectar_fuga_temporal,
    detectar_fuga_target,
    detectar_multicolinealidad,
)
from .modelos import (
    evaluar_clasificacion,
    evaluar_regresion,
    validacion_cruzada,
    validacion_cruzada_temporal,
    optimizar_hiperparametros,
    graficar_importancia_caracteristicas,
    guardar_modelo_entrenado,
    cargar_modelo_entrenado,
    exportar_modelo,
)
from .sklearn.transformers import (
    SafeImputerTransformer,
    SafeOneHotTransformer,
    SafeScalerTransformer,
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
__author__ = "José Daniel Portillo"

__all__ = [
    "DSGuardianError",
    "DataValidationError",
    "DataLeakageError",
    "ModelAuditingError",
    "ImbalanceWarning",
    "imputar_nulos",
    "tratar_duplicados",
    "codificar_variables",
    "escalar_caracteristicas",
    "preprocessing_imputar_nulos",
    "preprocessing_codificar_variables",
    "preprocessing_escalar_caracteristicas",
    "resumir_datos",
    "missing_values_table",
    "optimizar_memoria",
    "detectar_outliers_iqr",
    "acotar_outliers_iqr",
    "revisar_datos_finales",
    "registrar_y_comparar_modelo",
    "generar_reporte_auditoria_markdown",
    "generar_reporte_auditoria_html",
    "generar_reporte_html",
    "generar_reporte_markdown",
    "detectar_fuga_temporal",
    "detectar_fuga_target",
    "detectar_multicolinealidad",
    "evaluar_clasificacion",
    "evaluar_regresion",
    "validacion_cruzada",
    "validacion_cruzada_temporal",
    "optimizar_hiperparametros",
    "graficar_importancia_caracteristicas",
    "guardar_modelo_entrenado",
    "cargar_modelo_entrenado",
    "exportar_modelo",
    "SafeImputerTransformer",
    "SafeOneHotTransformer",
    "SafeScalerTransformer",
    "configurar_estilo",
    "plot_distribucion",
    "plot_categorica",
    "plot_correlacion",
    "plot_importancia_caracteristicas",
    "plot_curva_roc",
    "__version__",
    "__author__",
]
