# Historial de Cambios (Changelog)

Todos los cambios notables realizados en el proyecto **DS Guardian** serán documentados en este archivo.

---

## [1.1.0] - 2026-07-31

### Añadido
* **Aislamiento Anti-Leakage en Winsorization**: Adición del parámetro `df_test` en `eda.acotar_outliers_iqr()` para calcular los límites del Rango Intercuartílico (IQR) exclusivamente sobre el conjunto de entrenamiento (`train`) y aplicarlos defensivamente en el conjunto de prueba (`test`).
* **Generación de Reportes Técnicos Markdown**: Adición de `generar_reporte_auditoria_markdown()` en `auditoria.py` para exportar informes ejecutivos completos de calidad de datos y evaluación de modelos.
* **Modo Detallado de Auditoría**: Soporte para `retornar_detalle=True` en `revisar_datos_finales()`, devolviendo un diccionario estructurado con el estado individual de cada regla QA.
* **Visualización Unificada de Importancia de Variables**: Consolidación de `modelos.graficar_importancia_caracteristicas()`, extendiendo el soporte para estimadores lineales (vía `coef_`) además de estimadores basados en árboles (`feature_importances_`).
* **Lanzador Interactivo de Escritorio**: Adición de `scripts/lanzar_ds_guardian.sh` y el acceso directo `DS_Guardian.desktop` con detección dinámica del entorno `.venv` y resolución automática de rutas del proyecto.
* **Manual de Usuario en PDF**: Compilación y generación automática de `DS_Guardian_User_Manual.pdf` localizado en el Escritorio.

### Modificado
* **Severidad de Data Leakage**: Corrección en la clasificación de severidad de Data Leakage en la función de auditoría QA, marcándolo como **ERROR FATAL (Rojo)**.
* **Ampliación de Pruebas Unitarias**: Extensión de la suite de pruebas `pytest` a 20 casos de prueba al 100% pasando sin fallos.
* **Compatibilidad con Pandas 3/4**: Eliminación de advertencias de deprecación mediante la inclusión de `string` en `select_dtypes`.
* **Referencia de API y Tutoriales**: Actualización completa de `docs/api.md`, `docs/tutorials/audit.md` y `docs/tutorials/eda.md`.

---

## [1.0.0] - 2026-07-10

### Añadido
* **Estructura de empaquetado**: Configuración PEP 517 para la librería mediante `pyproject.toml` y `requirements.txt`.
* **Excepciones personalizadas**: Lanzamiento de `DataValidationError` y `ModelAuditingError` ante fallos de integridad de datos.
* **Soporte Headless para Visualizaciones**: Adición del parámetro `save_path` en todas las funciones de graficado para salvar heatmaps y distribuciones directamente a disco sin colgar terminales remotas.
* **Documentación premium**: Estructuración del sitio MkDocs con diseño material, tema claro/oscuro adaptativo y soporte para diagramas Mermaid.
* **Módulo de pruebas**: Suite de pruebas unitarias implementada con `pytest` en la carpeta `tests/` logrando validar 14 casos clave de funcionamiento de la librería.
* **CI/CD Pipeline**: Configuración de GitHub Actions para automatizar validaciones en múltiples versiones de Python.
* **Ejemplos prácticos**: Adición de la carpeta `examples/` que incluye códigos ejecutables listos para clasificación, regresión, análisis exploratorio y auditoría.
* **Archivo de Licencia**: Inclusión del archivo formal de Licencia MIT (`LICENSE`).

### Modificado
* **Tratamiento de outliers**: Rediseño del algoritmo de outliers en `eda.py` aplicando técnicas seguras de *Winsorization/Capping* sobre las variables numéricas.
* **Escalamiento seguro**: Refactorización en `limpieza.py` para aislar el ajuste y evitar *Data Leakage*.
* **Rebranding**: Renombrado global de la biblioteca a **DS Guardian: AI Framework for Robust Data Science Projects**.
