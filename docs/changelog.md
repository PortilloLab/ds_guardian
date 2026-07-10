# Historial de Cambios (Changelog)

Todos los cambios notables realizados en el proyecto **DS Guardian** serán documentados en este archivo.

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
