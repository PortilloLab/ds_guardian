# Hoja de Ruta (Roadmap)

Este documento describe el plan a futuro y las nuevas características planeadas para **DS Guardian**.

## Próximas Características (Q3 2026)

* **Integración con MLflow y Weights & Biases**: Permitir el envío automático de las métricas registradas a servidores de seguimiento remotos para visualización y comparación avanzada.
* **Detección de Data Drift (Deriva de datos)**: Implementar métodos estadísticos (como test de Kolmogorov-Smirnov) para detectar si las distribuciones de los datos de entrada en producción han variado respecto al entrenamiento.
* **Soporte para Series Temporales**: Funciones específicas de preprocesamiento de desfase (lags), medias móviles e imputaciones seguras en el dominio temporal sin leakage retrospectivo.

## Características de Producción (Q4 2026)

* **Autogeneración de Reportes PDF**: Exportación automatizada de reportes visuales en PDF con la auditoría del dataset y el desempeño final del modelo.
* **Integración CI/CD robusta**: Plantillas listas para usar de GitHub Actions y GitLab CI/CD para automatizar la ejecución de pruebas y validaciones previas a la entrega.
