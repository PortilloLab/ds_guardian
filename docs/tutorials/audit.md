# Tutorial: Auditoría de Datos e Historial de Modelos

El módulo `ds_guardian.auditoria` funciona como un agente QA automatizado que valida la sanidad de tus datasets finales y guarda registros históricos de rendimiento.

## 1. Auditoría del Dataset Final

Antes de pasar un DataFrame de características al método `.fit()` del estimador de Machine Learning, es vital validar que esté limpio para evitar caídas catastróficas en tiempo de ejecución. 

`revisar_datos_finales` analiza de manera estática y dinámica el DataFrame y arroja:
* **ERRORES FATALES (Rojo)**: Si detecta variables de texto no codificadas, valores nulos remanentes o una correlación perfecta (>0.99) con la variable objetivo (**Data Leakage**). Retorna `False` (o dict con `'aprobado': False`).
* **ADVERTENCIAS (Amarillo)**: Si detecta multicolinealidad entre características numéricas (>0.95) o clases fuertemente desbalanceadas (>65% clase mayoritaria).
* **OK (Verde)**: Si las verificaciones básicas son exitosas. Retorna `True` (o dict con `'aprobado': True`).

```python
from ds_guardian import auditoria

# Ejecutar auditoría de sanidad obteniendo el detalle completo
detalle = auditoria.revisar_datos_finales(X_train_scaled, y=y_train, retornar_detalle=True)

if detalle['aprobado']:
    print("El dataset es seguro para el modelo.")
else:
    print("Corrige las inconsistencias señaladas en la consola.")
```

---

## 2. Generación de Reportes Ejecutivos en Markdown

Puedes generar un reporte completo en formato Markdown listo para presentar o guardar en la documentación de tu proyecto:

```python
# Generar reporte técnico Markdown
auditoria.generar_reporte_auditoria_markdown(
    nombre_proyecto="Clasificación de Clientes Churn",
    df_info={
        'filas': len(X_train),
        'columnas': X_train.shape[1],
        'nulos': X_train.isnull().sum().sum(),
        'memoria_mb': round(X_train.memory_usage().sum() / 1024 / 1024, 2)
    },
    resultado_auditoria=detalle['aprobado'],
    metricas_modelo={'Accuracy': 0.92, 'F1-Score': 0.89, 'ROC-AUC': 0.94},
    top_features=top_features_df,
    output_path="reports/reporte_auditoria_churn.md",
    detalle_auditoria=detalle
)
```

---

## 3. Historial Comparativo de Experimentos

Permite llevar un registro de las métricas obtenidas por tus modelos a lo largo del desarrollo. Si una nueva ejecución del modelo mejora o empeora los resultados históricos del proyecto, el framework te lo notificará:

```python
# Guarda la métrica del experimento en 'historial_proyectos.json'
auditoria.registrar_y_comparar_modelo(
    nombre_proyecto='Clasificacion_Clientes',
    metrica_principal_nombre='F1-Score',
    valor_metrica=0.86,
    maximizar=True       # True si un valor mayor es mejor (F1, Accuracy), False si es un error (MSE)
)
```
