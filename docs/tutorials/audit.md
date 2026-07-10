# Tutorial: Auditoría de Datos e Historial de Modelos

El módulo `ds_guardian.auditoria` funciona como un agente QA automatizado que valida la sanidad de tus datasets finales y guarda registros históricos de rendimiento.

## 1. Auditoría del Dataset Final

Antes de pasar un DataFrame de características al método `.fit()` del estimador de Machine Learning, es vital validar que esté limpio para evitar caídas catastróficas en tiempo de ejecución. 

`revisar_datos_finales` analiza de manera estática y dinámica el DataFrame y arroja:
* **ERRORES FATALES (Rojo)**: Si detecta variables de texto no codificadas o valores nulos remanentes. Retorna `False`.
* **ADVERTENCIAS (Amarillo)**: Si detecta correlaciones sospechosamente perfectas con el target (Data Leakage) o clases fuertemente desbalanceadas.
* **OK (Verde)**: Si las verificaciones básicas son exitosas. Retorna `True`.

```python
from ds_guardian import auditoria

# Ejecutar auditoría de sanidad
pasa_la_auditoria = auditoria.revisar_datos_finales(X_train_scaled, y=y_train)

if pasa_la_auditoria:
    print("El dataset es seguro para el modelo.")
else:
    print("Corrige las inconsistencias señaladas en la consola.")
```

---

## 2. Historial Comparativo de Experimentos

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
