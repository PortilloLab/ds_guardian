# Tutorial: Evaluación y Optimización de Modelos

El módulo `ds_guardian.modelos` ofrece envoltorios optimizados para medir la calidad real de tus predictores y buscar hiperparámetros.

## 1. Ajuste de Hiperparámetros por Validación Cruzada

Busca la combinación óptima de parámetros para tu estimador evitando el sobreajuste (overfitting):

```python
from ds_guardian import modelos
from sklearn.ensemble import RandomForestClassifier

# Definir la grilla de búsqueda
grilla_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Ejecutar optimización aleatoria estructurada
mejor_modelo = modelos.optimizar_hiperparametros(
    RandomForestClassifier(random_state=42),
    param_grid=grilla_params,
    X=X_train_scaled,
    y=y_train,
    cv=5,               # Pliegues de validación cruzada
    n_iter=10,          # Intentos de combinaciones
    scoring='f1'
)
```

---

## 2. Evaluación Detallada de Predicciones

### Evaluación de Clasificación:
Genera un reporte consolidando Precision, Recall, F1-Score y Accuracy, guardando además una matriz de confusión interactiva:

```python
# Evaluar y guardar matriz de confusión
modelos.evaluar_clasificacion(
    y_true=y_test, 
    y_pred=y_pred, 
    y_prob=y_prob, 
    save_path='plots/matriz_confusion.png'
)
```

### Evaluación de Regresión:
Calcula e imprime métricas estándar de error absoluto, cuadrático y coeficiente de determinación:

```python
# Evaluar regresores continuos
modelos.evaluar_regresion(y_true=y_test, y_pred=y_pred)
```
