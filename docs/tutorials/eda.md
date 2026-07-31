# Tutorial: Análisis Exploratorio de Datos (EDA)

El módulo `ds_guardian.eda` proporciona herramientas útiles para comprender la salud de tus datos y optimizar el consumo de recursos de computación.

## 1. Optimización del Consumo de Memoria RAM

Cuando trabajas con DataFrames de gran escala, Pandas suele asignar tipos de datos excesivamente grandes por defecto (como `int64` o `float64`). El método `optimizar_memoria` analiza los rangos de valores numéricos de las columnas y realiza un downcast seguro (ej: a `int8` o `float32`), disminuyendo la memoria consumida hasta en un 70%.

```python
import pandas as pd
from ds_guardian import eda

# Cargar tu dataset
df = pd.read_csv('datos_gigantes.csv')

# Optimizar de forma automática
df_optimizado = eda.optimizar_memoria(df)
```

---

## 2. Detección y Tratamiento de Outliers (Winsorization)

La presencia de outliers (valores atípicos) puede distorsionar fuertemente las predicciones de los modelos basados en distancias (como regresión lineal o SVM).

### Detección con el Rango Intercuartílico (IQR):
```python
# Reporta el número de outliers identificados en el DataFrame por columna
eda.detectar_outliers_iqr(df)
```

### Winsorization (Capping sin Data Leakage):
En lugar de descartar filas valiosas eliminando outliers, limitamos (acotamos) los valores extremos reemplazándolos con los límites mínimo y máximo definidos por el Rango Intercuartílico.

Para evitar **Data Leakage**, los límites del IQR deben calcularse exclusivamente sobre el conjunto de entrenamiento (`train`) y aplicarse al de prueba (`test`):

```python
# Acota los valores extremos en train y test sin contaminar los límites de train
X_train_canned, X_test_capped = eda.acotar_outliers_iqr(
    df=X_train, 
    columnas=['Ingresos', 'Edad'], 
    df_test=X_test
)
```
