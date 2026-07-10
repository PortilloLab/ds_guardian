# Tutorial: Limpieza Segura (Cleaning)

El módulo `ds_guardian.limpieza` se encarga de preparar los datos numéricos y categóricos para que el estimador no falle, asegurando un aislamiento total entre entrenamiento y validación.

## 1. Imputación de Valores Faltantes (Nulos)

Para evitar la fuga de datos (Data Leakage), calculamos las estadísticas de imputación (como la mediana o la moda) **únicamente** sobre los datos de entrenamiento y aplicamos esos mismos valores fijos a los conjuntos de prueba:

```python
from ds_guardian import limpieza

# Imputación segura sin fugas
X_train_clean, X_test_clean = limpieza.imputar_nulos(
    X_train, 
    X_test, 
    estrategia_num='median',        # 'mean' o 'median'
    estrategia_cat='most_frequent'  # 'most_frequent'
)
```

---

## 2. Codificación Categórica Alineada

Si aplicamos One-Hot Encoding por separado sobre el conjunto de entrenamiento y de validación, es altamente probable que el número y orden de columnas resultantes difieran (debido a categorías raras presentes solo en un subconjunto). 

`codificar_variables` previene esto de forma interna, alineando automáticamente las dimensiones finales:

```python
# Codifica categóricas y asegura simetría de columnas en Train y Test
X_train_enc, X_test_enc = limpieza.codificar_variables(X_train_clean, X_test_clean)
```

---

## 3. Escalamiento Seguro de Características

Escalamos variables numéricas preservando las estadísticas descriptivas estrictamente del conjunto de entrenamiento:

```python
# Escalar a media 0 y varianza 1 (Standard) o rango [0,1] (MinMax)
X_train_scaled, X_test_scaled = limpieza.escalar_caracteristicas(
    X_train_enc, 
    X_test_enc, 
    metodo='standard'
)
```
