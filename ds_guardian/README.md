# DS Guardian – AI Framework for Robust Data Science Projects

`ds_guardian` es un framework diseñado para blindar tus proyectos de Ciencia de Datos y Machine Learning. Proporciona herramientas avanzadas y seguras para el análisis exploratorio (EDA), la limpieza y preparación de datos, la visualización premium, el modelado y la auditoría automática (QA) para evitar fallas comunes en entornos de producción (Data Leakage, outliers, desbalances y nulos).

## Módulos del Framework

| Módulo | Funciones Clave | Propósito |
|--------|-----------------|-----------|
| `eda.py` | `resumir_datos`, `missing_values_table`, `optimizar_memoria`, `detectar_outliers_iqr`, `acotar_outliers_iqr` | Exploración de datos y tratamiento de valores atípicos (Capping/Winsorization). |
| `limpieza.py` | `imputar_nulos`, `tratar_duplicados`, `codificar_variables`, `escalar_caracteristicas` | Preprocesamiento de datos e imputación/escalamiento seguro (libre de *Data Leakage*). |
| `visualizacion.py` | `configurar_estilo` (soporta temas `light` y `dark`), `plot_distribucion`, `plot_categorica`, `plot_correlacion`, `plot_importancia_caracteristicas` | Visualizaciones premium y reporte visual de variables clave. |
| `modelos.py` | `evaluar_clasificacion` (Matriz y ROC-AUC), `evaluar_regresion` (MSE, RMSE, MAE, R2), `validacion_cruzada`, `optimizar_hiperparametros` | Entrenamiento, validación cruzada y búsqueda automática de hiperparámetros (Random Search). |
| `auditoria.py` | `revisar_datos_finales`, `registrar_y_comparar_modelo` | Agente QA inteligente con colores en consola que audita la sanidad de tus datasets (leakage, nulos, desbalances) y compara el historial del rendimiento de tus modelos. |

---

## Ejemplo Rápido de Uso

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ds_guardian import eda, limpieza, visualizacion, modelos, auditoria

# 1. Cargar y explorar
df = pd.read_csv('dataset.csv')
df = eda.optimizar_memoria(df)
df = eda.acotar_outliers_iqr(df, columnas=['columna_num'])

# 2. Dividir para evitar Data Leakage
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Limpieza y preparación segura
X_train_imp, X_test_imp = limpieza.imputar_nulos(X_train, X_test)
X_train_enc, X_test_enc = limpieza.codificar_variables(X_train_imp, X_test_imp)
X_train_scaled, X_test_scaled = limpieza.escalar_caracteristicas(X_train_enc, X_test_enc, metodo='standard')

# 4. Auditoría QA
if auditoria.revisar_datos_finales(X_train_scaled, y=y_train):
    
    # 5. Optimización e Hiperparámetros
    modelo_base = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, None]}
    mejor_modelo = modelos.optimizar_hiperparametros(modelo_base, param_grid, X_train_scaled, y_train, cv=3)
    
    # 6. Entrenamiento y predicción
    mejor_modelo.fit(X_train_scaled, y_train)
    preds = mejor_modelo.predict(X_test_scaled)
    probs = mejor_modelo.predict_proba(X_test_scaled)
    
    # 7. Evaluación y Comparación con Historial
    modelos.evaluar_clasificacion(y_test, preds, probs)
    accuracy = (preds == y_test).mean()
    auditoria.registrar_y_comparar_modelo('Mi_Proyecto', 'Accuracy', accuracy)
```

---

## Instalación y Configuración

1. Coloca la carpeta `ds_guardian` en el directorio raíz de tu proyecto.
2. Asegúrate de tener instaladas las librerías necesarias:
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib python-docx
   ```
3. Importa el framework normalmente en tus scripts de Python.

---
*Este framework está diseñado para garantizar que los modelos se entrenen con datos de máxima calidad, protegiendo tus flujos contra errores de desbordamiento, sesgos de desbalanceo e inconsistencias entre train y test.*
