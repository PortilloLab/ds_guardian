# Guía de Inicio Rápido

Esta guía te ayudará a ejecutar tu primer flujo de trabajo auditado de Machine Learning con **DS Guardian** en pocos minutos.

## 1. Instalación Rápida

Puedes instalar el framework de manera local y editable navegando al directorio del repositorio y ejecutando:

```bash
pip install -e .
```

O instalando las dependencias directamente:

```bash
pip install -r requirements.txt
```

*Para más detalles sobre la instalación y el entorno, consulta la página de [Instalación](installation.md).*

## 2. Creación del script `ejemplo.py`

Aquí tienes un pipeline completo que muestra cómo preprocesar de forma segura, auditar los datos y entrenar un clasificador RandomForest:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ds_guardian import eda, limpieza, modelos, auditoria

# 1. Carga de datos de ejemplo (con nulos y outliers)
np.random.seed(42)
df = pd.DataFrame({
    'Edad': [25, 30, np.nan, 45, 50, 55, 60, 65, 70, 75],
    'Salario': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 600000, -100000],
    'Target': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
})

# 2. Exploración y Capping de Outliers
df = eda.acotar_outliers_iqr(df, columnas=['Salario'])

# 3. División Train/Test antes de limpiar para evitar Data Leakage
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Preprocesamiento aislado y seguro
X_train, X_test = limpieza.imputar_nulos(X_train, X_test)
X_train, X_test = limpieza.escalar_caracteristicas(X_train, X_test, metodo='standard')

# 5. Auditoría QA
if auditoria.revisar_datos_finales(X_train, y=y_train):
    # Entrenar modelo
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    
    # Predecir y evaluar
    y_pred = modelo.predict(X_test)
    modelos.evaluar_clasificacion(y_test, y_pred, save_path='plots/confusion_matrix.png')
```

## 3. Ejecutar y Observar

Corre el script en tu terminal:

```bash
python ejemplo.py
```

* Verás en la consola la salida coloreada del módulo de auditoría de datos.
* Se generará el gráfico de la matriz de confusión en `plots/confusion_matrix.png`.
