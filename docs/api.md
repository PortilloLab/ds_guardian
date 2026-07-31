# Referencia de la API (API Reference)

Este documento contiene la firma y descripción de las principales funciones expuestas por **DS Guardian**.

---

## Módulo: `ds_guardian.eda`

### `resumir_datos(df)`
Imprime dimensiones, tipos de datos y estadísticas descriptivas de forma segura controlando datasets vacíos.

### `missing_values_table(df)`
Calcula y formatea una tabla con el recuento y porcentaje de nulos por columna.

### `optimizar_memoria(df)`
Optimiza los tipos de datos de las columnas numéricas (ej. downcast a `int8` o `float32`) para reducir el consumo en RAM.

### `detectar_outliers_iqr(df)`
Detecta y reporta outliers en variables numéricas usando el Rango Intercuartílico (IQR).

### `acotar_outliers_iqr(df, columnas=None, factor=1.5, df_test=None)`
Aplica Winsorization acotando valores extremos dentro de los límites del IQR sin eliminar filas. Si se pasa `df_test`, los límites del IQR se calculan únicamente con `df` (train) y se aplican defensivamente sobre `df_test` evitando data leakage.

---

## Módulo: `ds_guardian.limpieza`

### `imputar_nulos(df_train, df_test=None, estrategia_num='median', estrategia_cat='most_frequent')`
Imputa nulos en características numéricas y categóricas evitando Data Leakage.

### `tratar_duplicados(df)`
Busca y remueve filas duplicadas en el DataFrame.

### `codificar_variables(df_train, df_test=None)`
Aplica One-Hot Encoding a variables categóricas alineando el número de columnas resultantes entre Train y Test.

### `escalar_caracteristicas(df_train, df_test=None, metodo='standard', columnas=None)`
Escala variables numéricas usando `StandardScaler` o `MinMaxScaler` aislando correctamente el conjunto de prueba.

---

## Módulo: `ds_guardian.visualizacion`

### `configurar_estilo(theme='light')`
Configura el estilo visual global de seaborn y matplotlib. Soporta temas `light` (claro) y `dark` (oscuro).

### `plot_distribucion(df, columna, save_path=None)`
Grafica histograma y curva KDE de una variable. Si se pasa `save_path`, la imagen se guarda en disco y no bloquea el proceso.

### `plot_categorica(df, columna, max_categorias=20, save_path=None)`
Gráfico de barras horizontales para variables categóricas.

### `plot_correlacion(df, save_path=None)`
Heatmap de correlación lineal sobre variables numéricas.

### `plot_importancia_caracteristicas(modelo, feature_names, max_features=15, save_path=None)`
Grafica el orden de importancia de variables del estimador (soporta modelos basados en árboles y modelos lineales con coeficientes).

---

## Módulo: `ds_guardian.modelos`

### `evaluar_clasificacion(y_true, y_pred, y_prob=None, save_path=None)`
Reporte detallado de clasificación, Accuracy, ROC-AUC y heatmap de la matriz de confusión.

### `evaluar_regresion(y_true, y_pred)`
Calcula e imprime MSE, RMSE, MAE y R2 Score.

### `graficar_importancia_caracteristicas(modelo, feature_names, max_features=15, save_path=None)`
Extrae y visualiza la importancia relativa de las características del modelo.

### `validacion_cruzada(modelo, X, y, cv=5, scoring='accuracy')`
Prueba el modelo mediante K-Fold Cross Validation informando medias y desviaciones estándar.

### `optimizar_hiperparametros(modelo, param_grid, X, y, cv=3, n_iter=10, scoring='accuracy')`
Optimización aleatoria de hiperparámetros con RandomizedSearchCV.

---

## Módulo: `ds_guardian.auditoria`

### `revisar_datos_finales(df, y=None, retornar_detalle=False)`
Agente QA que ejecuta un checklist de calidad en el dataset e informa advertencias/errores con colores en la consola. Si `retornar_detalle=True`, devuelve un diccionario completo con el estado de cada validación individual.

### `generar_reporte_auditoria_markdown(nombre_proyecto, df_info, resultado_auditoria, metricas_modelo, top_features=None, output_path="reporte_auditoria.md", detalle_auditoria=None)`
Genera un informe técnico completo en formato Markdown (HTML/GitHub compatible) con los resultados de la auditoría QA, calidad de datos y evaluación del modelo.

### `registrar_y_comparar_modelo(nombre_proyecto, metrica_principal_nombre, valor_metrica, maximizar=True)`
Guarda el desempeño del modelo en `historial_proyectos.json` y alerta si mejoró o empeoró respecto a ejecuciones previas.
