# Preguntas Frecuentes (FAQ)

### ¿Qué es el "Data Leakage" y cómo lo previene DS Guardian?
El *Data Leakage* o fuga de información es el error que ocurre cuando los datos de entrenamiento se contaminan con información del conjunto de prueba. Esto genera métricas de validación falsamente optimistas. DS Guardian lo previene dividiendo explícitamente el entrenamiento y la prueba en sus funciones de preprocesamiento, aplicando transformaciones (`fit_transform` en Train, `transform` en Test).

### ¿Por qué mi script se queda colgado cuando uso visualizaciones?
Si ejecutas tu script en entornos headless (servidores remotos, contenedores Docker o tareas en segundo plano en VS Code), la visualización interactiva de matplotlib (`plt.show()`) intentará abrir una ventana del entorno gráfico y quedará bloqueada. Para solucionarlo, pasa el parámetro `save_path` en las funciones de gráficos (ej: `save_path='plots/grafico.png'`) para guardarlo a disco de forma no interactiva.

### ¿Puedo utilizar DS Guardian con otros modelos además de scikit-learn?
Sí, el preprocesamiento genera DataFrames y arrays estándar de NumPy y Pandas, que son totalmente compatibles con librerías como XGBoost, LightGBM, CatBoost e incluso frameworks de Deep Learning como PyTorch o TensorFlow.

### ¿Dónde se guardan los historiales de los modelos?
Se registran localmente en el archivo `historial_proyectos.json` en el directorio de ejecución actual de tu script, permitiéndote comparar las métricas históricas de tu modelo entre diferentes ejecuciones.
