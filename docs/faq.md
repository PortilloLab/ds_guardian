# Preguntas Frecuentes (FAQ)

### ¿Qué es el "Data Leakage" y cómo lo previene DS Guardian?
El *Data Leakage* o fuga de información ocurre cuando los datos de entrenamiento se contaminan con información del conjunto de prueba. Esto genera métricas de validación falsamente optimistas. DS Guardian lo previene dividiendo explícitamente los conjuntos de entrenamiento y prueba en sus funciones de preprocesamiento y aplicando transformaciones de forma segura (`fit_transform` en Train y `transform` en Test).

### ¿Por qué mi script se queda colgado cuando uso visualizaciones?
Si ejecutas tu script en entornos headless (servidores remotos, contenedores Docker o tareas en segundo plano en VS Code), la visualización interactiva de matplotlib (`plt.show()`) intentará abrir una ventana del entorno gráfico y quedará bloqueada. Para evitarlo, usa `save_path` en las funciones de gráficos (por ejemplo, `save_path='plots/grafico.png'`) para guardar la imagen en disco sin abrir una ventana interactiva.

### ¿Puedo utilizar DS Guardian con otros modelos además de scikit-learn?
Sí. El preprocesamiento genera DataFrames y arrays estándar de NumPy y Pandas, compatibles con librerías como XGBoost, LightGBM, CatBoost e incluso frameworks de Deep Learning como PyTorch o TensorFlow.

### ¿Dónde se guardan los historiales de los modelos?
Los historiales se registran localmente en el archivo `historial_proyectos.json`, en el directorio de ejecución actual de tu script, para que puedas comparar métricas históricas entre ejecuciones.
