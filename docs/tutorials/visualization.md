# Tutorial: Visualización Premium

El módulo `ds_guardian.visualizacion` permite generar reportes gráficos limpios y modernos, ideales para presentaciones técnicas o anexos de ingeniería.

## 1. Configuración Global de Estilos

El framework soporta temas visuales claro (`light`) y oscuro (`dark`) de forma automática:

```python
from ds_guardian import visualizacion

# Configurar el estilo visual por defecto del notebook o script
visualizacion.configurar_estilo(theme='dark')
```

---

## 2. Generación de Gráficos No Bloqueantes (Headless)

Si estás ejecutando el pipeline en un servidor remoto o un script automatizado en segundo plano, llamar a gráficos interactivos detendrá la ejecución del programa. Todas las funciones de visualización de **DS Guardian** admiten el argumento `save_path` para evitar esto:

```python
# Grafica y almacena directamente a disco sin abrir ventanas interactivas
visualizacion.plot_distribucion(df, columna='Edad', save_path='plots/edad_dist.png')

# Heatmap de correlación lineal
visualizacion.plot_correlacion(df, save_path='plots/matriz_correlacion.png')
```

---

## 3. Importancia de Variables en Modelos

Permite graficar de manera ordenada y estilizada qué variables del dataset tuvieron mayor peso en la toma de decisión del algoritmo entrenado:

```python
# Grafica las características más relevantes para el estimador
visualizacion.plot_importancia_caracteristicas(
    modelo_entrenado, 
    feature_names=list(X_train.columns), 
    max_features=10, 
    save_path='plots/feature_importances.png'
)
```
