# Arquitectura de DS Guardian

Este documento describe la estructura interna de **DS Guardian** y cómo interactúan sus diferentes módulos.

## Estructura de Módulos

El framework sigue una arquitectura desacoplada donde cada submódulo tiene una única responsabilidad bien definida:

```
ds_guardian/
├── __init__.py           # Expone la API pública del framework
├── eda.py                # Módulo de exploración y tratamiento de outliers (Winsorization)
├── limpieza.py           # Módulo de imputación, codificación y escalado libre de leakage
├── visualizacion.py      # Estilos gráficos y plots compatibles con headless
├── modelos.py            # Métricas, validación cruzada y búsqueda de hiperparámetros
├── auditoria.py          # Agente QA y registro histórico de experimentos
└── exceptions.py         # Excepciones personalizadas (DataValidationError, etc.)
```

## Flujo de Datos y Control de Data Leakage

El mayor error de diseño en Ciencia de Datos es el **Data Leakage**. Ocurre cuando información del conjunto de prueba (Test) se filtra hacia el conjunto de entrenamiento (Train). Por ejemplo, al escalar un dataset completo aplicando `StandardScaler().fit_transform(df)`, el promedio y la desviación estándar calculados incluyen filas de Test, lo que distorsiona las métricas de validación del modelo.

Para evitar esto, `ds_guardian.limpieza` implementa un patrón donde toda transformación de dos conjuntos se realiza de la siguiente manera:

```python
# Patrón interno seguro en ds_guardian
scaler = StandardScaler()
df_train_scaled = scaler.fit_transform(df_train)  # Solo aprende de Train
df_test_scaled = scaler.transform(df_test)        # Transforma Test usando lo aprendido
```

## El Agente de Auditoría (auditoria.py)

El módulo de auditoría actúa como una barrera de seguridad (Gateway). Analiza de forma estática y dinámica el DataFrame final antes de que se pase a la fase de modelado de `scikit-learn`:

* **Nulos Residuales**: Levanta un error fatal si alguna columna tiene valores nulos sobrevivientes.
* **Textos no codificados**: Levanta un error si existen variables de tipo `object` o `category` que romperían el estimador.
* **Fuga de datos (Leakage)**: Evalúa la correlación de cada columna contra la variable objetivo. Si es $>0.99$, avisa de inmediato.
* **Desbalance de Clases**: Emite advertencias si la clase minoritaria está severamente desbalanceada.
