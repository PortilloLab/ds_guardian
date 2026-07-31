# Resumen del Proceso: Desarrollo e Implementación de DS Guardian

Este documento sirve como guía paso a paso y explicación técnica de todo el ciclo de vida del proyecto **DS Guardian**, desde su concepción de software hasta su despliegue continuo (CI/CD) en la nube. Está estructurado de forma didáctica para ser utilizado en entrevistas de trabajo, presentaciones o clases.

---

## 1. ¿Qué es DS Guardian?
**DS Guardian** es un framework en Python diseñado para añadir robustez y gobernanza a proyectos de Ciencia de Datos. Actúa como un "guardián" o intermediario que asegura la calidad de los datos, gestiona modelos de Machine Learning y genera visualizaciones automáticas, todo bajo un sistema estricto de auditoría y control de errores.

### Arquitectura del Paquete:
* **`limpieza`**: Módulo para la imputación de valores nulos, eliminación de duplicados y preprocesamiento automático.
* **`modelos`**: Wrapper estructurado para entrenar, evaluar e inspeccionar métricas de modelos (Clasificación/Regresión).
* **`visualizacion`**: Generación automática de gráficos clave (correlación, importancia de variables, curvas de aprendizaje).
* **`auditoria`**: Monitoreo y registro de ejecuciones de entrenamiento, guardando un historial en JSON.
* **`exceptions`**: Excepciones personalizadas para capturar errores de datos o entrenamiento de forma limpia.

---

## 2. Fase de Desarrollo y Estructura del Código
El proyecto sigue las mejores prácticas de empaquetado de software en Python:

```text
ds_guardian/
├── ds_guardian/            # Código fuente del paquete
│   ├── __init__.py
│   ├── limpieza.py
│   ├── modelos.py
│   ├── visualizacion.py
│   ├── auditoria.py
│   └── exceptions.py
├── docs/                   # Archivos fuente de documentación (Markdown)
│   ├── index.md
│   ├── getting-started.md
│   ├── tutorials/
│   └── ...
├── tests/                  # Pruebas unitarias
│   ├── test_limpieza.py
│   └── test_modelos.py
├── .github/
│   └── workflows/
│       └── docs.yml        # Configuración de CI/CD para la documentación
├── mkdocs.yml              # Configuración del sitio web de MkDocs
├── requirements.txt        # Dependencias del proyecto
└── setup.py                # Configuración de instalación del paquete
```

---

## 3. Pruebas Unitarias y Robustez (QA)
Para garantizar la confiabilidad antes de cualquier despliegue, implementamos pruebas automatizadas utilizando **`pytest`**:
* **Prueba de Limpieza**: Se verifica que los valores nulos sean correctamente tratados y que las excepciones personalizadas se lancen al recibir datos corruptos o no estructurados.
* **Prueba de Modelos**: Se valida que el framework lance errores si se intenta evaluar un modelo no entrenado.

---

## 4. Documentación Dinámica con MkDocs (Material)
Para la documentación del framework, se optó por **MkDocs** con el tema **Material**, que proporciona una interfaz premium, adaptativa (responsive) y con soporte para modo oscuro y claro de forma automática.

* **Configuración (`mkdocs.yml`)**: Define la estructura de navegación del sitio (API, Guías, Tutoriales) y activa extensiones avanzadas como diagramas **Mermaid** y bloques de código con resaltado de sintaxis.

---

## 5. Integración y Despliegue Continuo (CI/CD)
El objetivo final era lograr que **cualquier cambio en la documentación se publique automáticamente en internet** sin intervención manual.

### Paso 1: Configurar GitHub Actions (`.github/workflows/docs.yml`)
Creamos un flujo de trabajo que se ejecuta automáticamente cada vez que hacemos un `git push` a la rama `main`:
1. Levanta un servidor virtual con **Ubuntu**.
2. Instala **Python**.
3. Descarga las dependencias del proyecto.
4. Ejecuta `mkdocs gh-deploy` para compilar los archivos Markdown en HTML estático.
5. Sube los archivos HTML listos para producción a una rama dedicada llamada **`gh-pages`**.

### Paso 2: Configurar GitHub Pages en la Web
Una vez creada la rama `gh-pages` por el flujo de trabajo:
1. Accedimos a la configuración del repositorio en GitHub (`Settings` -> `Pages`).
2. Configuramos la fuente de despliegue (**Build and deployment**):
   * **Rama (Branch)**: Seleccionamos `gh-pages`.
   * **Directorio**: Seleccionamos `/ (root)` (raíz).
3. GitHub reconoció los archivos estáticos en el directorio raíz de la rama y activó el sitio.

---

## 6. Puntos Clave para Explicar en una Entrevista / Clase
* **Automatización total**: El desarrollador solo escribe documentación en Markdown dentro de la carpeta `/docs` y hace `git push`. La nube hace el resto.
* **Separación de ramas**: La rama `main` contiene el código fuente y desarrollo, mientras que `gh-pages` contiene únicamente el sitio web compilado en HTML/CSS, manteniendo el repositorio limpio y organizado.
* **Solución de problemas reales**: Durante el despliegue, detectamos que GitHub Pages retornaba un error 404 porque estaba buscando el sitio en una carpeta `/docs` dentro de la rama `gh-pages`. Al ajustar la raíz del despliegue al directorio `/ (root)`, el problema se solucionó inmediatamente.
