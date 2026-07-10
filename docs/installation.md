# Guía de Instalación

Este documento detalla los requisitos del sistema y las diferentes opciones para instalar **DS Guardian** de forma portable en cualquier entorno.

## Requisitos de Entorno

* **Versión de Python**: Python 3.9 o superior.
* **Dependencias Core**:
  * `pandas` >= 1.5.0
  * `numpy` >= 1.20.0
  * `scikit-learn` >= 1.0.0
  * `seaborn` >= 0.12.0
  * `matplotlib` >= 3.5.0
  * `python-docx` >= 1.0.0

---

## 🚀 Quick Start (Inicio Rápido)

Para descargar, instalar el framework e iniciar la documentación local en un entorno limpio:

```bash
# 1. Clonar el repositorio
git clone https://github.com/holahola144/ds_guardian.git

# 2. Entrar a la carpeta del proyecto
cd ds_guardian

# 3. Instalar en modo editable
pip install -e .

# 4. Iniciar el servidor local de documentación
mkdocs serve
```

Luego, abre en tu navegador la dirección:
**`http://127.0.0.1:8000`**

---

## Opción 1: Instalación de Dependencias desde requirements.txt

Si prefieres no instalar el framework como paquete en el sistema, sino simplemente importar la subcarpeta `ds_guardian/` en tus propios scripts, puedes instalar únicamente sus dependencias:

```bash
pip install -r requirements.txt
```

> [!NOTE]
> Si utilizas **Conda** o un entorno virtual (`venv`, `poetry`, etc.), asegúrate de activarlo en tu terminal antes de ejecutar `pip install` o `mkdocs serve`.

---

## Verificación de la Instalación

Para verificar que la librería se ha instalado correctamente y no hay conflictos de importación, ejecuta:

```bash
python prueba_robusta.py
```
Si la consola imprime el mensaje de finalización exitosa y no arroja errores de importación, la instalación ha concluido con éxito.
