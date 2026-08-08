#!/usr/bin/env bash

# Resolve project directory dynamically
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    PROJECT_DIR="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../pyproject.toml" ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    PROJECT_DIR="$(pwd)"
fi

if [ -f "$PROJECT_DIR/.venv/bin/python3" ]; then
    PYTHON_EXEC="$PROJECT_DIR/.venv/bin/python3"
elif [ -f "$PROJECT_DIR/venv/bin/python3" ]; then
    PYTHON_EXEC="$PROJECT_DIR/venv/bin/python3"
else
    PYTHON_EXEC="python3"
fi

cd "$PROJECT_DIR" || exit 1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

echo "============================================================"
echo "         🛡️ DS GUARDIAN - AI DATA GOVERNANCE SYSTEM"
echo "============================================================"
echo " [1] Auditar cualquier Dataset (Cargar desde Descargas / Ruta)"
echo " [2] Generar Reporte de Auditoría (Markdown)"
echo " [3] Ejecutar Prueba de Estrés e Integridad (Kaggle Datasets)"
echo " [4] Ejecutar Suite de Tests Unitarios (Pytest)"
echo " [5] Exit / Salir"
echo "============================================================"
read -rp "Select an option / Seleccione una opción [1-5]: " choice

case "$choice" in
    1)
        echo -e "\n[+] Iniciando módulo de auditoría de datasets personalizados..."
        "$PYTHON_EXEC" "$PROJECT_DIR/scripts/auditar_custom_dataset.py"
        ;;
    2)
        echo -e "\n[+] Running Data Quality & Audit Generator..."
        "$PYTHON_EXEC" "$PROJECT_DIR/scripts/generate_report.py"
        ;;
    3)
        echo -e "\n[+] Running Kaggle Faults & Churn Audit..."
        "$PYTHON_EXEC" "$PROJECT_DIR/scripts/prueba_kaggle_churn.py"
        ;;
    4)
        echo -e "\n[+] Running Pytest Unit Test Suite..."
        "$PYTHON_EXEC" -m pytest "$PROJECT_DIR/tests/" -v
        ;;
    5)
        echo "Exiting DS Guardian. Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid option."
        ;;
esac

echo ""
read -rp "Press Enter to finish..."
