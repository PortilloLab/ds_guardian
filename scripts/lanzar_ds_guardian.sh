#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    PROJECT_DIR="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../pyproject.toml" ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [ -d "/home/jose/ejer_de_phyton" ]; then
    PROJECT_DIR="/home/jose/ejer_de_phyton"
else
    PROJECT_DIR="$(pwd)"
fi

if [ -f "$PROJECT_DIR/.venv/bin/python3" ]; then
    PYTHON_EXEC="$PROJECT_DIR/.venv/bin/python3"
elif [ -f "$PROJECT_DIR/venv/bin/python3" ]; then
    PYTHON_EXEC="$PROJECT_DIR/venv/bin/python3"
elif [ -f "/home/jose/anaconda3/bin/python" ]; then
    PYTHON_EXEC="/home/jose/anaconda3/bin/python"
else
    PYTHON_EXEC="python3"
fi

cd "$PROJECT_DIR" || exit 1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

echo "============================================================"
echo "         🛡️ DS GUARDIAN - AI DATA GOVERNANCE SYSTEM"
echo "============================================================"
echo " [1] Run Data Quality & Audit Report (Generar Reporte Markdown)"
echo " [2] Run Robust Stress Test (Prueba Robusta de Integridad)"
echo " [3] Run Kaggle Steel Faults Audit Test"
echo " [4] Run Full Unit Test Suite (Pytest)"
echo " [5] Exit / Salir"
echo "============================================================"
read -rp "Select an option / Seleccione una opción [1-5]: " choice

case "$choice" in
    1)
        echo -e "\n[+] Running Data Quality & Audit Generator..."
        "$PYTHON_EXEC" "$PROJECT_DIR/scripts/generate_report.py"
        ;;
    2)
        echo -e "\n[+] Running Robustness & Stress Test..."
        "$PYTHON_EXEC" "$PROJECT_DIR/scripts/prueba_robusta.py"
        ;;
    3)
        echo -e "\n[+] Running Kaggle Steel Faults Audit..."
        "$PYTHON_EXEC" "$PROJECT_DIR/scripts/prueba_kaggle_faults.py"
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
