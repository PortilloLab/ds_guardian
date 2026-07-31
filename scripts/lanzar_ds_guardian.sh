#!/usr/bin/env bash

# Resolve project directory dynamically
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXEC="/home/jose/anaconda3/bin/python"

if [ ! -f "$PYTHON_EXEC" ]; then
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
