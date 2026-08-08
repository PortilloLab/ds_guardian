#!/bin/bash
# Script de inicio automatizado para Soluciones Informática JD

PROJECT_DIR="/home/jose/Development/PortilloLab/Soluciones_JD_1"

# Verificar si el servidor PHP ya está ejecutándose en el puerto 8000
if ! pgrep -f "php -S localhost:8000" > /dev/null; then
    cd "$PROJECT_DIR"
    php -S localhost:8000 > /dev/null 2>&1 &
    sleep 1
fi

# Abrir el navegador en la URL del portal
xdg-open "http://localhost:8000"
