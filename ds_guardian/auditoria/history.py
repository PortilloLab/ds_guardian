import json
import os
from datetime import datetime
from ..logging import get_logger, C_BOLD, C_BLUE, C_GREEN, C_RED, C_RESET

logger = get_logger(__name__)
HISTORIAL_FILE = 'historial_proyectos.json'

def registrar_y_comparar_modelo(
    nombre_proyecto: str, 
    metrica_principal_nombre: str, 
    valor_metrica: float, 
    maximizar: bool = True
) -> None:
    print(f"{C_BOLD}{C_BLUE}--- 🕵️ DS Guardian: Registro de Modelos ---{C_RESET}")
    historial = {}
    if os.path.exists(HISTORIAL_FILE):
        try:
            with open(HISTORIAL_FILE, 'r') as f:
                historial = json.load(f)
        except Exception:
            pass
            
    if nombre_proyecto in historial:
        ultimo_valor = historial[nombre_proyecto]['valor']
        print(f"Historial para '{nombre_proyecto}'. Último {metrica_principal_nombre}: {ultimo_valor:.4f}")
        diferencia = valor_metrica - ultimo_valor
        mejoro = (diferencia > 0) if maximizar else (diferencia < 0)
        if mejoro:
            print(f"{C_GREEN}{C_BOLD}🎉 ¡Rendimiento mejoró en {abs(diferencia):.4f}!{C_RESET}")
        elif diferencia == 0:
            print("↔️ Mismo rendimiento.")
        else:
            print(f"{C_RED}{C_BOLD}⚠️ Rendimiento cayó en {abs(diferencia):.4f}.{C_RESET}")
    else:
        print(f"Primer registro para '{nombre_proyecto}'. Métrica ({metrica_principal_nombre}): {valor_metrica:.4f}")
        
    historial[nombre_proyecto] = {
        'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrica': metrica_principal_nombre,
        'valor': valor_metrica
    }
    with open(HISTORIAL_FILE, 'w') as f:
        json.dump(historial, f, indent=4)
    print(f"{C_GREEN}✅ Historial actualizado en '{HISTORIAL_FILE}'.{C_RESET}")
