import pandas as pd
import numpy as np
from ds_guardian import auditoria

def run_audit_example():
    print("\n--- EJEMPLO: Auditoría y QA con DS Guardian ---")
    
    # 1. Crear dataset con múltiples fallos y riesgos de datos
    df_fallido = pd.DataFrame({
        'Feature_A': [1.0, 2.0, np.nan, 4.0],  # Error: Nulo
        'Feature_B': ['Texto', 'Texto2', 'Texto3', 'Texto4'],  # Error: Columna categórica de texto sin codificar
        'Feature_C': [100.0, 101.0, 99.0, 102.0],
        'Leakage_Col': [10.0, 20.0, 30.0, 40.0]  # Riesgo de Data Leakage (correlación total con target)
    })
    y = [1.0, 2.0, 3.0, 4.0]  # Target correlacionado al 100% con Leakage_Col y Feature_C
    
    # Ejecutar revisión
    print("Revisando dataset con fallos:")
    resultado_fallido = auditoria.revisar_datos_finales(df_fallido, y=y)
    print(f"¿Pasó la auditoría? -> {resultado_fallido}")
    
    # 2. Crear dataset correcto
    df_correcto = pd.DataFrame({
        'Feature_A': [1.0, 2.0, 3.0, 4.0],
        'Feature_C': [100.0, 101.0, 99.0, 102.0],
    })
    
    print("\nRevisando dataset correcto:")
    resultado_correcto = auditoria.revisar_datos_finales(df_correcto, y=[0, 1, 0, 1])
    print(f"¿Pasó la auditoría? -> {resultado_correcto}")
    
    # 3. Registrar modelo en el JSON local y comparar rendimiento
    print("\nRegistrando rendimiento inicial:")
    auditoria.registrar_y_comparar_modelo("Proyecto_Demo", "F1-Score", 0.82)
    
    print("\nRegistrando rendimiento mejorado:")
    auditoria.registrar_y_comparar_modelo("Proyecto_Demo", "F1-Score", 0.88)

if __name__ == "__main__":
    run_audit_example()
