import json
import os
from datetime import datetime
from typing import Any, Optional, Dict
import pandas as pd
import numpy as np
from .exceptions import DataValidationError

HISTORIAL_FILE = 'historial_proyectos.json'

# Códigos de color ANSI para una salida premium por consola
C_GREEN = '\033[92m'
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_BLUE = '\033[94m'
C_BOLD = '\033[1m'
C_RESET = '\033[0m'

def revisar_datos_finales(df: pd.DataFrame, y: Optional[Any] = None) -> bool:
    """
    Agente QA (DS Guardian): Verifica que el dataset esté limpio y libre de riesgos
    antes de entrenar el modelo. Escanea nulos, tipos de datos, multicolinealidad,
    desbalance de clases y potencial fuga de información (Data Leakage).
    
    Args:
        df: DataFrame de pandas con las características.
        y: Serie o arreglo con el target (opcional).
        
    Returns:
        True si pasa la auditoría básica (sin errores fatales), False de lo contrario.
        
    Raises:
        DataValidationError: Si el DataFrame está vacío o es None.
    """
    if df is None or df.empty:
        raise DataValidationError("El DataFrame a auditar está vacío o es None.")
        
    print(f"\n{C_BOLD}{C_BLUE}--- 🕵️ DS Guardian: Auditoría de Datos ---{C_RESET}")
    errores = 0
    advertencias = 0
    
    # 1. Nulos sobrevivientes (FATAL)
    nulos = df.isnull().sum().sum()
    if nulos > 0:
        print(f"{C_RED}❌ ERROR FATAL: Quedaron {nulos} valores nulos en el dataset.{C_RESET}")
        errores += 1
    else:
        print(f"{C_GREEN}✅ OK: No hay valores nulos.{C_RESET}")
        
    # 2. Tipos de datos no numéricos (FATAL para la mayoría de modelos)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        print(f"{C_RED}❌ ERROR FATAL: Hay {len(cat_cols)} columnas categóricas sin procesar (ej. {list(cat_cols)[:3]}).{C_RESET}")
        errores += 1
    else:
        print(f"{C_GREEN}✅ OK: Todas las variables son numéricas.{C_RESET}")
        
    # 3. Detección de Multicolinealidad (ADVERTENCIA)
    df_num = df.select_dtypes(include=[np.number])
    if not df_num.empty:
        corr_matrix = df_num.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = []
        for col in upper.columns:
            correlated = upper.index[upper[col] > 0.95].tolist()
            if correlated:
                high_corr_pairs.append((col, correlated))
        
        if len(high_corr_pairs) > 0:
            print(f"{C_YELLOW}⚠️ ADVERTENCIA: Multicolinealidad detectada. Columnas altamente correlacionadas (>0.95):{C_RESET}")
            for col, corr_with in high_corr_pairs:
                print(f"        - {col} está altamente correlacionada con: {corr_with}")
            advertencias += 1
        else:
            print(f"{C_GREEN}✅ OK: No se detectó multicolinealidad severa.{C_RESET}")

    # 4. Desbalance de Clases (ADVERTENCIA) y Data Leakage (FATAL) si se pasa el target
    if y is not None:
        y_series = pd.Series(y)
        # Desbalance
        if len(y_series.unique()) == 2 or y_series.nunique() < 10:
            counts = y_series.value_counts(normalize=True)
            max_pct = counts.max()
            if max_pct > 0.65:
                print(f"{C_YELLOW}⚠️ ADVERTENCIA: Clase objetivo desbalanceada. La clase mayoritaria representa el {max_pct*100:.2f}% de las muestras.{C_RESET}")
                print("        Se recomienda usar estratificación (StratifiedKFold) o técnicas de balanceo (SMOTE, submuestreo, etc.).")
                advertencias += 1
            else:
                print(f"{C_GREEN}✅ OK: Distribución de clase objetivo balanceada.{C_RESET}")
        
        # Fuga de Información (Data Leakage)
        for col in df_num.columns:
            try:
                corr = np.abs(df_num[col].corr(y_series))
                if corr > 0.99:
                    print(f"{C_RED}❌ ERROR FATAL: La columna '{col}' tiene una correlación de {corr:.4f} con el target.{C_RESET}")
                    print("        Esto indica una fuga de datos (Data Leakage). ¡Elimina esta columna de las características!")
                    errores += 1
            except Exception:
                pass
                
    # Diagnóstico Final
    if errores > 0:
        print(f"\n{C_BOLD}{C_RED}⚠️ RESULTADO: La auditoría falló con {errores} error(es) fatal(es) y {advertencias} advertencia(s). Corrige los errores antes de continuar.{C_RESET}")
        return False
    elif advertencias > 0:
        print(f"\n{C_BOLD}{C_YELLOW}🏆 RESULTADO: Datos listos para modelar, pero con {advertencias} advertencia(s) a tener en cuenta.{C_RESET}")
        return True
    else:
        print(f"\n{C_BOLD}{C_GREEN}🏆 RESULTADO: Auditoría exitosa. Datos limpios y seguros para producción o modelado.{C_RESET}")
        return True

def registrar_y_comparar_modelo(
    nombre_proyecto: str, 
    metrica_principal_nombre: str, 
    valor_metrica: float, 
    maximizar: bool = True
) -> None:
    """
    Guarda la métrica de este proyecto en un JSON y la compara con la última registrada,
    generando alertas visuales si mejora o empeora.
    
    Args:
        nombre_proyecto: Identificador del proyecto.
        metrica_principal_nombre: Nombre de la métrica (ej: 'Accuracy', 'MSE').
        valor_metrica: Valor numérico obtenido.
        maximizar: Si la métrica es mejor cuando es más alta (True) o más baja (False).
    """
    print(f"\n{C_BOLD}{C_BLUE}--- 🕵️ DS Guardian: Auditoría de Modelos ---{C_RESET}")
    
    historial = {}
    if os.path.exists(HISTORIAL_FILE):
        try:
            with open(HISTORIAL_FILE, 'r') as f:
                historial = json.load(f)
        except Exception:
            pass
            
    # Comparar si ya existe el proyecto
    if nombre_proyecto in historial:
        ultimo_valor = historial[nombre_proyecto]['valor']
        print(f"Historial encontrado para '{nombre_proyecto}'. Último valor de {metrica_principal_nombre}: {ultimo_valor:.4f}")
        
        diferencia = valor_metrica - ultimo_valor
        
        if maximizar:
            mejoro = diferencia > 0
        else:
            mejoro = diferencia < 0
            
        if mejoro:
            print(f"{C_GREEN}{C_BOLD}🎉 ¡Felicidades! Mejoraste la métrica en {abs(diferencia):.4f} respecto al intento anterior.{C_RESET}")
        elif diferencia == 0:
            print("↔️ El modelo mantiene exactamente el mismo rendimiento.")
        else:
            print(f"{C_RED}{C_BOLD}⚠️ ATENCIÓN: El rendimiento empeoró en {abs(diferencia):.4f} respecto al último intento.{C_RESET}")
    else:
        print(f"Primer registro para el proyecto '{nombre_proyecto}'.")
        print(f"Métrica inicial ({metrica_principal_nombre}): {valor_metrica:.4f}")
        
    # Guardar nuevo valor
    historial[nombre_proyecto] = {
        'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrica': metrica_principal_nombre,
        'valor': valor_metrica
    }
    
    with open(HISTORIAL_FILE, 'w') as f:
        json.dump(historial, f, indent=4)
    print(f"{C_GREEN}✅ Historial actualizado con éxito.{C_RESET}")
