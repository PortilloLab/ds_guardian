from typing import Optional, Any
import pandas as pd
from .detector_leakage import detectar_multicolinealidad, detectar_fuga_target
from ..exceptions import DataValidationError
from ..logging import get_logger, C_BOLD, C_BLUE, C_GREEN, C_RED, C_YELLOW, C_RESET

logger = get_logger(__name__)

def revisar_datos_finales(df: pd.DataFrame, y: Optional[Any] = None) -> bool:
    """Agente QA (DS Guardian): Audita la sanidad de los datos antes de entrenar."""
    if df is None or df.empty:
        raise DataValidationError("El DataFrame a auditar está vacío o es None.")
        
    print(f"
{C_BOLD}{C_BLUE}--- 🕵️ DS Guardian: Auditoría de Datos ---{C_RESET}")
    errores = 0
    advertencias = 0
    
    # 1. Nulos restantes (FATAL)
    nulos = df.isnull().sum().sum()
    if nulos > 0:
        print(f"{C_RED}❌ ERROR FATAL: Quedaron {nulos} valores nulos en el dataset.{C_RESET}")
        errores += 1
    else:
        print(f"{C_GREEN}✅ OK: No hay valores nulos.{C_RESET}")
        
    # 2. Variables categóricas no codificadas (FATAL)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        print(f"{C_RED}❌ ERROR FATAL: Hay {len(cat_cols)} columnas categóricas sin procesar: {list(cat_cols)[:3]}.{C_RESET}")
        errores += 1
    else:
        print(f"{C_GREEN}✅ OK: Todas las variables son numéricas.{C_RESET}")
        
    # 3. Multicolinealidad (ADVERTENCIA)
    high_corr = detectar_multicolinealidad(df, umbral=0.95)
    if high_corr:
        print(f"{C_YELLOW}⚠️ ADVERTENCIA: Multicolinealidad detectada en {len(high_corr)} variables (>0.95):{C_RESET}")
        for col, corr_with in high_corr:
            print(f"        - {col} altamente correlacionada con: {corr_with}")
        advertencias += 1
    else:
        print(f"{C_GREEN}✅ OK: No se detectó multicolinealidad severa.{C_RESET}")

    # 4. Target & Data Leakage (FATAL / WARN)
    if y is not None:
        y_series = pd.Series(y)
        if y_series.nunique() <= 10:
            counts = y_series.value_counts(normalize=True)
            if counts.max() > 0.65:
                print(f"{C_YELLOW}⚠️ ADVERTENCIA: Clase desbalanceada (mayoritaria: {counts.max()*100:.1f}%).{C_RESET}")
                advertencias += 1
            else:
                print(f"{C_GREEN}✅ OK: Distribución de clase objetivo balanceada.{C_RESET}")
                
        leaks = detectar_fuga_target(df, y, umbral=0.99)
        if leaks:
            for col, corr in leaks:
                print(f"{C_RED}❌ ERROR FATAL: Fuga de datos detectada en '{col}' (corr con target: {corr:.4f}).{C_RESET}")
                errores += 1

    if errores > 0:
        print(f"
{C_BOLD}{C_RED}⚠️ RESULTADO: Auditoría falló con {errores} error(es) fatal(es).{C_RESET}")
        return False
    elif advertencias > 0:
        print(f"
{C_BOLD}{C_YELLOW}🏆 RESULTADO: Datos aptos para modelar con {advertencias} advertencia(s).{C_RESET}")
        return True
    else:
        print(f"
{C_BOLD}{C_GREEN}🏆 RESULTADO: Auditoría exitosa. Datos limpios y seguros para producción.{C_RESET}")
        return True
