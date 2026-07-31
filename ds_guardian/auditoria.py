import json
import os
from datetime import datetime
from typing import Any, Optional, Dict, Union
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

def revisar_datos_finales(
    df: pd.DataFrame,
    y: Optional[Any] = None,
    retornar_detalle: bool = False,
) -> Union[bool, Dict[str, Any]]:
    """
    Agente QA (DS Guardian): Verifica que el dataset esté limpio y libre de riesgos
    antes de entrenar el modelo. Escanea nulos, tipos de datos, multicolinealidad,
    desbalance de clases y potencial fuga de información (Data Leakage).
    
    Args:
        df: DataFrame de pandas con las características.
        y: Serie o arreglo con el target (opcional).
        retornar_detalle: Si es True, retorna un diccionario con el resultado de
            cada chequeo individual (sin_nulos, tipos_ok, sin_leakage, etc.),
            además del veredicto general en la clave 'aprobado'. Útil para pasarle
            datos reales a `generar_reporte_auditoria_markdown` en vez de que
            asuma que todo está en orden.
        
    Returns:
        Si retornar_detalle es False (default): True si pasa la auditoría básica
        (sin errores fatales), False en caso contrario.
        Si retornar_detalle es True: diccionario con el detalle de cada chequeo
        y la clave 'aprobado' con el veredicto general.
        
    Raises:
        DataValidationError: Si el DataFrame está vacío o es None.
    """
    if df is None or df.empty:
        raise DataValidationError("El DataFrame a auditar está vacío o es None.")
        
    print(f"\n{C_BOLD}{C_BLUE}--- 🕵️ DS Guardian: Auditoría de Datos ---{C_RESET}")
    errores = 0
    advertencias = 0
    detalle: Dict[str, Any] = {}
    
    # 1. Nulos sobrevivientes (FATAL)
    nulos = df.isnull().sum().sum()
    if nulos > 0:
        print(f"{C_RED}❌ ERROR FATAL: Quedaron {nulos} valores nulos en el dataset.{C_RESET}")
        errores += 1
        detalle['sin_nulos'] = False
    else:
        print(f"{C_GREEN}✅ OK: No hay valores nulos.{C_RESET}")
        detalle['sin_nulos'] = True
        
    # 2. Tipos de datos no numéricos (FATAL para la mayoría de modelos)
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
    if len(cat_cols) > 0:
        print(f"{C_RED}❌ ERROR FATAL: Hay {len(cat_cols)} columnas categóricas sin procesar (ej. {list(cat_cols)[:3]}).{C_RESET}")
        errores += 1
        detalle['tipos_ok'] = False
    else:
        print(f"{C_GREEN}✅ OK: Todas las variables son numéricas.{C_RESET}")
        detalle['tipos_ok'] = True
        
    # 3. Detección de Multicolinealidad (ADVERTENCIA)
    df_num = df.select_dtypes(include=[np.number])
    detalle['sin_multicolinealidad'] = True
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
            detalle['sin_multicolinealidad'] = False
        else:
            print(f"{C_GREEN}✅ OK: No se detectó multicolinealidad severa.{C_RESET}")

    # 4. Desbalance de Clases (ADVERTENCIA) y Data Leakage (FATAL) si se pasa el target
    detalle['sin_leakage'] = True
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
                detalle['clases_balanceadas'] = False
            else:
                print(f"{C_GREEN}✅ OK: Distribución de clase objetivo balanceada.{C_RESET}")
                detalle['clases_balanceadas'] = True
        
        # Fuga de Información (Data Leakage)
        for col in df_num.columns:
            try:
                corr = np.abs(df_num[col].corr(y_series))
                if corr > 0.99:
                    print(f"{C_RED}❌ ERROR FATAL: La columna '{col}' tiene una correlación de {corr:.4f} con el target.{C_RESET}")
                    print("        Esto indica una fuga de datos (Data Leakage). ¡Elimina esta columna de las características!")
                    errores += 1
                    detalle['sin_leakage'] = False
            except Exception:
                pass
                
    # Diagnóstico Final
    if errores > 0:
        print(f"\n{C_BOLD}{C_RED}⚠️ RESULTADO: La auditoría falló con {errores} error(es) fatal(es) y {advertencias} advertencia(s). Corrige los errores antes de continuar.{C_RESET}")
        aprobado = False
    elif advertencias > 0:
        print(f"\n{C_BOLD}{C_YELLOW}🏆 RESULTADO: Datos listos para modelar, pero con {advertencias} advertencia(s) a tener en cuenta.{C_RESET}")
        aprobado = True
    else:
        print(f"\n{C_BOLD}{C_GREEN}🏆 RESULTADO: Auditoría exitosa. Datos limpios y seguros para producción o modelado.{C_RESET}")
        aprobado = True

    if retornar_detalle:
        detalle['aprobado'] = aprobado
        return detalle
    return aprobado

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

def generar_reporte_auditoria_markdown(
    nombre_proyecto: str,
    df_info: Dict[str, Any],
    resultado_auditoria: bool,
    metricas_modelo: Dict[str, float],
    top_features: Optional[pd.DataFrame] = None,
    output_path: str = "reporte_auditoria.md",
    detalle_auditoria: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Genera un informe técnico completo en formato Markdown con los resultados
    de la auditoría QA, calidad de datos y evaluación del modelo.
    
    Args:
        nombre_proyecto: Nombre del proyecto o dataset.
        df_info: Diccionario con llaves como 'filas', 'columnas', 'nulos', 'memoria_mb'.
        resultado_auditoria: Indica si el dataset aprobó la auditoría QA.
        metricas_modelo: Diccionario con las métricas obtenidas.
        top_features: DataFrame opcional con 'Feature' e 'Importance'.
        output_path: Ruta de salida para el archivo Markdown.
        detalle_auditoria: Diccionario opcional con el resultado real de cada
            chequeo (tal como lo retorna `revisar_datos_finales(..., retornar_detalle=True)`),
            con llaves como 'tipos_ok', 'sin_leakage', 'sin_multicolinealidad'.
            Si no se provee, el reporte no asume que estos chequeos pasaron:
            solo los marca como ✅ cuando `resultado_auditoria` es True (ya que
            en ese caso ningún chequeo fatal pudo haber fallado); si la
            auditoría fue rechazada y no hay detalle, se muestra como
            "no verificado" en vez de asumir que está todo bien.
        
    Returns:
        Ruta del archivo Markdown generado.
    """
    fecha = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    estado = "✅ APROBADO PARA PRODUCCIÓN" if resultado_auditoria else "❌ RECHAZADO (Errores Críticos)"

    detalle_auditoria = detalle_auditoria or {}

    def _linea_chequeo(clave: str, texto_ok: str, texto_fail: str) -> str:
        if clave in detalle_auditoria:
            return texto_ok if detalle_auditoria[clave] else texto_fail
        if resultado_auditoria:
            # Si la auditoría completa pasó, ningún chequeo fatal pudo haber fallado.
            return texto_ok
        # Auditoría rechazada y sin detalle: no asumir que este chequeo puntual pasó.
        return "⚠️ No verificado (ver detalle completo en la salida de consola)"

    tipos_linea = _linea_chequeo(
        'tipos_ok',
        "✅ Numéricos y codificados en One-Hot / Label Encoding",
        "❌ Hay columnas categóricas sin procesar",
    )
    leakage_linea = _linea_chequeo(
        'sin_leakage',
        "✅ Sin correlación perfecta con la variable objetivo",
        "❌ Se detectó alta correlación (posible fuga de datos) con la variable objetivo",
    )
    multicolinealidad_linea = df_info.get(
        'multicolinealidad',
        _linea_chequeo(
            'sin_multicolinealidad',
            "✅ Sin alertas severas",
            "⚠️ Se detectaron variables altamente correlacionadas entre sí",
        ),
    )

    md_content = f"""# 🛡️ Reporte de Gobernanza y Auditoría de Datos – DS Guardian

**Proyecto:** {nombre_proyecto}  
**Fecha de Generación:** {fecha}  
**Estado QA:** {estado}  

---

## 📊 1. Resumen de la Estructura de Datos

| Métrica | Valor |
| :--- | :--- |
| **Total de Filas / Muestras** | {df_info.get('filas', 'N/A')} |
| **Total de Columnas** | {df_info.get('columnas', 'N/A')} |
| **Valores Nulos Restantes** | {df_info.get('nulos', 0)} |
| **Uso de Memoria RAM** | {df_info.get('memoria_mb', 'N/A')} MB |

---

## 🕵️ 2. Diagnóstico del Agente de Auditoría QA

* **Control de Nulos:** {'✅ Sin nulos' if df_info.get('nulos', 0) == 0 else '❌ Con nulos'}
* **Tipos de Datos:** {tipos_linea}
* **Fuga de Datos (Data Leakage):** {leakage_linea}
* **Multicolinealidad:** {multicolinealidad_linea}

---

## 🤖 3. Métricas de Evaluación del Modelo

| Métrica de Rendimiento | Valor Obtenido |
| :--- | :--- |
"""
    for k, v in metricas_modelo.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        md_content += f"| **{k}** | {val_str} |\n"

    if top_features is not None and not top_features.empty:
        md_content += """
---

## 💡 4. Explicabilidad del Modelo (Top Características)

| Ranking | Variable / Característica | Importancia Relativa |
| :---: | :--- | :---: |
"""
        for idx, row in top_features.head(10).iterrows():
            feat = row['Feature']
            imp = f"{row['Importance'] * 100:.2f}%" if isinstance(row['Importance'], float) else str(row['Importance'])
            md_content += f"| {idx + 1} | `{feat}` | {imp} |\n"

    md_content += """
---
*Generado automáticamente por el Framework DS Guardian – AI Data Science Governance System.*
"""

    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"\n{C_BOLD}{C_GREEN}📄 Reporte de auditoría Markdown generado en: '{output_path}'{C_RESET}")
    return output_path

