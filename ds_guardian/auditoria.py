import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, Union
import pandas as pd
import numpy as np
from .exceptions import DataValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HISTORIAL_FILE = PROJECT_ROOT / 'historial_proyectos.json'

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
    historial_path = Path(HISTORIAL_FILE)
    if historial_path.exists():
        try:
            with historial_path.open('r', encoding='utf-8') as f:
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
    
    historial_path.parent.mkdir(parents=True, exist_ok=True)
    with historial_path.open('w', encoding='utf-8') as f:
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


def generar_reporte_auditoria_html(
    nombre_proyecto: str,
    df_info: Dict[str, Any],
    resultado_auditoria: bool,
    metricas_modelo: Dict[str, float],
    top_features: Optional[pd.DataFrame] = None,
    output_path: str = "reportes/reporte_auditoria.html"
) -> str:
    """
    Genera un informe interactivo y estilizado en formato HTML con diseño oscuro / glassmorphic,
    tarjetas de métricas, diagnóstico QA y explicabilidad.
    
    Args:
        nombre_proyecto: Nombre del proyecto o dataset.
        df_info: Métricas descriptivas (filas, columnas, nulos, memoria).
        resultado_auditoria: Resultado booleano de la auditoría.
        metricas_modelo: Diccionario con los scores de rendimiento.
        top_features: DataFrame con el top de características importantes.
        output_path: Ruta de salida del archivo HTML.
        
    Returns:
        Ruta del archivo HTML generado.
    """
    fecha = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    badge_estado = '<span style="background-color: #2e7d32; color: #fff; padding: 6px 16px; border-radius: 20px; font-weight: bold;">✅ APROBADO PARA PRODUCCIÓN</span>' if resultado_auditoria else '<span style="background-color: #c62828; color: #fff; padding: 6px 16px; border-radius: 20px; font-weight: bold;">❌ RECHAZADO (Advertencias/Errores)</span>'

    metrics_cards_html = ""
    for k, v in metricas_modelo.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        metrics_cards_html += f"""
        <div style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 20px; text-align: center; flex: 1; min-width: 150px;">
            <div style="font-size: 0.9em; color: #a0aec0; margin-bottom: 8px;">{k}</div>
            <div style="font-size: 1.8em; font-weight: bold; color: #63b3ed;">{val_str}</div>
        </div>
        """

    features_rows_html = ""
    if top_features is not None and not top_features.empty:
        for idx, row in top_features.head(10).iterrows():
            feat = row['Feature']
            imp = f"{row['Importance'] * 100:.2f}%" if isinstance(row['Importance'], float) else str(row['Importance'])
            pct_val = row['Importance'] * 100 if isinstance(row['Importance'], float) else 0
            features_rows_html += f"""
            <tr>
                <td style="padding: 12px; text-align: center; font-weight: bold; color: #cbd5e0;">{idx + 1}</td>
                <td style="padding: 12px; color: #e2e8f0; font-family: monospace;">{feat}</td>
                <td style="padding: 12px; text-align: right; color: #63b3ed; font-weight: bold;">{imp}</td>
                <td style="padding: 12px; width: 40%;">
                    <div style="background: #2d3748; border-radius: 6px; overflow: hidden; height: 10px;">
                        <div style="background: linear-gradient(90deg, #3182ce, #63b3ed); width: {min(100, max(5, pct_val))}%; height: 100%;"></div>
                    </div>
                </td>
            </tr>
            """

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DS Guardian - Reporte de Auditoría: {nombre_proyecto}</title>
    <style>
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: #1e293b;
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
            padding: 32px;
            border: 1px solid #334155;
        }}
        h1 {{
            color: #38bdf8;
            font-size: 2em;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .subtitle {{
            color: #94a3b8;
            font-size: 0.95em;
            margin-bottom: 24px;
        }}
        .grid {{
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 32px;
        }}
        .section {{
            background: #0f172a;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 28px;
            border: 1px solid #334155;
        }}
        h2 {{
            color: #f1f5f9;
            font-size: 1.3em;
            margin-top: 0;
            border-bottom: 1px solid #334155;
            padding-bottom: 12px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background: #1e293b;
            color: #94a3b8;
            text-align: left;
            padding: 12px;
            font-size: 0.85em;
            text-transform: uppercase;
        }}
        tr:nth-child(even) {{ background: rgba(255, 255, 255, 0.02); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ DS Guardian Governance Report</h1>
        <div class="subtitle">Proyecto: <strong>{nombre_proyecto}</strong> | Generado: {fecha}</div>
        
        <div style="margin-bottom: 28px;">
            {badge_estado}
        </div>

        <div class="section">
            <h2>📊 Resumen de Estructura de Datos</h2>
            <div class="grid">
                <div style="flex: 1; background: #1e293b; padding: 16px; border-radius: 8px; text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.85em;">Total Filas</div>
                    <div style="font-size: 1.5em; font-weight: bold; color: #f8fafc;">{df_info.get('filas', 'N/A')}</div>
                </div>
                <div style="flex: 1; background: #1e293b; padding: 16px; border-radius: 8px; text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.85em;">Total Columnas</div>
                    <div style="font-size: 1.5em; font-weight: bold; color: #f8fafc;">{df_info.get('columnas', 'N/A')}</div>
                </div>
                <div style="flex: 1; background: #1e293b; padding: 16px; border-radius: 8px; text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.85em;">Uso de RAM</div>
                    <div style="font-size: 1.5em; font-weight: bold; color: #f8fafc;">{df_info.get('memoria_mb', 'N/A')} MB</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🤖 Métricas de Evaluación de Modelo</h2>
            <div class="grid">
                {metrics_cards_html}
            </div>
        </div>

        {"<div class='section'><h2>💡 Explicabilidad del Modelo (Top Features)</h2><table><thead><tr><th>#</th><th>Característica</th><th style='text-align: right;'>Importancia</th><th>Visualización</th></tr></thead><tbody>" + features_rows_html + "</tbody></table></div>" if features_rows_html else ""}

        <div style="text-align: center; color: #64748b; font-size: 0.85em; margin-top: 32px;">
            Generado automáticamente por <strong>DS Guardian - AI Governance System</strong>.
        </div>
    </div>
</body>
</html>
"""

    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"{C_BOLD}{C_GREEN}🌐 Reporte interactivo HTML generado en: '{output_path}'{C_RESET}")
    return output_path


