import os
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
from .detector_leakage import detectar_multicolinealidad, detectar_fuga_target


def generar_reporte_html(df: pd.DataFrame, y: Optional[Any] = None, output_path: str = "reporte_ds_guardian.html") -> str:
    """Genera un informe ejecutivo interactivo en HTML con diseño premium (glassmorphism)."""
    total_filas, total_cols = df.shape
    nulos = df.isnull().sum().sum()
    pct_nulos = (nulos / (total_filas * total_cols)) * 100 if total_filas > 0 else 0
    cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    high_corr = detectar_multicolinealidad(df, umbral=0.95)
    leaks = detectar_fuga_target(df, y, umbral=0.99) if y is not None else []

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>DS Guardian - Executive Audit Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 2rem; }}
        .card {{ background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }}
        h1 {{ color: #38bdf8; margin-top: 0; }}
        h2 {{ color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 0.5rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }}
        .metric {{ background: #1e293b; padding: 1rem; border-radius: 8px; text-align: center; border-left: 4px solid #38bdf8; }}
        .metric-value {{ font-size: 1.8rem; font-weight: bold; color: #f8fafc; }}
        .status-pass {{ color: #4ade80; font-weight: bold; }}
        .status-fail {{ color: #f87171; font-weight: bold; }}
        .status-warn {{ color: #fbbf24; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #1e293b; color: #38bdf8; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>🛡️ DS Guardian — Executive Audit Report</h1>
        <p>Generado el: <strong>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</strong></p>
        <div class="grid">
            <div class="metric"><div>Filas Total</div><div class="metric-value">{total_filas:,}</div></div>
            <div class="metric"><div>Columnas</div><div class="metric-value">{total_cols}</div></div>
            <div class="metric"><div>Nulos Total</div><div class="metric-value {'status-fail' if nulos > 0 else 'status-pass'}">{nulos} ({pct_nulos:.2f}%)</div></div>
            <div class="metric"><div>Multicolinealidad</div><div class="metric-value {'status-warn' if high_corr else 'status-pass'}">{len(high_corr)} pares</div></div>
            <div class="metric"><div>Data Leakage</div><div class="metric-value {'status-fail' if leaks else 'status-pass'}">{len(leaks)} alertas</div></div>
        </div>
    </div>
    <div class="card">
        <h2>📋 Resumen de Sanidad del Dataset</h2>
        <table>
            <tr><th>Auditoría</th><th>Resultado</th><th>Detalles</th></tr>
            <tr><td>Ausencia de Valores Nulos</td><td class="{'status-pass' if nulos == 0 else 'status-fail'}">{'✅ PASÓ' if nulos == 0 else '❌ FALLÓ'}</td><td>{nulos} nulos detectados en el dataset.</td></tr>
            <tr><td>Codificación Categórica</td><td class="{'status-pass' if len(cat_cols) == 0 else 'status-fail'}">{'✅ PASÓ' if len(cat_cols) == 0 else '❌ FALLÓ'}</td><td>{len(cat_cols)} columnas sin codificar.</td></tr>
            <tr><td>Escáner de Data Leakage (>0.99)</td><td class="{'status-pass' if len(leaks) == 0 else 'status-fail'}">{'✅ LIBRE' if len(leaks) == 0 else '❌ ALERTA FATAL'}</td><td>{len(leaks)} columnas sospechosas detectadas.</td></tr>
        </table>
    </div>
</body>
</html>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_path


def generar_reporte_markdown(df: pd.DataFrame, y: Optional[Any] = None, output_path: str = "reporte_ds_guardian.md") -> str:
    """Genera un informe en formato Markdown."""
    nulos = df.isnull().sum().sum()
    md_content = f"""# 🛡️ DS Guardian — Audit Summary Report
*Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

- **Dimensiones:** {df.shape[0]} filas × {df.shape[1]} columnas
- **Nulos:** {nulos}
- **Variables Numéricas:** {len(df.select_dtypes(include=['number']).columns)}
- **Estado Auditoría:** {'✅ APROBADO' if nulos == 0 else '❌ RECHAZADO'}
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    return output_path


def generar_reporte_auditoria_markdown(
    nombre_proyecto: str,
    df_info: Dict[str, Any],
    resultado_auditoria: bool,
    metricas_modelo: Dict[str, float],
    top_features: Optional[pd.DataFrame] = None,
    output_path: str = "reporte_auditoria.md",
    detalle_auditoria: Optional[Dict[str, Any]] = None,
) -> str:
    fecha = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    estado = "✅ APROBADO PARA PRODUCCIÓN" if resultado_auditoria else "❌ RECHAZADO (Errores Críticos)"
    detalle_auditoria = detalle_auditoria or {}

    def _linea_chequeo(clave: str, texto_ok: str, texto_fail: str) -> str:
        if clave in detalle_auditoria:
            return texto_ok if detalle_auditoria[clave] else texto_fail
        if resultado_auditoria:
            return texto_ok
        return "⚠️ No verificado (ver detalle completo en la salida de consola)"

    tipos_linea = _linea_chequeo('tipos_ok', "✅ Numéricos y codificados en One-Hot / Label Encoding", "❌ Hay columnas categóricas sin procesar")
    leakage_linea = _linea_chequeo('sin_leakage', "✅ Sin correlación perfecta con la variable objetivo", "❌ Se detectó alta correlación (posible fuga de datos) con la variable objetivo")
    multicolinealidad_linea = _linea_chequeo('sin_multicolinealidad', "✅ Sin alertas severas", "⚠️ Se detectaron variables altamente correlacionadas entre sí")

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
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    return output_path


def generar_reporte_auditoria_html(
    nombre_proyecto: str,
    df_info: Dict[str, Any],
    resultado_auditoria: bool,
    metricas_modelo: Dict[str, float],
    top_features: Optional[pd.DataFrame] = None,
    output_path: str = "reportes/reporte_auditoria.html",
) -> str:
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
        body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background-color: #0f172a; color: #f8fafc; margin: 0; padding: 40px 20px; }}
        .container {{ max-width: 900px; margin: 0 auto; background: #1e293b; border-radius: 16px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); padding: 32px; border: 1px solid #334155; }}
        h1 {{ color: #38bdf8; font-size: 2em; margin-bottom: 8px; display: flex; align-items: center; gap: 12px; }}
        .subtitle {{ color: #94a3b8; font-size: 0.95em; margin-bottom: 24px; }}
        .grid {{ display: flex; gap: 16px; flex-wrap: wrap; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 18px; }}
        th, td {{ padding: 12px; border-bottom: 1px solid #334155; text-align: left; }}
        th {{ color: #63b3ed; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ DS Guardian</h1>
        <div class="subtitle">{fecha}</div>
        {badge_estado}
        <div class="grid" style="margin-top: 24px;">{metrics_cards_html}</div>
        <table>
            <tr><th>Métrica</th><th>Valor</th></tr>
            {''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in metricas_modelo.items())}
        </table>
        {f'<h2>Top Features</h2><table><tr><th>#</th><th>Feature</th><th>Importancia</th><th>Barra</th></tr>{features_rows_html}</table>' if features_rows_html else ''}
    </div>
</body>
</html>"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return output_path
