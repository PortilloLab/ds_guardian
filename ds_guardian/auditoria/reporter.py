import os
from datetime import datetime
import pandas as pd
from typing import Optional, Any
from .detector_leakage import detectar_multicolinealidad, detectar_fuga_target

def generar_reporte_html(df: pd.DataFrame, y: Optional[Any] = None, output_path: str = "reporte_ds_guardian.html") -> str:
    """Genera un informe ejecutivo interactivo en HTML con diseño premium (glassmorphism)."""
    total_filas, total_cols = df.shape
    nulos = df.isnull().sum().sum()
    pct_nulos = (nulos / (total_filas * total_cols)) * 100 if total_filas > 0 else 0
    cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    num_cols = list(df.select_dtypes(include=['number']).columns)
    
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
            <div class="metric">
                <div>Filas Total</div>
                <div class="metric-value">{total_filas:,}</div>
            </div>
            <div class="metric">
                <div>Columnas</div>
                <div class="metric-value">{total_cols}</div>
            </div>
            <div class="metric">
                <div>Nulos Total</div>
                <div class="metric-value {'status-fail' if nulos > 0 else 'status-pass'}">{nulos} ({pct_nulos:.2f}%)</div>
            </div>
            <div class="metric">
                <div>Multicolinealidad</div>
                <div class="metric-value {'status-warn' if high_corr else 'status-pass'}">{len(high_corr)} pares</div>
            </div>
            <div class="metric">
                <div>Data Leakage</div>
                <div class="metric-value {'status-fail' if leaks else 'status-pass'}">{len(leaks)} alertas</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>📋 Resumen de Sanidad del Dataset</h2>
        <table>
            <tr><th>Auditoría</th><th>Resultado</th><th>Detalles</th></tr>
            <tr>
                <td>Ausencia de Valores Nulos</td>
                <td class="{ 'status-pass' if nulos == 0 else 'status-fail' }">{ '✅ PASÓ' if nulos == 0 else '❌ FALLÓ' }</td>
                <td>{nulos} nulos detectados en el dataset.</td>
            </tr>
            <tr>
                <td>Codificación Categórica</td>
                <td class="{ 'status-pass' if len(cat_cols) == 0 else 'status-fail' }">{ '✅ PASÓ' if len(cat_cols) == 0 else '❌ FALLÓ' }</td>
                <td>{len(cat_cols)} columnas sin codificar.</td>
            </tr>
            <tr>
                <td>Escáner de Data Leakage (>0.99)</td>
                <td class="{ 'status-pass' if len(leaks) == 0 else 'status-fail' }">{ '✅ LIBRE' if len(leaks) == 0 else '❌ ALERTA FATAL' }</td>
                <td>{len(leaks)} columnas sospechosas detectadas.</td>
            </tr>
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
