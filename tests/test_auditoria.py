import pytest
import pandas as pd
import numpy as np
from ds_guardian.auditoria import revisar_datos_finales
from ds_guardian.exceptions import DataValidationError

def test_revisar_datos_finales_con_nulos():
    df = pd.DataFrame({'A': [1.0, 2.0, np.nan]})
    # Si hay nulos, no pasa el auditor básico (retorna False)
    assert revisar_datos_finales(df) is False

def test_revisar_datos_finales_con_textos():
    df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': ['Texto', 'Texto', 'Texto']})
    # Si hay texto sin codificar, retorna False
    assert revisar_datos_finales(df) is False

def test_revisar_datos_finales_correcto():
    df = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
    # Si está limpio y numérico, retorna True
    assert revisar_datos_finales(df) is True

def test_revisar_datos_finales_raises_validation_error():
    with pytest.raises(DataValidationError):
        revisar_datos_finales(None)

def test_generar_reporte_auditoria_markdown(tmp_path):
    from ds_guardian.auditoria import generar_reporte_auditoria_markdown
    
    report_file = str(tmp_path / "test_report.md")
    df_info = {'filas': 100, 'columnas': 10, 'nulos': 0, 'memoria_mb': 0.05}
    metrics = {'Accuracy': 0.95, 'F1-Score': 0.94}
    df_imp = pd.DataFrame({'Feature': ['colA', 'colB'], 'Importance': [0.6, 0.4]})
    
    res_file = generar_reporte_auditoria_markdown(
        nombre_proyecto="TestProject",
        df_info=df_info,
        resultado_auditoria=True,
        metricas_modelo=metrics,
        top_features=df_imp,
        output_path=report_file
    )
    
    assert res_file == report_file
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()
    assert "# 🛡️ Reporte de Gobernanza y Auditoría de Datos – DS Guardian" in content
    assert "TestProject" in content
    assert "0.95" in content

