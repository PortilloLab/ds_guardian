import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import ds_guardian
from ds_guardian.auditoria import revisar_datos_finales, registrar_y_comparar_modelo
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


def test_revisar_datos_finales_retornar_detalle():
    """
    Con retornar_detalle=True, el resultado debe seguir siendo compatible
    en su clave 'aprobado' y además exponer el detalle real de cada chequeo.
    """
    df_leak = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0], 'leak': [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])

    detalle = revisar_datos_finales(df_leak, y=y, retornar_detalle=True)
    assert isinstance(detalle, dict)
    assert detalle['aprobado'] is False
    assert detalle['sin_leakage'] is False
    assert detalle['tipos_ok'] is True
    assert detalle['sin_nulos'] is True


def test_reporte_markdown_no_miente_sobre_leakage_real():
    """
    Regresión: el reporte Markdown NO debe declarar '✅ Sin correlación perfecta'
    cuando la auditoría real detectó data leakage. Antes de este fix, esas líneas
    estaban hardcodeadas en el reporte sin importar el resultado real.
    """
    from ds_guardian.auditoria import generar_reporte_auditoria_markdown

    df_leak = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0], 'leak': [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])

    detalle = revisar_datos_finales(df_leak, y=y, retornar_detalle=True)
    assert detalle['aprobado'] is False

    output_path = "reporte_test_leakage_regresion.md"
    try:
        generar_reporte_auditoria_markdown(
            nombre_proyecto="RegresionLeakage",
            df_info={'filas': 4, 'columnas': 2, 'nulos': 0, 'memoria_mb': 0.01},
            resultado_auditoria=detalle['aprobado'],
            metricas_modelo={'Accuracy': 1.0},
            detalle_auditoria=detalle,
            output_path=output_path,
        )
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "❌ RECHAZADO" in content
        # La línea de Fuga de Datos NO debe decir que está todo OK.
        assert "Sin correlación perfecta con la variable objetivo" not in content
        assert "posible fuga de datos" in content.lower()
    finally:
        import os
        if os.path.exists(output_path):
            os.remove(output_path)


def test_public_api_exports_expected_objects():
    assert "revisar_datos_finales" in ds_guardian.__all__
    assert "optimizar_memoria" in ds_guardian.__all__
    assert "registrar_y_comparar_modelo" in ds_guardian.__all__
    assert ds_guardian.__version__ == "1.0.0"


def test_registrar_y_comparar_modelo_usa_ruta_del_proyecto(monkeypatch, tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    historial_path = project_root / "historial_proyectos.json"
    if historial_path.exists():
        historial_path.unlink()

    monkeypatch.chdir(tmp_path)
    registrar_y_comparar_modelo("path_project", "Accuracy", 0.82)

    assert historial_path.exists()
    assert not (tmp_path / "historial_proyectos.json").exists()


