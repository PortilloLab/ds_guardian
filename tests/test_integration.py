import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ds_guardian import (
    imputar_nulos, 
    codificar_variables, 
    escalar_caracteristicas, 
    revisar_datos_finales,
    registrar_y_comparar_modelo
)
from ds_guardian.auditoria.reporter import generar_reporte_html, generar_reporte_markdown
from ds_guardian.modelos import evaluar_clasificacion, exportar_modelo

def test_full_data_science_pipeline(tmp_path):
    # 1. Dataset Sintético con nulos, categóricas y números
    df = pd.DataFrame({
        'edad': [25, 30, np.nan, 45, 50, 35],
        'salario': [50000, 60000, 70000, np.nan, 90000, 80000],
        'ciudad': ['Posadas', 'BA', 'Posadas', 'Cordoba', None, 'BA'],
        'target': [0, 1, 0, 1, 0, 1]
    })
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 2. Pipeline de Limpieza & Transformación
    X_imp = imputar_nulos(X)
    X_enc = codificar_variables(X_imp)
    X_scaled = escalar_caracteristicas(X_enc)
    
    # 3. Auditoría QA
    qa_pass = revisar_datos_finales(X_scaled, y=y)
    assert qa_pass is True
    
    # 4. Entrenar y Evaluar Modelo
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_scaled, y)
    preds = clf.predict(X_scaled)
    metrics = evaluar_clasificacion(y, preds)
    assert metrics['accuracy'] == 1.0
    
    # 5. Exportar y Reporte HTML
    model_file = str(tmp_path / "model.joblib")
    html_file = str(tmp_path / "report.html")
    exportar_modelo(clf, model_file)
    generar_reporte_html(X_scaled, y, html_file)
    
    assert os.path.exists(model_file)
    assert os.path.exists(html_file)
