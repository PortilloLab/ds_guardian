"""
Script de Prueba de Estrés (Stress Test) para DS Guardian
Demuestra el comportamiento del framework ante datos corruptos, outliers y data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ds_guardian import eda, limpieza, modelos, auditoria

print("="*70)
print("🔥 INICIANDO PRUEBA DE ESTRÉS DE DS GUARDIAN 🔥")
print("="*70)

# 1. Generar Dataset "Sucio" y Problemático
np.random.seed(101)
n_samples = 200

data = {
    'Edad': np.random.choice([25, 35, 45, 55, np.nan, -10, 150], size=n_samples),
    'Ingresos': np.random.choice([30000, 50000, 75000, np.nan, 50000000], size=n_samples),
    'Nivel_Educacion': np.random.choice(['Secundario', 'Universitario', 'Posgrado', np.nan], size=n_samples),
    # Fuga de datos simulada (Columna casi idéntica al Target)
    'Fuga_Perfecta': [0]*n_samples, 
    'Target': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
}

# Inyectar Data Leakage deliberado
df = pd.DataFrame(data)
df['Fuga_Perfecta'] = df['Target']  # Correlación de 1.0 con la meta

print(f"\n1. Dataset Sucio Generado: {df.shape[0]} filas, {df.shape[1]} columnas")
print("-> Muestra de datos con nulos, outliers y columna de fuga:")
print(df.head())

# 2. Módulo EDA: Optimización de memoria y Detección de Outliers
print("\n" + "-"*50)
print("2. PRUEBA MÓDULO EDA (Optimización & Outliers)")
print("-"*50)
df = eda.optimizar_memoria(df)
outliers = eda.detectar_outliers_iqr(df)
df = eda.acotar_outliers_iqr(df, columnas=['Ingresos', 'Edad'])

# 3. Separación Train/Test antes de limpiar
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Módulo LIMPIEZA: Preprocesamiento sin fugas
print("\n" + "-"*50)
print("3. PRUEBA MÓDULO LIMPIEZA (Imputación y Codificación)")
print("-"*50)
X_train, X_test = limpieza.imputar_nulos(X_train, X_test)
X_train, X_test = limpieza.codificar_variables(X_train, X_test)

# 5. Módulo AUDITORÍA: El Agente QA en Acción
print("\n" + "-"*50)
print("4. PRUEBA MÓDULO AUDITORÍA (Control de Calidad QA)")
print("-"*50)
print("El auditor analizará si el dataset es apto antes de entrenar...")

es_valido = auditoria.revisar_datos_finales(X_train, y=y_train)

# 6. Entrenamiento & Registro si aprueba
if es_valido:
    print("\n" + "-"*50)
    print("5. PRUEBA MÓDULO MODELOS (Entrenamiento & Registro)")
    print("-"*50)
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    metrics = modelos.evaluar_clasificacion(y_test, y_pred, save_path='plots/estres_confusion_matrix.png')
    
    # Registrar en historial
    auditoria.registrar_y_comparar_modelo(
        nombre_proyecto="Prueba_Estres_DS_Guardian",
        metrica_principal_nombre="F1-Score",
        valor_metrica=metrics['f1_score']
    )

print("\n" + "="*70)
print("✅ PRUEBA DE ESTRÉS FINALIZADA EXITOSAMENTE")
print("="*70)
