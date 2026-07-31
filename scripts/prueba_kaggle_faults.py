"""
Prueba Real de DS Guardian: Dataset Kaggle Steel Plates Faults
===============================================================
Incluye eliminación de colinealidad, optimización de hiperparámetros,
explicabilidad (Feature Importance) y generación automática de reporte de auditoría en Markdown.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from ds_guardian import eda, limpieza, modelos, auditoria

def ejecutar_prueba_faults_optimizada():
    print("=" * 70)
    print("⚙️ INICIANDO PRUEBA COMPLETA: STEEL PLATES FAULTS (DS GUARDIAN) ⚙️")
    print("=" * 70)

    # 1. Cargar Dataset
    filepath = "data/faults.csv"
    print(f"\n📂 1. Cargando dataset desde '{filepath}'...")
    df = pd.read_csv(filepath)

    # Consolidados de fallas multiclase
    fault_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    df['Fault_Type'] = df[fault_cols].idxmax(axis=1)
    df = df.drop(columns=fault_cols)

    # 2. Eliminación de Variables Multicolineales (Recomendación del Auditor QA)
    cols_redundantes = ['X_Maximum', 'Y_Maximum', 'Y_Perimeter', 'Sum_of_Luminosity', 'TypeOfSteel_A400']
    print(f"\n🧹 2. Eliminando {len(cols_redundantes)} variables redundantes detectadas por auditoría:")
    print(f"   -> Columns dropped: {cols_redundantes}")
    df = df.drop(columns=cols_redundantes)

    # 3. Módulo EDA: Optimización & Outliers
    print("\n" + "-" * 50)
    print("📊 3. MÓDULO EDA (Optimización de Memoria & Outliers)")
    print("-" * 50)
    mem_init = df.memory_usage().sum() / 1024 / 1024
    df = eda.optimizar_memoria(df)
    mem_end = df.memory_usage().sum() / 1024 / 1024

    cols_num = ['Pixels_Areas', 'Length_of_Conveyer', 'Steel_Plate_Thickness']
    print(f"   -> Acotando outliers por método IQR en: {cols_num}")
    df = eda.acotar_outliers_iqr(df, columnas=cols_num)

    # 4. Separación Train / Test
    print("\n" + "-" * 50)
    print("✂️ 4. SEPARACIÓN TRAIN / TEST")
    print("-" * 50)
    X = df.drop(columns=['Fault_Type'])
    y = df['Fault_Type']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"   -> Train set: {X_train.shape[0]} muestras")
    print(f"   -> Test set:  {X_test.shape[0]} muestras")

    # 5. Módulo Limpieza
    print("\n" + "-" * 50)
    print("🧹 5. MÓDULO LIMPIEZA (Imputación & Codificación)")
    print("-" * 50)
    X_train, X_test = limpieza.imputar_nulos(X_train, X_test)
    X_train, X_test = limpieza.codificar_variables(X_train, X_test)

    # 6. Módulo Auditoría (Re-verificación QA)
    print("\n" + "-" * 50)
    print("🕵️ 6. MÓDULO AUDITORÍA (Re-verificación de Calidad QA)")
    print("-" * 50)
    es_valido = auditoria.revisar_datos_finales(X_train, y=y_train)

    if not es_valido:
        print("❌ La auditoría detectó errores críticos. Entrenamiento bloqueado.")
        return

    # 7. Módulo Modelos: Optimización Automática de Hiperparámetros
    print("\n" + "-" * 50)
    print("⚡ 7. MÓDULO MODELOS (Optimización de Hiperparámetros vía Random Search)")
    print("-" * 50)
    
    base_clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    best_clf = modelos.optimizar_hiperparametros(
        modelo=base_clf,
        param_grid=param_grid,
        X=X_train,
        y=y_train,
        cv=3,
        n_iter=15,
        scoring='f1_macro'
    )

    # 8. Evaluación del Modelo Optimizado
    print("\n" + "-" * 50)
    print("🤖 8. EVALUACIÓN DEL MODELO CON MEJORES HIPERPARÁMETROS")
    print("-" * 50)
    y_pred = best_clf.predict(X_test)
    y_prob = best_clf.predict_proba(X_test)

    output_plot_cm = 'plots/steel_faults_tuned_confusion_matrix.png'
    modelos.evaluar_clasificacion(y_test, y_pred, y_prob=y_prob, save_path=output_plot_cm)

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    score_f1_macro = f1_score(y_test, y_pred, average='macro')

    # 9. Explicabilidad: Importancia de Características
    print("\n" + "-" * 50)
    print("💡 9. MÓDULO EXPLICABILIDAD (Feature Importance)")
    print("-" * 50)
    output_plot_fi = 'plots/steel_faults_tuned_feature_importance.png'
    df_imp = modelos.graficar_importancia_caracteristicas(
        best_clf, feature_names=list(X_train.columns), top_n=10, save_path=output_plot_fi
    )
    print("   -> Top 10 Características tras optimización de hiperparámetros:")
    print(df_imp.head(10).to_string(index=False))

    # 10. Registro y Comparación en el Historial Auditado
    print("\n" + "-" * 50)
    print("📈 10. REGISTRO Y COMPARACIÓN EN HISTORIAL DE MODELOS")
    print("-" * 50)
    auditoria.registrar_y_comparar_modelo(
        nombre_proyecto="Kaggle_Steel_Plates_Faults",
        metrica_principal_nombre="F1-Macro",
        valor_metrica=score_f1_macro
    )

    # 11. Generación Automática de Reporte de Auditoría en Markdown
    print("\n" + "-" * 50)
    print("📄 11. GENERACIÓN AUTOMÁTICA DE REPORTE EN MARKDOWN")
    print("-" * 50)
    df_info = {
        'filas': df.shape[0],
        'columnas': df.shape[1],
        'nulos': df.isnull().sum().sum(),
        'memoria_mb': round(mem_end, 2),
        'multicolinealidad': '✅ 5 columnas redundantes eliminadas'
    }
    metricas_dict = {
        'Accuracy (Exactitud)': acc,
        'ROC-AUC Score (Multiclase)': roc_auc,
        'F1-Score Macro': score_f1_macro
    }
    report_path = "reports/reporte_auditoria_steel_faults.md"
    auditoria.generar_reporte_auditoria_markdown(
        nombre_proyecto="Kaggle Steel Plates Faults",
        df_info=df_info,
        resultado_auditoria=es_valido,
        metricas_modelo=metricas_dict,
        top_features=df_imp,
        output_path=report_path
    )

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETO Y REPORTE GENERADO CON ÉXITO POR DS GUARDIAN")
    print("=" * 70)

if __name__ == '__main__':
    ejecutar_prueba_faults_optimizada()
