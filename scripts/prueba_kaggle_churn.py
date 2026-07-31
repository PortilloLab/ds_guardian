"""
Prueba Real de DS Guardian con el Dataset de Kaggle: Customer Churn
====================================================================
Demuestra el ciclo de vida completo de gobierno de datos, auditoría 
y entrenamiento de modelos sobre un dataset real de 10,000 registros.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from ds_guardian import eda, limpieza, modelos, auditoria

def ejecutar_prueba_kaggle():
    print("=" * 70)
    print("🚀 INICIANDO PRUEBA CON DATASET DE KAGGLE: CUSTOMER CHURN 🚀")
    print("=" * 70)

    # 1. Carga del Dataset
    dataset_path = "data/customer_churn_business_dataset.csv"
    print(f"\n📂 1. Cargando dataset desde '{dataset_path}'...")
    df = pd.read_csv(dataset_path)
    print(f"   -> Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

    # 2. EDA y Optimización de Memoria
    print("\n" + "-" * 50)
    print("📊 2. MÓDULO EDA (Optimización de Memoria & Outliers)")
    print("-" * 50)
    df = eda.optimizar_memoria(df)

    # Detectar y acotar outliers en variables numéricas clave
    cols_num = ['age', 'tenure_months', 'monthly_fee', 'total_revenue', 'support_tickets']
    print(f"   -> Acotando outliers por método IQR en: {cols_num}")
    df = eda.acotar_outliers_iqr(df, columnas=cols_num)

    # 3. Preparación de Variables (Drop ID)
    print("\n" + "-" * 50)
    print("✂️ 3. SEPARACIÓN TRAIN / TEST")
    print("-" * 50)
    
    # Eliminamos el ID único para prevenir memorización/leakage
    X = df.drop(columns=['customer_id', 'churn'])
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   -> Train set: {X_train.shape[0]} muestras")
    print(f"   -> Test set:  {X_test.shape[0]} muestras")

    # 4. Preprocesamiento Seguro (Limpieza)
    print("\n" + "-" * 50)
    print("🧹 4. MÓDULO LIMPIEZA (Imputación & Codificación)")
    print("-" * 50)
    X_train, X_test = limpieza.imputar_nulos(X_train, X_test)
    X_train, X_test = limpieza.codificar_variables(X_train, X_test)
    print(f"   -> Dimensiones finales tras codificación One-Hot: {X_train.shape[1]} características")

    # 5. Auditoría QA antes de Entrenar
    print("\n" + "-" * 50)
    print("🕵️ 5. MÓDULO AUDITORÍA (Control de Calidad QA)")
    print("-" * 50)
    es_valido = auditoria.revisar_datos_finales(X_train, y=y_train)

    if not es_valido:
        print("❌ La auditoría detectó errores críticos. Entrenamiento cancelado.")
        return

    # 6. Entrenamiento & Evaluación de Modelo
    print("\n" + "-" * 50)
    print("🤖 6. MÓDULO MODELOS (Random Forest & Evaluación)")
    print("-" * 50)
    clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    # Probabilidades de predicción y umbral adaptado para datos desbalanceados
    y_prob = clf.predict_proba(X_test)[:, 1]
    umbral = 0.25  # Adaptado para detectar fuga con clase minoritaria del 10%
    y_pred = (y_prob >= umbral).astype(int)
    
    # Evaluación y gráfico de Matriz de Confusión
    output_plot = 'plots/kaggle_churn_confusion_matrix.png'
    modelos.evaluar_clasificacion(y_test, y_pred, y_prob=y_prob, save_path=output_plot)

    score_f1 = f1_score(y_test, y_pred, pos_label=1)
    print(f"   -> F1-Score (Clase Churn=1): {score_f1:.4f}")

    # 7. Auditoría e Historial de Modelos
    print("\n" + "-" * 50)
    print("📈 7. REGISTRO EN HISTORIAL AUDITADO")
    print("-" * 50)
    auditoria.registrar_y_comparar_modelo(
        nombre_proyecto="Kaggle_Customer_Churn",
        metrica_principal_nombre="F1-Score",
        valor_metrica=score_f1
    )

    print("\n" + "=" * 70)
    print("✅ PRUEBA KAGGLE FINALIZADA CON ÉXITO CON DS GUARDIAN")
    print("=" * 70)

if __name__ == '__main__':
    ejecutar_prueba_kaggle()
