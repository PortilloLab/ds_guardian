import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ds_guardian import eda, limpieza, modelos, auditoria, configurar_estilo

print("=====================================================================")
print("   INICIANDO WORKFLOW: PROTECTING DATA SCIENCE WORKFLOWS")
print("=====================================================================\n")

# 1. Crear datos sucios (Edad con nulos, Salario con outliers extremos)
np.random.seed(42)
n_samples = 400
df = pd.DataFrame({
    'Edad': np.random.randint(18, 70, n_samples).astype(float),
    'Salario': np.random.normal(60000, 15000, n_samples),
    'Abandono': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
})
df.loc[::10, 'Edad'] = np.nan
df.loc[15:18, 'Salario'] = [600000.0, -120000.0, 500000.0, 800000.0]

# 2. EDA: Capping de outliers en Salario
print("[1] Aplicando Winsorization en Salario...")
df = eda.acotar_outliers_iqr(df, columnas=['Salario'])

# 3. Separar Train y Test para evitar Data Leakage
X = df.drop('Abandono', axis=1)
y = df['Abandono']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Limpieza Segura (imputar y escalar)
print("\n[2] Preprocesando conjuntos Train y Test de forma segura...")
X_train, X_test = limpieza.imputar_nulos(X_train, X_test)
X_train, X_test = limpieza.escalar_caracteristicas(X_train, X_test, metodo='standard')

# 5. Auditoría QA de los datos
print("\n[3] Ejecutando Auditoría QA de DS Guardian...")
pasa_auditoria = auditoria.revisar_datos_finales(X_train, y=y_train)

if pasa_auditoria:
    # 6. Entrenar y Evaluar el Modelo
    print("\n[4] Entrenando el modelo RandomForest...")
    modelo = RandomForestClassifier(max_depth=5, random_state=42)
    modelo.fit(X_train, y_train)
    
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)
    
    # 7. Graficar y guardar la matriz de confusión
    print("\n[5] Generando Gráfico de Matriz de Confusión...")
    os.makedirs('plots', exist_ok=True)
    modelos.evaluar_clasificacion(y_test, y_pred, y_prob, save_path='plots/confusion_matrix.png')
    
    # 8. Generar Reporte de Auditoría en archivo local
    print("\n[6] Escribiendo Reporte de Auditoría...")
    accuracy = (y_pred == y_test).mean()
    with open('plots/reporte_auditoria.txt', 'w') as f:
        f.write("==================================================\n")
        f.write("        REPORTE DE AUDITORÍA Y MODELADO\n")
        f.write("==================================================\n")
        f.write(f"Fecha: {pd.Timestamp.now()}\n")
        f.write(f"Estado de Auditoría: PASADA (OK)\n")
        f.write(f"Dimensiones Train: {X_train.shape}\n")
        f.write(f"Accuracy del Modelo: {accuracy:.4f}\n")
        f.write("Métricas guardadas exitosamente.\n")
    print("Reporte guardado en: plots/reporte_auditoria.txt")
    
    # Comparar rendimiento con el histórico
    auditoria.registrar_y_comparar_modelo("Workflow_Ejemplo", "Accuracy", accuracy)

print("\n=====================================================================")
print("   WORKFLOW FINALIZADO CON ÉXITO")
print("=====================================================================")
