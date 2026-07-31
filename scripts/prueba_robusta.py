import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Importar el nuevo framework
from ds_guardian import eda, limpieza, visualizacion, modelos, auditoria

print("\n========================================================")
print("   INICIANDO PRUEBA DEL FRAMEWORK: DS GUARDIAN")
print("========================================================\n")

# 1. Crear dataset de prueba sucio y desbalanceado
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'Edad': np.random.randint(15, 65, n_samples),
    'Salario': np.random.normal(50000, 15000, n_samples),
    'Departamento': np.random.choice(['Ventas', 'IT', 'RRHH'], n_samples),
    # Inyectamos fuerte desbalance de clases (85% activos, 15% inactivos)
    'Activo': np.random.choice([0, 1], n_samples, p=[0.15, 0.85])
})

# Inyectar nulos
df.loc[10:50, 'Edad'] = np.nan
df.loc[100:120, 'Departamento'] = np.nan

# Inyectar outliers extremos en Salario
df.loc[0:5, 'Salario'] = [500000, 600000, 550000, 800000, 900000, -100000]

print("--- 1. Exploración y Tratamiento de Outliers (EDA) ---")
df = eda.optimizar_memoria(df)
eda.resumir_datos(df)
eda.missing_values_table(df)

print("\nOutliers detectados originalmente:")
eda.detectar_outliers_iqr(df)

# Tratamiento de outliers (Winsorization/Capping)
print("\nAplicando Winsorization (Capping) de outliers en 'Salario'...")
df = eda.acotar_outliers_iqr(df, columnas=['Salario'])
print("Outliers después del tratamiento:")
eda.detectar_outliers_iqr(df)

print("\n--- 2. Limpieza e Imputación (Prevención de Leakage) ---")
X = df.drop('Activo', axis=1)
y = df['Activo']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputar nulos de forma aislada
X_train_imp, X_test_imp = limpieza.imputar_nulos(X_train, X_test)

# Codificar categóricas alineando Train y Test
X_train_enc, X_test_enc = limpieza.codificar_variables(X_train_imp, X_test_imp)

# Escalar características numéricas de forma segura
X_train_scaled, X_test_scaled = limpieza.escalar_caracteristicas(X_train_enc, X_test_enc, metodo='standard')

print("\n--- 3. Auditoría de Datos de Entrada (DS Guardian QA) ---")
# Probamos el agente revisor pasándole también el target para comprobar desbalances y leakage
auditoria_pasada = auditoria.revisar_datos_finales(X_train_scaled, y=y_train)

if auditoria_pasada:
    print("\n--- 4. Modelado y Optimización de Hiperparámetros ---")
    
    # Modelo base
    modelo_base = RandomForestClassifier(random_state=42)
    
    # Parámetros para tuning
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # Optimización automática con Cross Validation interna
    mejor_modelo = modelos.optimizar_hiperparametros(modelo_base, param_grid, X_train_scaled, y_train, cv=3, n_iter=6)
    
    # Ajuste final
    mejor_modelo.fit(X_train_scaled, y_train)
    
    # Predicción y evaluación
    y_pred = mejor_modelo.predict(X_test_scaled)
    y_prob = mejor_modelo.predict_proba(X_test_scaled)
    
    print("\n--- 5. Evaluación de Desempeño ---")
    modelos.evaluar_clasificacion(y_test, y_pred, y_prob, save_path='plots/confusion_matrix.png')
    modelos.validacion_cruzada(mejor_modelo, X_train_scaled, y_train, cv=5)
    
    # Guardar métrica en historial y comparar con intentos previos
    accuracy = (y_pred == y_test).mean()
    auditoria.registrar_y_comparar_modelo('DS_Guardian_Prueba', 'Accuracy', accuracy)

    # Configurar estilo visual premium (Tema Oscuro) y guardar gráficos
    print("\n--- 6. Generando Gráficos Premium ---")
    visualizacion.configurar_estilo(theme='dark')
    visualizacion.plot_importancia_caracteristicas(mejor_modelo, list(X_train_scaled.columns), save_path='plots/feature_importances.png')
    visualizacion.plot_distribucion(df, 'Salario', save_path='plots/salario_distribucion.png')
    visualizacion.plot_correlacion(df, save_path='plots/matriz_correlacion.png')

print("\n========================================================")
print("   PRUEBA DE DS GUARDIAN CONCLUIDA EXITOSAMENTE")
print("========================================================\n")
