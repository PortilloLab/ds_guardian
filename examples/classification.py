import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ds_guardian import eda, limpieza, modelos, auditoria, configurar_estilo

def run_classification_pipeline():
    print("\n--- EJEMPLO: Clasificación con DS Guardian ---")
    np.random.seed(42)
    n_samples = 500

    # Crear datos
    df = pd.DataFrame({
        'Edad': np.random.randint(18, 60, n_samples),
        'Score': np.random.normal(500, 100, n_samples),
        'Categoria': np.random.choice(['Premium', 'Basic', 'Free'], n_samples),
        'Target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })

    # Inyectar algunos nulos
    df.loc[10:30, 'Edad'] = np.nan
    df.loc[15:20, 'Categoria'] = np.nan

    # 1. Separar Train y Test
    X = df.drop('Target', axis=1)
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Imputar nulos de forma segura
    X_train, X_test = limpieza.imputar_nulos(X_train, X_test)

    # 3. Codificar variables categóricas
    X_train, X_test = limpieza.codificar_variables(X_train, X_test)

    # 4. Escalar características
    X_train, X_test = limpieza.escalar_caracteristicas(X_train, X_test, metodo='standard')

    # 5. Auditar datos de entrenamiento
    if auditoria.revisar_datos_finales(X_train, y=y_train):
        modelo = RandomForestClassifier(max_depth=5, random_state=42)
        modelo.fit(X_train, y_train)
        
        # Predicción
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)

        # 6. Evaluar
        modelos.evaluar_clasificacion(y_test, y_pred, y_prob, save_path='plots/ejemplo_clasificacion_matriz.png')
        
        # 7. Registrar
        accuracy = (y_pred == y_test).mean()
        auditoria.registrar_y_comparar_modelo("Ejemplo_Clasificacion", "Accuracy", accuracy)
        
        # Gráficos
        configurar_estilo(theme='light')
        visualizacion.plot_importancia_caracteristicas(modelo, list(X_train.columns), save_path='plots/ejemplo_importancia.png')

if __name__ == "__main__":
    run_classification_pipeline()
