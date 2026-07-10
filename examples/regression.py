import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from ds_guardian import eda, limpieza, modelos, auditoria

def run_regression_pipeline():
    print("\n--- EJEMPLO: Regresión con DS Guardian ---")
    np.random.seed(42)
    n_samples = 300

    # Crear datos
    df = pd.DataFrame({
        'Superficie': np.random.normal(70, 20, n_samples),
        'Habitaciones': np.random.randint(1, 5, n_samples),
        'Precio': np.zeros(n_samples)
    })
    # Definir target con outliers inyectados
    df['Precio'] = df['Superficie'] * 1500 + df['Habitaciones'] * 5000 + np.random.normal(0, 10000, n_samples)
    df.loc[15:20, 'Precio'] = df.loc[15:20, 'Precio'] * 10  # Outliers

    # 1. Capping de outliers en el target
    df = eda.acotar_outliers_iqr(df, columnas=['Precio'])

    X = df.drop('Precio', axis=1)
    y = df['Precio']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Imputar e iniciar preprocesamiento
    X_train, X_test = limpieza.imputar_nulos(X_train, X_test)
    X_train, X_test = limpieza.escalar_caracteristicas(X_train, X_test, metodo='minmax')

    # 3. Auditar
    if auditoria.revisar_datos_finales(X_train, y=y_train):
        modelo = Ridge()
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        # 4. Evaluar regresión
        modelos.evaluar_regresion(y_test, y_pred)

if __name__ == "__main__":
    run_regression_pipeline()
