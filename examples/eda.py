import pandas as pd
import numpy as np
from ds_guardian import eda

def run_eda_example():
    print("\n--- EJEMPLO: Análisis Exploratorio de Datos (EDA) con DS Guardian ---")
    
    # Crear DataFrame sucio y de gran tamaño conceptual para optimizar
    np.random.seed(42)
    n_samples = 10000
    
    df = pd.DataFrame({
        'Id_Cliente': np.arange(n_samples),
        'Edad': np.random.randint(18, 90, n_samples).astype(float),
        'Salario': np.random.normal(50000, 15000, n_samples),
        'Status': np.random.choice(['Active', 'Inactive', None], n_samples)
    })
    
    # Inyectar algunos valores extremos (outliers)
    df.loc[0:20, 'Salario'] = df.loc[0:20, 'Salario'] * 12
    
    # 1. Optimizar memoria RAM
    df = eda.optimizar_memoria(df)
    
    # 2. Resumir estadísticas de forma segura
    eda.resumir_datos(df)
    
    # 3. Mostrar tabla de valores nulos
    nulos = eda.missing_values_table(df)
    print("\nTabla de nulos:")
    print(nulos)
    
    # 4. Detectar outliers con IQR
    eda.detectar_outliers_iqr(df)
    
    # 5. Aplicar capping sobre outliers de Salario
    print("\nAplicando capping sobre outliers en columna 'Salario'...")
    df_capped = eda.acotar_outliers_iqr(df, columnas=['Salario'])
    
    print("\nOutliers tras aplicar capping:")
    eda.detectar_outliers_iqr(df_capped)

if __name__ == "__main__":
    run_eda_example()
