from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def configurar_estilo(theme: str = 'light') -> None:
    """
    Configura un estilo agradable y premium para los gráficos.
    
    Args:
        theme: Nombre del tema ('light' o 'dark').
    """
    if theme == 'dark':
        sns.set_theme(style="darkgrid", palette="mako")
        plt.rcParams.update({
            'figure.facecolor': '#1e1e1e',
            'axes.facecolor': '#2d2d2d',
            'text.color': '#ffffff',
            'axes.labelcolor': '#ffffff',
            'xtick.color': '#cccccc',
            'ytick.color': '#cccccc'
        })
    else:
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({
            'figure.facecolor': '#ffffff',
            'axes.facecolor': '#f8f9fa',
            'text.color': '#333333',
            'axes.labelcolor': '#333333',
            'xtick.color': '#555555',
            'ytick.color': '#555555'
        })
    plt.rcParams['figure.figsize'] = (10, 6)

def _guardar_o_mostrar(save_path: Optional[str] = None) -> None:
    """Helper interno para guardar el gráfico o mostrarlo controlando entornos headless."""
    if save_path:
        # Crear directorios si no existen
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Gráfico guardado en: {save_path}")
        plt.close()
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"Advertencia: No se pudo mostrar el gráfico interactivo ({e}). Considera usar 'save_path'.")
            plt.close()

def plot_distribucion(df: pd.DataFrame, columna: str, save_path: Optional[str] = None) -> None:
    """
    Grafica la distribución de una variable numérica usando Seaborn.
    
    Args:
        df: DataFrame de pandas.
        columna: Nombre de la columna numérica.
        save_path: Ruta de archivo opcional para guardar el gráfico (ej: 'plots/dist.png').
    """
    try:
        if df is None or df.empty:
            print("Error: DataFrame vacío o None.")
            return
            
        if df[columna].isnull().all():
            print(f"Error: La columna '{columna}' está completamente vacía (nulos).")
            return
            
        plt.figure()
        sns.histplot(data=df, x=columna, kde=True)
        plt.title(f'Distribución de {columna}')
        plt.tight_layout()
        _guardar_o_mostrar(save_path)
    except Exception as e:
        print(f"Error al graficar distribución de '{columna}': {e}")

def plot_categorica(
    df: pd.DataFrame, 
    columna: str, 
    max_categorias: int = 20, 
    save_path: Optional[str] = None
) -> None:
    """
    Grafica el conteo de frecuencias para una variable categórica.
    
    Args:
        df: DataFrame de pandas.
        columna: Nombre de la columna categórica o discreta.
        max_categorias: Límite máximo de categorías únicas a mostrar.
        save_path: Ruta de archivo opcional para guardar el gráfico.
    """
    try:
        if df is None or df.empty:
            print("Error: DataFrame vacío o None.")
            return
            
        if df[columna].nunique() > max_categorias:
            print(f"Advertencia: '{columna}' tiene demasiadas categorías únicas ({df[columna].nunique()}). Mostrando top {max_categorias}.")
            orden = df[columna].value_counts().iloc[:max_categorias].index
        else:
            orden = df[columna].value_counts().index
            
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, y=columna, order=orden, palette='viridis')
        plt.title(f'Frecuencia de {columna}')
        plt.tight_layout()
        _guardar_o_mostrar(save_path)
    except Exception as e:
        print(f"Error al graficar categórica '{columna}': {e}")

def plot_correlacion(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Grafica el mapa de calor de las correlaciones solo para variables numéricas.
    
    Args:
        df: DataFrame de pandas.
        save_path: Ruta de archivo opcional para guardar el gráfico.
    """
    try:
        if df is None or df.empty:
            print("Error: DataFrame vacío o None.")
            return
            
        df_num = df.select_dtypes(include=[np.number])
        if df_num.empty:
            print("Error: No hay variables numéricas para correlacionar.")
            return
            
        plt.figure(figsize=(12, 8))
        corr = df_num.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        _guardar_o_mostrar(save_path)
    except Exception as e:
        print(f"Error al graficar correlación: {e}")

def plot_importancia_caracteristicas(
    modelo,
    feature_names: List[str],
    max_features: int = 15,
    save_path: Optional[str] = None
) -> None:
    """
    Grafica la importancia de características de un modelo entrenado
    (ej: RandomForest, o modelos lineales con `coef_`).

    Nota: esta función es un wrapper sobre
    `ds_guardian.modelos.graficar_importancia_caracteristicas`, que concentra
    la lógica real de cálculo y ploteo (evita mantener dos implementaciones
    duplicadas). Se mantiene acá por compatibilidad con el nombre documentado
    históricamente en este módulo.

    Args:
        modelo: Estimador de sklearn entrenado con atributo 'feature_importances_'
            o 'coef_'.
        feature_names: Lista de nombres de las características.
        max_features: Cantidad máxima de columnas a graficar.
        save_path: Ruta de archivo opcional para guardar el gráfico.
    """
    from .modelos import graficar_importancia_caracteristicas
    from .exceptions import ModelAuditingError

    try:
        graficar_importancia_caracteristicas(
            modelo,
            feature_names=feature_names,
            top_n=max_features,
            save_path=save_path,
        )
    except ModelAuditingError as e:
        print(f"Error al graficar importancia de características: {e}")
