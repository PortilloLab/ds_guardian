import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def configurar_estilo(tema: str = 'dark'):
    if tema == 'dark':
        plt.style.use('dark_background')
        sns.set_theme(style='darkgrid', palette='dark')
    else:
        sns.set_theme(style='whitegrid', palette='muted')

def plot_correlacion(df: pd.DataFrame, figsize=(10, 8)):
    configurar_estilo('dark')
    plt.figure(figsize=figsize)
    num_df = df.select_dtypes(include=['number'])
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("DS Guardian — Matriz de Correlación")
    plt.tight_layout()
    plt.show()
