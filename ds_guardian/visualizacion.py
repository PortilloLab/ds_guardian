from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def configurar_estilo(tema: str = 'dark'):
    if tema == 'dark':
        plt.style.use('dark_background')
        sns.set_theme(style='darkgrid', palette='dark')
    else:
        sns.set_theme(style='whitegrid', palette='muted')


def plot_distribucion(df: pd.DataFrame, columna: str, figsize=(8, 5), save_path: Optional[str] = None):
    configurar_estilo('dark')
    plt.figure(figsize=figsize)
    sns.histplot(df[columna], kde=True)
    plt.title(f"Distribución de {columna}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return df


def plot_categorica(df: pd.DataFrame, columna: str, target: Optional[str] = None, figsize=(8, 5), save_path: Optional[str] = None):
    configurar_estilo('dark')
    plt.figure(figsize=figsize)
    if target is not None and target in df.columns:
        sns.boxplot(data=df, x=columna, y=target)
    else:
        df[columna].value_counts().plot(kind='bar')
    plt.title(f"Variable categórica: {columna}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return df


def plot_correlacion(df: pd.DataFrame, figsize=(10, 8), save_path: Optional[str] = None):
    configurar_estilo('dark')
    plt.figure(figsize=figsize)
    num_df = df.select_dtypes(include=['number'])
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("DS Guardian — Matriz de Correlación")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return num_df.corr()


def plot_importancia_caracteristicas(df: pd.DataFrame, figsize=(8, 5), save_path: Optional[str] = None):
    configurar_estilo('dark')
    plt.figure(figsize=figsize)
    if 'Feature' in df.columns and 'Importance' in df.columns:
        df_sorted = df.sort_values('Importance', ascending=False)
        sns.barplot(data=df_sorted, x='Importance', y='Feature', palette='viridis')
    else:
        raise ValueError("El DataFrame debe contener 'Feature' e 'Importance'.")
    plt.title("Importancia de variables")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return df


def plot_curva_roc(y_true, y_prob, figsize=(8, 6), save_path: Optional[str] = None):
    from sklearn.metrics import roc_curve, auc
    configurar_estilo('dark')
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return roc_auc
