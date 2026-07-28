from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, 
                              confusion_matrix, roc_auc_score, mean_squared_error)
from sklearn.model_selection import cross_val_score
from .exceptions import ModelAuditingError

def evaluar_clasificacion(
    y_true: Any, 
    y_pred: Any, 
    y_prob: Optional[Any] = None, 
    save_path: Optional[str] = None
) -> None:
    """
    Imprime métricas avanzadas para modelos de clasificación y grafica la matriz de confusión.
    
    Args:
        y_true: Valores reales (ground truth).
        y_pred: Valores predichos por el modelo.
        y_prob: Probabilidades predichas por el modelo (opcional).
        save_path: Ruta de archivo opcional para guardar el heatmap de la matriz.
        
    Raises:
        ModelAuditingError: Si los inputs tienen dimensiones inconsistentes.
    """
    if len(y_true) != len(y_pred):
        raise ModelAuditingError("Los conjuntos y_true e y_pred tienen longitudes distintas.")
        
    print("--- Evaluación de Clasificación ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    
    if y_prob is not None:
        try:
            y_prob_arr = np.array(y_prob)
            y_true_arr = np.array(y_true)
            # Si es multiclase, necesita 'ovo' o 'ovr'
            if len(np.unique(y_true_arr)) > 2:
                auc = roc_auc_score(y_true_arr, y_prob_arr, multi_class='ovr')
            else:
                # Si y_prob es 2D (Nx2), tomar la clase positiva
                if len(y_prob_arr.shape) > 1 and y_prob_arr.shape[1] == 2:
                    y_prob_arr = y_prob_arr[:, 1]
                auc = roc_auc_score(y_true_arr, y_prob_arr)
            print(f"ROC-AUC Score: {auc:.4f}")
        except Exception as e:
            print(f"No se pudo calcular ROC-AUC: {e}")
            
    print("\nReporte de Clasificación:\n", classification_report(y_true, y_pred))
    
    # Graficar Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.tight_layout()
    
    if save_path:
        import os
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Matriz de confusión guardada en: {save_path}")
        plt.close()
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"Advertencia: No se pudo mostrar la matriz interactiva ({e}). Considera usar 'save_path'.")
            plt.close()

def evaluar_regresion(y_true: Any, y_pred: Any) -> None:
    """
    Imprime métricas avanzadas para modelos de regresión (MSE, RMSE, MAE y R2).
    
    Args:
        y_true: Valores reales.
        y_pred: Valores predichos.
        
    Raises:
        ModelAuditingError: Si los inputs tienen dimensiones inconsistentes.
    """
    if len(y_true) != len(y_pred):
        raise ModelAuditingError("Los conjuntos y_true e y_pred tienen longitudes distintas.")
        
    from sklearn.metrics import mean_absolute_error, r2_score
    print("--- Evaluación de Regresión ---")
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

def validacion_cruzada(
    modelo: Any, 
    X: Any, 
    y: Any, 
    cv: int = 5, 
    scoring: str = 'accuracy'
) -> np.ndarray:
    """
    Ejecuta K-Fold Cross Validation y muestra los resultados en consola.
    
    Args:
        modelo: Estimador de scikit-learn.
        X: Características (features).
        y: Variable objetivo (target).
        cv: Número de splits/pliegues.
        scoring: Métrica de evaluación.
        
    Returns:
        Arreglo con los scores por cada pliegue.
    """
    print(f"--- Validación Cruzada ({cv} Folds) ---")
    print(f"Métrica: {scoring}")
    scores = cross_val_score(modelo, X, y, cv=cv, scoring=scoring)
    
    print(f"Scores por fold: {np.round(scores, 4)}")
    print(f"Media (Mean): {scores.mean():.4f}")
    print(f"Desv. Est. (Std): {scores.std():.4f}")
    return scores

def optimizar_hiperparametros(
    modelo: Any, 
    param_grid: Dict[str, Any], 
    X: Any, 
    y: Any, 
    cv: int = 3, 
    n_iter: int = 10, 
    scoring: str = 'accuracy'
) -> Any:
    """
    Busca los mejores hiperparámetros usando RandomizedSearchCV de forma robusta.
    
    Args:
        modelo: Estimador de scikit-learn.
        param_grid: Grilla/Distribuciones de parámetros.
        X: Características (features).
        y: Variable objetivo (target).
        cv: Splits para la validación cruzada interna.
        n_iter: Cantidad de iteraciones aleatorias.
        scoring: Métrica para optimizar.
        
    Returns:
        El estimador ajustado con los mejores parámetros.
    """
    from sklearn.model_selection import RandomizedSearchCV
    print("\n--- Iniciando Ajuste de Hiperparámetros (Random Search) ---")
    search = RandomizedSearchCV(
        modelo, param_distributions=param_grid, n_iter=n_iter, 
        cv=cv, scoring=scoring, random_state=42, n_jobs=-1
    )
    search.fit(X, y)
    print(f"Mejor score ({scoring}): {search.best_score_:.4f}")
    print(f"Mejores parámetros: {search.best_params_}")
    return search.best_estimator_

def graficar_importancia_caracteristicas(
    modelo: Any,
    feature_names: List[str],
    top_n: int = 15,
    save_path: Optional[str] = None
) -> Any:
    """
    Calcula y grafica la importancia de las características de un modelo entrenado.
    
    Args:
        modelo: Estimador entrenado que disponga de `feature_importances_` o `coef_`.
        feature_names: Nombres de las características/columnas.
        top_n: Número máximo de características a mostrar.
        save_path: Ruta opcional para guardar el gráfico.
        
    Returns:
        DataFrame ordenado con las características y sus importancias.
    """
    import pandas as pd
    
    if hasattr(modelo, "feature_importances_"):
        importances = modelo.feature_importances_
    elif hasattr(modelo, "coef_"):
        importances = np.abs(modelo.coef_).mean(axis=0) if modelo.coef_.ndim > 1 else np.abs(modelo.coef_[0])
    else:
        raise ModelAuditingError("El modelo proporcionado no posee atributos 'feature_importances_' ni 'coef_'.")
        
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    df_top = df_imp.head(top_n)
    
    plt.figure(figsize=(8, max(4, int(top_n * 0.35))))
    sns.barplot(data=df_top, x='Importance', y='Feature', hue='Feature', palette='Blues_r', legend=False)
    plt.title(f'Top {min(top_n, len(df_imp))} Importancia de Características')
    plt.xlabel('Importancia Relativa')
    plt.ylabel('Característica / Variable')
    plt.tight_layout()
    
    if save_path:
        import os
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Gráfico de importancia guardado en: {save_path}")
        plt.close()
    else:
        try:
            plt.show()
        except Exception:
            plt.close()
            
    return df_imp

