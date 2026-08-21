import joblib
import json
import os
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from .logging import get_logger

logger = get_logger(__name__)

def evaluar_clasificacion(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """Calcula métricas completas de clasificación."""
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    accuracy = report['accuracy']
    
    auc = None
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                auc = float(roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob))
            else:
                auc = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
        except Exception:
            pass
            
    logger.info(f"Clasificación — Accuracy: {accuracy:.4f} | ROC-AUC: {auc if auc else 'N/A'}")
    return {"accuracy": accuracy, "auc": auc if auc else 0.0}

def evaluar_regresion(y_true, y_pred) -> Dict[str, float]:
    """Calcula métricas completas de regresión."""
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    logger.info(f"Regresión — R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return {"r2": r2, "rmse": rmse, "mae": mae, "mse": mse}

def optimizar_hiperparametros(modelo, param_grid: Dict[str, Any], X_train, y_train, cv=3, n_iter=10):
    """Búsqueda aleatoria de hiperparámetros (RandomizedSearchCV)."""
    search = RandomizedSearchCV(modelo, param_distributions=param_grid, n_iter=n_iter, cv=cv, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    logger.info(f"Mejores Parámetros encontrados: {search.best_params_}")
    return search.best_estimator_

def exportar_modelo(modelo, filepath: str = "modelo_ds_guardian.joblib", metadata: Optional[Dict[str, Any]] = None) -> str:
    """Exporta el modelo entrenado con metadatos asociados."""
    joblib.dump(modelo, filepath)
    if metadata:
        meta_path = filepath.replace(".joblib", "_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
    logger.info(f"Modelo guardado exitosamente en: {filepath}")
    return filepath
