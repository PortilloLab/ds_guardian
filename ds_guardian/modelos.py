import joblib
import json
import os
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from .logging import get_logger

logger = get_logger(__name__)

def evaluar_clasificacion(y_true, y_pred, y_prob=None) -> Dict[str, float]:
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

def validacion_cruzada_temporal(modelo, X, y, n_splits: int = 5) -> np.ndarray:
    """Validación cruzada respetando la secuencia temporal (TimeSeriesSplit)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(modelo, X, y, cv=tscv, scoring='accuracy')
    logger.info(f"Validación Cruzada Temporal ({n_splits} splits): Mean Score = {scores.mean():.4f} +/- {scores.std():.4f}")
    return scores

def exportar_modelo(modelo, filepath: str = "modelo_ds_guardian.joblib", metadata: Optional[Dict[str, Any]] = None) -> str:
    joblib.dump(modelo, filepath)
    if metadata:
        meta_path = filepath.replace(".joblib", "_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
    logger.info(f"Modelo exportado en: {filepath}")
    return filepath
