import json
import os
from typing import Dict, Any, Optional, List
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from .exceptions import ModelAuditingError
from .logging import get_logger

logger = get_logger(__name__)


def _validate_lengths(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ModelAuditingError("Las longitudes de y_true y y_pred no coinciden.")


def evaluar_clasificacion(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    _validate_lengths(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    accuracy = report['accuracy']
    auc = None
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                auc = float(roc_auc_score(y_true, y_prob[:, 1] if getattr(y_prob, 'ndim', 1) > 1 else y_prob))
            else:
                auc = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
        except Exception:
            pass
    logger.info(f"Clasificación — Accuracy: {accuracy:.4f} | ROC-AUC: {auc if auc else 'N/A'}")
    return {"accuracy": accuracy, "auc": auc if auc else 0.0}


def evaluar_regresion(y_true, y_pred) -> Dict[str, float]:
    _validate_lengths(y_true, y_pred)
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
    return metrics


def validacion_cruzada(modelo, X, y, cv=5, scoring='accuracy') -> np.ndarray:
    scores = cross_val_score(modelo, X, y, cv=cv, scoring=scoring)
    logger.info(f"Validación Cruzada ({cv} folds): Mean Score = {scores.mean():.4f} +/- {scores.std():.4f}")
    return scores


def validacion_cruzada_temporal(modelo, X, y, n_splits: int = 5) -> np.ndarray:
    """Validación cruzada respetando la secuencia temporal (TimeSeriesSplit)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(modelo, X, y, cv=tscv, scoring='accuracy')
    logger.info(f"Validación Cruzada Temporal ({n_splits} splits): Mean Score = {scores.mean():.4f} +/- {scores.std():.4f}")
    return scores


def optimizar_hiperparametros(modelo, param_grid, X, y, cv=3, n_iter=10, scoring='accuracy', random_state: int = 42):
    search = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=None,
    )
    search.fit(X, y)
    logger.info(f"Mejor score: {search.best_score_:.4f}, mejor params: {search.best_params_}")
    return search.best_estimator_


def graficar_importancia_caracteristicas(modelo, feature_names: Optional[List[str]] = None, save_path: Optional[str] = None) -> pd.DataFrame:
    if not hasattr(modelo, 'feature_importances_'):
        raise ModelAuditingError("El modelo no expone feature_importances_.")
    importances = getattr(modelo, 'feature_importances_')
    names = feature_names or [f'Feature_{i}' for i in range(len(importances))]
    df_imp = pd.DataFrame({"Feature": names, "Importance": importances})
    df_imp = df_imp.sort_values("Importance", ascending=False).reset_index(drop=True)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.barh(df_imp['Feature'], df_imp['Importance'])
        plt.title('Importancia de Características')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    return df_imp


def guardar_modelo_entrenado(modelo, filepath: str = "modelo_ds_guardian.joblib", metadata: Optional[Dict[str, Any]] = None) -> str:
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    joblib.dump(modelo, filepath)
    if metadata:
        meta_path = filepath.replace(".joblib", "_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
    logger.info(f"Modelo exportado en: {filepath}")
    return filepath


def cargar_modelo_entrenado(filepath: str):
    return joblib.load(filepath)


def exportar_modelo(modelo, filepath: str = "modelo_ds_guardian.joblib", metadata: Optional[Dict[str, Any]] = None) -> str:
    return guardar_modelo_entrenado(modelo, filepath=filepath, metadata=metadata)
