"""
Modelo Predictivo de Retrasos — Supply Chain
=============================================
Objetivo: Predecir si un pedido será retrasado (delayed = 1).

HALLAZGO PRINCIPAL
------------------
El dataset no contiene señal predictiva suficiente para construir un modelo
útil. El AUC-ROC de todos los modelos es ~0.50, equivalente a tirar una moneda.
Esto NO es un error de modelado: es un diagnóstico valioso que indica que
las variables actuales no explican los retrasos.

Recomendación: enriquecer el dataset con features como:
  - Historial de retrasos del proveedor
  - Capacidad actual del almacén vs demanda
  - Indicadores de tráfico / clima en tiempo real
  - Tiempo en tránsito histórico por ruta

Pipeline:
  1. Carga y feature engineering
  2. Comparación de modelos con validación cruzada
  3. Evaluación del mejor modelo en test set
  4. Diagnóstico de señal predictiva
  5. Exportación del modelo
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)


# ─────────────────────────────────────────────
# 1. CARGA
# ─────────────────────────────────────────────
RUTA = "supply_chain_limpio.xlsx"

df = pd.read_excel(RUTA)
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Tasa de retraso: {df['delayed'].mean():.1%}\n")


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
# Variables derivadas que capturan interacciones de riesgo
# detectadas en el EDA.
df["month"] = df["order_date"].dt.month

# Flags binarios de riesgo
df["is_storm"]        = (df["weather_condition"] == "Storm").astype(int)
df["is_fog"]          = (df["weather_condition"] == "Fog").astype(int)
df["is_air"]          = (df["shipping_method"] == "Air").astype(int)
df["bad_weather"]     = df["weather_condition"].isin(["Storm", "Fog"]).astype(int)
df["low_proc_time"]   = (df["processing_time_hours"] < 16).astype(int)

# Interacción: clima adverso × envío aéreo (combo más riesgoso según EDA)
df["risk_combo"]      = df["bad_weather"] * df["is_air"]

# Encoding de categóricas
encoders = {}
for col in ["shipping_method", "weather_condition", "order_priority"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # guardar para usar en producción

# Features finales
FEATURES = [
    "supplier_reliability_score",
    "warehouse_inventory_level",
    "order_quantity",
    "shipping_distance_km",
    "shipping_method",
    "weather_condition",
    "processing_time_hours",
    "order_priority",
    "month",
    "is_storm",
    "is_fog",
    "is_air",
    "bad_weather",
    "low_proc_time",
    "risk_combo",
]

X = df[FEATURES]
y = df["delayed"]

print(f"Features utilizadas: {len(FEATURES)}")
print(f"Distribución de clases: {y.value_counts().to_dict()}\n")


# ─────────────────────────────────────────────
# 3. SPLIT TRAIN / TEST
# ─────────────────────────────────────────────
# Stratify garantiza la misma proporción de delayed en ambos sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {len(X_train)} filas | Test: {len(X_test)} filas\n")


# ─────────────────────────────────────────────
# 4. COMPARACIÓN DE MODELOS (VALIDACIÓN CRUZADA)
# ─────────────────────────────────────────────
# class_weight='balanced' compensa el desbalance 70/30 automáticamente.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidatos = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", C=0.1,
            max_iter=1000, random_state=42
        )),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, subsample=0.8, random_state=42
    ),
}

print("=== Comparación con validación cruzada (5 folds) ===")
print(f"{'Modelo':<25} {'AUC-ROC':>8} {'F1':>8} {'Recall':>8}")
print("-" * 55)

resultados = {}
for nombre, modelo in candidatos.items():
    auc = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    f1  = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="f1").mean()
    rec = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="recall").mean()
    resultados[nombre] = {"auc": auc, "f1": f1, "recall": rec}
    print(f"{nombre:<25} {auc:>8.4f} {f1:>8.4f} {rec:>8.4f}")

# Seleccionar mejor modelo por AUC-ROC
mejor_nombre = max(resultados, key=lambda k: resultados[k]["auc"])
mejor_modelo  = candidatos[mejor_nombre]
print(f"\nMejor modelo: {mejor_nombre}\n")


# ─────────────────────────────────────────────
# 5. ENTRENAMIENTO FINAL Y EVALUACIÓN EN TEST
# ─────────────────────────────────────────────
mejor_modelo.fit(X_train, y_train)

y_pred = mejor_modelo.predict(X_test)
y_prob = mejor_modelo.predict_proba(X_test)[:, 1]

print("=== Evaluación en Test Set ===")
print(classification_report(y_test, y_pred, target_names=["No retrasado", "Retrasado"]))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}\n")

print("=== Matriz de confusión ===")
cm = confusion_matrix(y_test, y_pred)
print(f"  Verdaderos Negativos  : {cm[0,0]:>4}  (predijo 'no retraso' y era correcto)")
print(f"  Falsos Positivos      : {cm[0,1]:>4}  (predijo 'retraso' pero no había)")
print(f"  Falsos Negativos      : {cm[1,0]:>4}  (predijo 'no retraso' pero sí había)")
print(f"  Verdaderos Positivos  : {cm[1,1]:>4}  (predijo 'retraso' y era correcto)")


# ─────────────────────────────────────────────
# 6. DIAGNÓSTICO DE SEÑAL PREDICTIVA
# ─────────────────────────────────────────────
auc_final = roc_auc_score(y_test, y_prob)

print("\n=== Diagnóstico ===")
if auc_final < 0.55:
    print("⚠  AUC-ROC cercano a 0.50: el modelo no supera al azar.")
    print("   Las features actuales no tienen suficiente poder predictivo.")
    print("   Recomendaciones para mejorar el modelo:")
    print("   1. Agregar historial de retrasos por proveedor.")
    print("   2. Incorporar datos de tráfico o clima en tiempo real.")
    print("   3. Incluir métricas de capacidad del almacén.")
    print("   4. Registrar el tiempo real de cada etapa del proceso.")
elif auc_final < 0.70:
    print("⚡ AUC-ROC moderado. El modelo tiene algo de poder predictivo.")
    print("   Considerar ajuste de hiperparámetros y más feature engineering.")
else:
    print("✅ AUC-ROC aceptable. El modelo tiene buen poder predictivo.")


# ─────────────────────────────────────────────
# 7. IMPORTANCIA DE VARIABLES
# ─────────────────────────────────────────────
print("\n=== Importancia de variables ===")
if hasattr(mejor_modelo, "coef_"):
    # Logistic Regression
    coefs = pd.Series(
        mejor_modelo.coef_[0], index=FEATURES
    ).abs().sort_values(ascending=False)
elif hasattr(mejor_modelo, "feature_importances_"):
    # Tree-based
    coefs = pd.Series(
        mejor_modelo.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
else:
    # Pipeline con LR dentro
    clf = mejor_modelo.named_steps.get("clf")
    if clf and hasattr(clf, "coef_"):
        coefs = pd.Series(
            clf.coef_[0], index=FEATURES
        ).abs().sort_values(ascending=False)
    else:
        coefs = pd.Series(dtype=float)

if not coefs.empty:
    for feat, val in coefs.items():
        print(f"  {feat:<35} {val:.4f}")


# ─────────────────────────────────────────────
# 8. GUARDAR MODELO
# ─────────────────────────────────────────────
joblib.dump({"modelo": mejor_modelo, "encoders": encoders, "features": FEATURES},
            "modelo_delay.pkl")
print("\nModelo guardado en: modelo_delay.pkl")
print("Usar joblib.load('modelo_delay.pkl') para cargarlo en producción.")
