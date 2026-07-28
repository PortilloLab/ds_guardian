# 🛡️ Reporte de Gobernanza y Auditoría de Datos – DS Guardian

**Proyecto:** Kaggle Steel Plates Faults  
**Fecha de Generación:** 2026-07-27 21:46:59  
**Estado QA:** ✅ APROBADO PARA PRODUCCIÓN  

---

## 📊 1. Resumen de la Estructura de Datos

| Métrica | Valor |
| :--- | :--- |
| **Total de Filas / Muestras** | 1941 |
| **Total de Columnas** | 23 |
| **Valores Nulos Restantes** | 0 |
| **Uso de Memoria RAM** | 0.15 MB |

---

## 🕵️ 2. Diagnóstico del Agente de Auditoría QA

* **Control de Nulos:** ✅ Sin nulos
* **Tipos de Datos:** ✅ Numéricos y codificados en One-Hot / Label Encoding
* **Fuga de Datos (Data Leakage):** ✅ Sin correlación perfecta con la variable objetivo
* **Multicolinealidad:** ✅ 5 columnas redundantes eliminadas

---

## 🤖 3. Métricas de Evaluación del Modelo

| Métrica de Rendimiento | Valor Obtenido |
| :--- | :--- |
| **Accuracy (Exactitud)** | 0.7757 |
| **ROC-AUC Score (Multiclase)** | 0.9636 |
| **F1-Score Macro** | 0.7932 |

---

## 💡 4. Explicabilidad del Modelo (Top Características)

| Ranking | Variable / Característica | Importancia Relativa |
| :---: | :--- | :---: |
| 1 | `Length_of_Conveyer` | 7.86% |
| 2 | `LogOfAreas` | 7.26% |
| 3 | `Pixels_Areas` | 7.19% |
| 4 | `X_Minimum` | 6.22% |
| 5 | `Steel_Plate_Thickness` | 5.68% |
| 6 | `Outside_X_Index` | 5.40% |
| 7 | `Minimum_of_Luminosity` | 5.19% |
| 8 | `Orientation_Index` | 4.95% |
| 9 | `Log_X_Index` | 4.81% |
| 10 | `SigmoidOfAreas` | 4.78% |

---
*Generado automáticamente por el Framework DS Guardian – AI Data Science Governance System.*
